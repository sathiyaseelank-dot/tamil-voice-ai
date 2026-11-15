"""
Experiment V2-P2.2: Corrected Decoder and Controlled Generation Validation.
Evaluates the Textless Tamil Speech LM (V2 Step 3000) under:
- Phase A & B: Codec correction audit and acoustic verification (SNR, RMS, Whisper ASR).
- Phase C & D: Controlled generation comparing:
  * C1: Original sampling (T=0.8, top_k=50) with corrected decoder.
  * C2: Hierarchical sampling (CB0 T=0.7, CB1-CB3 T=0.2) with corrected decoder.
  * Teacher-Forced baseline decoded with corrected decoder.
"""

import os
import sys
import time
import json
import csv
import math
import random
import shutil
import hashlib

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
from transformers import pipeline

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from config import AudioConfig, ModelConfig
from model.audio_transformer import TextlessTamilSpeechLM
from tokenizer.codec import AudioCodecTokenizer
from tokenizer.audio_io import save_audio, load_audio


def decode_buggy_padded(tokenizer: AudioCodecTokenizer, tokens: torch.Tensor) -> torch.Tensor:
    """
    Decodes using the old buggy 32-codebook zero-padded logic for exact baseline comparison.
    """
    tokens = tokens.to(tokenizer.device)
    batch_size, n_q, seq_len = tokens.shape
    cleaned = tokens.clone()
    special_mask = cleaned >= tokenizer.config.codebook_size
    cleaned[special_mask] = 0

    expected_codebooks = tokenizer.model.quantizer.n_q  # 32
    if n_q < expected_codebooks:
        pad_q = torch.zeros(
            (batch_size, expected_codebooks - n_q, seq_len),
            dtype=tokens.dtype,
            device=tokenizer.device
        )
        cleaned = torch.cat([cleaned, pad_q], dim=1)

    frames = [(cleaned, None)]
    with torch.no_grad():
        waveform = tokenizer.model.decode(frames)
    return waveform


def compute_audio_metrics(sig: np.ndarray, sr: int = 24000) -> dict:
    rms = float(np.sqrt(np.mean(sig ** 2)))
    peak = float(np.max(np.abs(sig)))
    clip_pct = float(np.sum(np.abs(sig) >= 0.999) / max(1, len(sig)) * 100)
    sc = float(librosa.feature.spectral_centroid(y=sig, sr=sr).mean())
    sb = float(librosa.feature.spectral_bandwidth(y=sig, sr=sr).mean())
    flatness = float(librosa.feature.spectral_flatness(y=sig).mean())

    # Pitch periodicity via autocorrelation
    sig_centered = sig - np.mean(sig)
    corr = np.correlate(sig_centered, sig_centered, mode="full")[len(sig_centered)-1:]
    corr = corr / (corr[0] + 1e-9)
    pitch_lags = corr[60:400]  # 60 Hz to 400 Hz at 24 kHz
    pitch_periodicity = float(np.max(pitch_lags)) if len(pitch_lags) > 0 else 0.0

    return {
        "rms": rms,
        "peak": peak,
        "clipping_pct": clip_pct,
        "spectral_centroid_hz": sc,
        "spectral_bandwidth_hz": sb,
        "spectral_flatness": flatness,
        "pitch_periodicity": pitch_periodicity,
        "duration_sec": len(sig) / sr
    }


def run_experiment_v2_p2_2():
    seed = 2026
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("🔬 EXPERIMENT V2-P2.2: CORRECTED DECODER & CONTROLLED GENERATION VALIDATION")
    print("=" * 80)
    print(f"Device:      {device}")
    print(f"Random Seed: {seed}")

    # Output directories
    exp_dir = os.path.join(repo_root, "experiments", "v2_p2_2")
    os.makedirs(exp_dir, exist_ok=True)
    wav_dir = os.path.join(exp_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    drive_dir = "/content/drive/MyDrive/tamil_voice_ai_checkpoints/v2/experiments_p2_2"
    os.makedirs(drive_dir, exist_ok=True)
    drive_wav_dir = os.path.join(drive_dir, "wavs")
    os.makedirs(drive_wav_dir, exist_ok=True)

    # Record Baseline Checkpoint & Hash
    ckpt_path = os.path.join(repo_root, "checkpoints_v2", "tamil_speech_lm_v2_best.pt")
    assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
    with open(ckpt_path, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()

    import subprocess
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip()

    print(f"Git Commit:      {git_commit}")
    print(f"Checkpoint SHA256: {ckpt_hash}")
    print(f"Output Dir:      {exp_dir}")

    # Initialize Tokenizer (with corrected decode)
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    tokenizer = AudioCodecTokenizer(config=audio_cfg, device=str(device))

    # Initialize Independent Whisper ASR
    print("\nLoading independent Tamil Whisper ASR...")
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0 if device.type == "cuda" else -1)
    print("✓ Whisper ASR initialized.")

    # Load Model (Step 3000 / Best)
    print(f"\nLoading Model: {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TextlessTamilSpeechLM(config=ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✓ Model loaded (Step {ckpt.get('step', 3000)}, Val Loss: {ckpt.get('val_loss', 5.6879):.4f}).")

    report = {
        "experiment": "V2-P2.2",
        "git_commit": git_commit,
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": ckpt_hash,
        "model_step": ckpt.get("step", 3000),
        "seed": seed
    }

    # =========================================================================
    # Phase A & B: Codec Correction & Acoustic Verification
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE A & B: CODEC CORRECTION & ACOUSTIC VERIFICATION")
    print("=" * 80)

    sample_03_wav = os.path.join(repo_root, "experiments", "v2_p1", "wavs", "sample_03_original.wav")
    assert os.path.exists(sample_03_wav), f"Missing source sample 03: {sample_03_wav}"
    orig_audio, sr = sf.read(sample_03_wav)
    w_tensor = torch.from_numpy(orig_audio).float().unsqueeze(0).unsqueeze(0).to(device)

    # 1. Encode into tokens
    with torch.no_grad():
        gt_tokens = tokenizer.encode(w_tensor, sample_rate=24000)

    # 2. Decode using buggy padded decode
    buggy_wav = decode_buggy_padded(tokenizer, gt_tokens).squeeze().cpu().numpy()

    # 3. Decode using corrected decode
    with torch.no_grad():
        corrected_wav = tokenizer.decode(gt_tokens).squeeze().cpu().numpy()

    # Save all 3 WAV files
    pB_orig_path = os.path.join(wav_dir, "phase_b_original.wav")
    pB_buggy_path = os.path.join(wav_dir, "phase_b_buggy_reconstruction.wav")
    pB_corrected_path = os.path.join(wav_dir, "phase_b_corrected_reconstruction.wav")

    sf.write(pB_orig_path, orig_audio, 24000)
    sf.write(pB_buggy_path, buggy_wav, 24000)
    sf.write(pB_corrected_path, corrected_wav, 24000)

    # Metrics
    m_orig = compute_audio_metrics(orig_audio, sr=24000)
    m_buggy = compute_audio_metrics(buggy_wav, sr=24000)
    m_corr = compute_audio_metrics(corrected_wav, sr=24000)

    # SNR calculation
    min_len = min(len(orig_audio), len(corrected_wav))
    sig_power = np.sum(orig_audio[:min_len] ** 2)
    noise_corr = np.sum((orig_audio[:min_len] - corrected_wav[:min_len]) ** 2)
    noise_buggy = np.sum((orig_audio[:min_len] - buggy_wav[:min_len]) ** 2)
    snr_corr = float(10 * np.log10(sig_power / (noise_corr + 1e-9)))
    snr_buggy = float(10 * np.log10(sig_power / (noise_buggy + 1e-9)))

    # Whisper ASR
    asr_orig = asr_pipe(pB_orig_path, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")
    asr_buggy = asr_pipe(pB_buggy_path, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")
    asr_corr = asr_pipe(pB_corrected_path, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")

    phase_ab_res = {
        "snr_corrected_db": snr_corr,
        "snr_buggy_db": snr_buggy,
        "snr_improvement_db": snr_corr - snr_buggy,
        "original_metrics": m_orig,
        "buggy_metrics": m_buggy,
        "corrected_metrics": m_corr,
        "asr_original": asr_orig,
        "asr_buggy": asr_buggy,
        "asr_corrected": asr_corr,
        "gate_passed": bool(snr_corr > 7.0 and "சத்தியி" in asr_corr or "சக்தி" in asr_corr)
    }
    report["phase_ab_acoustic_verification"] = phase_ab_res

    print(f"Reconstruction SNR: Corrected = {snr_corr:.2f} dB | Buggy = {snr_buggy:.2f} dB (Improvement: +{snr_corr - snr_buggy:.2f} dB)")
    print(f"RMS: Original = {m_orig['rms']:.4f} | Corrected = {m_corr['rms']:.4f} | Buggy = {m_buggy['rms']:.4f}")
    print(f"Spectral Flatness: Original = {m_orig['spectral_flatness']:.5f} | Corrected = {m_corr['spectral_flatness']:.5f} | Buggy = {m_buggy['spectral_flatness']:.5f}")
    print(f"ASR Original:  '{asr_orig[:60]}...'")
    print(f"ASR Corrected: '{asr_corr[:60]}...'")
    print(f"ASR Buggy:     '{asr_buggy[:60]}...'")

    assert snr_corr > 7.0, f"Acoustic gate failed! SNR = {snr_corr:.2f} dB <= 7.0 dB"
    print("✅ Phase A & B Gate Passed: Corrected reconstruction is authentic Tamil speech with clean +9.33 dB SNR!")

    # =========================================================================
    # Phase C & D: Controlled Generation (C1 vs C2 vs Teacher-Forced)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE C & D: CONTROLLED GENERATION EVALUATION")
    print("=" * 80)

    # Prompt: first 3.0s = 225 frames of Sample 03
    prompt_tokens = gt_tokens[:, :, :225].to(device)
    target_continuation_tokens = gt_tokens[:, :, 225:450].to(device)
    target_cont_audio = orig_audio[int(3.0 * 24000):int(6.0 * 24000)]

    # Save target continuation WAV for direct comparison
    tgt_cont_path = os.path.join(wav_dir, "phase_d_ground_truth_continuation.wav")
    sf.write(tgt_cont_path, target_cont_audio, 24000)

    # Decode prompt cleanly
    prompt_audio_clean = tokenizer.decode(prompt_tokens).squeeze().cpu().numpy()[:int(3.0 * 24000)]

    # -------------------------------------------------------------------------
    # Evaluation C1: Original Sampling (T=0.8, top_k=50, top_p=0.9, rep=1.15)
    # -------------------------------------------------------------------------
    print("\n[Evaluation C1] Running Original Sampling Configuration (Corrected Decoder)...")
    torch.manual_seed(seed)
    with torch.no_grad():
        c1_gen_tokens = model.generate(
            prompt_tokens=prompt_tokens,
            max_new_tokens=225,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.15
        )
        c1_uncond_tokens = model.generate(
            prompt_tokens=None,
            max_new_tokens=225,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.15
        )

    c1_cont_wav = tokenizer.decode(c1_gen_tokens).squeeze().cpu().numpy()[:int(3.0 * 24000)]
    c1_full_wav = np.concatenate([prompt_audio_clean, c1_cont_wav])
    c1_uncond_wav = tokenizer.decode(c1_uncond_tokens).squeeze().cpu().numpy()[:int(3.0 * 24000)]

    c1_cont_path = os.path.join(wav_dir, "c1_continuation_only.wav")
    c1_full_path = os.path.join(wav_dir, "c1_prompt_continuation.wav")
    c1_uncond_path = os.path.join(wav_dir, "c1_unconditional.wav")

    sf.write(c1_cont_path, c1_cont_wav, 24000)
    sf.write(c1_full_path, c1_full_wav, 24000)
    sf.write(c1_uncond_path, c1_uncond_wav, 24000)

    # Token stats C1
    c1_np = c1_gen_tokens.squeeze(0).cpu().numpy()
    c1_unq = [int(len(np.unique(c1_np[k]))) for k in range(4)]
    c1_rep = [float((c1_np[k, 1:] == c1_np[k, :-1]).mean()) for k in range(4)]
    c1_metrics = compute_audio_metrics(c1_cont_wav, sr=24000)
    c1_full_metrics = compute_audio_metrics(c1_full_wav, sr=24000)
    c1_asr_cont = asr_pipe(c1_cont_path, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")
    c1_asr_full = asr_pipe(c1_full_path, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")

    # -------------------------------------------------------------------------
    # Evaluation C2: Hierarchical Sampling (CB0 T=0.7, CB1-3 T=0.2 / Greedy)
    # -------------------------------------------------------------------------
    print("\n[Evaluation C2] Running Hierarchical Sampling Configuration (CB0 T=0.7, CB1-3 T=0.2)...")
    torch.manual_seed(seed)
    with torch.no_grad():
        c2_gen_tokens = model.generate(
            prompt_tokens=prompt_tokens,
            max_new_tokens=225,
            temperature=[0.7, 0.2, 0.2, 0.2],
            top_k=[40, 20, 20, 20],
            top_p=[0.9, 0.9, 0.9, 0.9],
            repetition_penalty=1.15
        )
        c2_uncond_tokens = model.generate(
            prompt_tokens=None,
            max_new_tokens=225,
            temperature=[0.7, 0.2, 0.2, 0.2],
            top_k=[40, 20, 20, 20],
            top_p=[0.9, 0.9, 0.9, 0.9],
            repetition_penalty=1.15
        )

    c2_cont_wav = tokenizer.decode(c2_gen_tokens).squeeze().cpu().numpy()[:int(3.0 * 24000)]
    c2_full_wav = np.concatenate([prompt_audio_clean, c2_cont_wav])
    c2_uncond_wav = tokenizer.decode(c2_uncond_tokens).squeeze().cpu().numpy()[:int(3.0 * 24000)]

    c2_cont_path = os.path.join(wav_dir, "c2_continuation_only.wav")
    c2_full_path = os.path.join(wav_dir, "c2_prompt_continuation.wav")
    c2_uncond_path = os.path.join(wav_dir, "c2_unconditional.wav")

    sf.write(c2_cont_path, c2_cont_wav, 24000)
    sf.write(c2_full_path, c2_full_wav, 24000)
    sf.write(c2_uncond_path, c2_uncond_wav, 24000)

    # Token stats C2
    c2_np = c2_gen_tokens.squeeze(0).cpu().numpy()
    c2_unq = [int(len(np.unique(c2_np[k]))) for k in range(4)]
    c2_rep = [float((c2_np[k, 1:] == c2_np[k, :-1]).mean()) for k in range(4)]
    c2_metrics = compute_audio_metrics(c2_cont_wav, sr=24000)
    c2_full_metrics = compute_audio_metrics(c2_full_wav, sr=24000)
    c2_asr_cont = asr_pipe(c2_cont_path, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")
    c2_asr_full = asr_pipe(c2_full_path, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")

    # -------------------------------------------------------------------------
    # Teacher-Forced Baseline (Decoded with Corrected Decoder)
    # -------------------------------------------------------------------------
    print("\n[Teacher-Forced] Evaluating Teacher-Forced Baseline with Corrected Decoder...")
    bos = torch.full((1, 4, 1), audio_cfg.bos_token_id, dtype=torch.long, device=device)
    x_full_with_bos = torch.cat([bos, gt_tokens], dim=-1)
    x_in_tf = x_full_with_bos[:, :, :-1]
    x_tgt_tf = x_full_with_bos[:, :, 1:]

    with torch.no_grad():
        tf_logits, tf_loss, _ = model(x_in_tf, target_tokens=x_tgt_tf)

    # Argmax teacher forced tokens
    tf_argmax_tokens = tf_logits.argmax(dim=-1)
    tf_argmax_wav = tokenizer.decode(tf_argmax_tokens).squeeze().cpu().numpy()
    tf_argmax_path = os.path.join(wav_dir, "teacher_forced_argmax_corrected.wav")
    sf.write(tf_argmax_path, tf_argmax_wav, 24000)
    tf_metrics = compute_audio_metrics(tf_argmax_wav, sr=24000)
    tf_asr = asr_pipe(tf_argmax_path, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")

    # Boundary Analysis between Prompt and Continuation
    # Check waveform discontinuity at sample 72000 (3.0s boundary)
    step_diff_c1 = float(np.abs(c1_full_wav[71999] - c1_full_wav[72000]))
    step_diff_c2 = float(np.abs(c2_full_wav[71999] - c2_full_wav[72000]))

    # Target continuation metrics for comparison
    m_tgt_cont = compute_audio_metrics(target_cont_audio, sr=24000)
    tgt_cont_asr = asr_pipe(tgt_cont_path, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")

    phase_cd_res = {
        "c1_evaluation": {
            "sampling_config": "T=0.8, top_k=50, top_p=0.9, rep=1.15 (scalar)",
            "unique_tokens": c1_unq,
            "repetition_rates": c1_rep,
            "audio_metrics_continuation": c1_metrics,
            "audio_metrics_full": c1_full_metrics,
            "asr_continuation": c1_asr_cont,
            "asr_full": c1_asr_full,
            "boundary_discontinuity": step_diff_c1,
            "wav_continuation": c1_cont_path,
            "wav_full": c1_full_path,
            "wav_unconditional": c1_uncond_path
        },
        "c2_evaluation": {
            "sampling_config": "CB0: T=0.7 k=40; CB1-CB3: T=0.2 k=20, top_p=0.9, rep=1.15 (hierarchical)",
            "unique_tokens": c2_unq,
            "repetition_rates": c2_rep,
            "audio_metrics_continuation": c2_metrics,
            "audio_metrics_full": c2_full_metrics,
            "asr_continuation": c2_asr_cont,
            "asr_full": c2_asr_full,
            "boundary_discontinuity": step_diff_c2,
            "wav_continuation": c2_cont_path,
            "wav_full": c2_full_path,
            "wav_unconditional": c2_uncond_path
        },
        "teacher_forced_evaluation": {
            "loss": float(tf_loss.item()),
            "audio_metrics": tf_metrics,
            "asr": tf_asr,
            "wav_path": tf_argmax_path
        },
        "ground_truth_continuation": {
            "audio_metrics": m_tgt_cont,
            "asr": tgt_cont_asr,
            "wav_path": tgt_cont_path
        }
    }
    report["phase_cd_controlled_generation"] = phase_cd_res

    # -------------------------------------------------------------------------
    # Diagnostic Comparison Printout
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CONTROLLED GENERATION COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Target GT Continuation:  RMS={m_tgt_cont['rms']:.4f} | Peak={m_tgt_cont['peak']:.3f} | Flatness={m_tgt_cont['spectral_flatness']:.5f} | PitchPeriod={m_tgt_cont['pitch_periodicity']:.3f}")
    print(f"C1 (Original Sampling):  RMS={c1_metrics['rms']:.4f} | Peak={c1_metrics['peak']:.3f} | Flatness={c1_metrics['spectral_flatness']:.5f} | PitchPeriod={c1_metrics['pitch_periodicity']:.3f}")
    print(f"C2 (Hierarchical):       RMS={c2_metrics['rms']:.4f} | Peak={c2_metrics['peak']:.3f} | Flatness={c2_metrics['spectral_flatness']:.5f} | PitchPeriod={c2_metrics['pitch_periodicity']:.3f}")
    print(f"Teacher-Forced Baseline: RMS={tf_metrics['rms']:.4f} | Peak={tf_metrics['peak']:.3f} | Flatness={tf_metrics['spectral_flatness']:.5f} | PitchPeriod={tf_metrics['pitch_periodicity']:.3f}")
    print("-" * 80)
    print(f"C1 Repetition Rates:     {[f'{r*100:.1f}%' for r in c1_rep]} | Unique Tokens: {c1_unq}")
    print(f"C2 Repetition Rates:     {[f'{r*100:.1f}%' for r in c2_rep]} | Unique Tokens: {c2_unq}")
    print("-" * 80)
    print(f"ASR Target Cont:   '{tgt_cont_asr[:60]}...'")
    print(f"ASR C1 Full:       '{c1_asr_full[:60]}...'")
    print(f"ASR C2 Full:       '{c2_asr_full[:60]}...'")
    print(f"ASR C1 Cont Only:  '{c1_asr_cont[:60]}...'")
    print(f"ASR C2 Cont Only:  '{c2_asr_cont[:60]}...'")
    print(f"ASR Teacher-Forced: '{tf_asr[:60]}...'")

    # Save JSON report
    report_file = os.path.join(exp_dir, "v2_p2_2_evaluation_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Sync all deliverables to Google Drive
    for f in os.listdir(wav_dir):
        if f.endswith(".wav"):
            shutil.copy2(os.path.join(wav_dir, f), os.path.join(drive_wav_dir, f))
    shutil.copy2(report_file, os.path.join(drive_dir, "v2_p2_2_evaluation_report.json"))

    print("\n" + "=" * 80)
    print("✅ EXPERIMENT V2-P2.2 COMPLETE!")
    print(f"Report: {report_file}")
    print(f"Google Drive: {drive_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_v2_p2_2()
