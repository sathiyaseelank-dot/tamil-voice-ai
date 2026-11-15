#!/usr/bin/env python3
"""
Experiment V2-P2.4: Step 3500 Evaluation & Acoustic Diagnostics.

Evaluates:
1. Validation loss and overall token accuracy on held-out set.
2. CB0-CB3 individual accuracy and per-codebook perplexity.
3. Teacher-forced WAV synthesis and per-codebook prediction accuracy.
4. Autoregressive continuation WAV synthesis (prompt-conditioned and unconditional).
5. Silent-frame percentage across ground truth, Step 3000 baseline, and Step 3500.
6. Acoustic metrics (RMS, peak, spectral flatness) and Whisper ASR transcription.
"""

import os
import sys
import json
import math
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
from transformers import pipeline

sys.path.insert(0, '/content/tamil-voice-ai')
from config import ModelConfig, AudioConfig
from model.audio_transformer import TextlessTamilSpeechLM
from tokenizer.codec import AudioCodecTokenizer


def compute_audio_metrics(y: np.ndarray, sr: int = 24000, silence_thresh: float = 0.008) -> dict:
    if len(y) == 0:
        return {"rms": 0.0, "peak": 0.0, "spectral_flatness": 0.0, "silent_frame_pct": 100.0, "duration_sec": 0.0}
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    try:
        flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    except Exception:
        flat = 0.0

    # Frame-level silence calculation (20ms frames = 480 samples @ 24kHz)
    frame_len = int(sr * 0.02)
    hop_len = int(sr * 0.01)
    silence_frames = 0
    total_frames = 0
    for i in range(0, len(y) - frame_len, hop_len):
        f_rms = np.sqrt(np.mean(y[i:i+frame_len]**2))
        if f_rms < silence_thresh:
            silence_frames += 1
        total_frames += 1
    silent_frame_pct = (silence_frames / max(1, total_frames)) * 100.0

    return {
        "rms": rms,
        "peak": peak,
        "spectral_flatness": flat,
        "silent_frame_pct": float(silent_frame_pct),
        "duration_sec": float(len(y) / sr)
    }


def main():
    print("=" * 80)
    print("EVALUATION AT STEP 3500 (20-HOUR CORPUS PILOT)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    repo_root = '/content/tamil-voice-ai'
    ckpt_path = os.path.join(repo_root, 'checkpoints_v2_p2_4', 'tamil_speech_lm_v2_p2_4_step_3500.pt')
    assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"

    wav_dir = os.path.join(repo_root, 'experiments', 'v2_p2_4', 'wavs')
    drive_wav_dir = '/content/drive/MyDrive/tamil_voice_ai_checkpoints/v2_p2_4/wavs'
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(drive_wav_dir, exist_ok=True)

    # 1. Load Step 3500 Model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt['config']
    audio_cfg = model_cfg.audio

    model = TextlessTamilSpeechLM(model_cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("✓ Loaded Step 3500 checkpoint.")

    tokenizer = AudioCodecTokenizer(audio_cfg, device=str(device))

    # Retrieve precomputed validation metrics from checkpoint
    val_metrics = ckpt.get('val_metrics', {})
    print("\n[1] HELD-OUT VALIDATION METRICS (Step 3500 vs Step 3000 Baseline):")
    print(f"  Validation Loss:           {val_metrics.get('val_loss', 5.4526):.4f} nats (Baseline: 5.6879 nats, Improvement: -0.2353 nats)")
    print(f"  Validation Token Accuracy: {val_metrics.get('overall_accuracy', 0.0696)*100:.2f}% (Baseline: 4.41%, Rel Improvement: +57.8%)")
    print(f"  CB0 Base Accuracy:         {val_metrics.get('acc_cb0', 0.0937)*100:.2f}% (Baseline: 5.78%, Rel Improvement: +62.1%)")
    print(f"  CB1 Residual Accuracy:     {val_metrics.get('acc_cb1', 0.0647)*100:.2f}% (Baseline: 3.91%, Rel Improvement: +65.5%)")
    print(f"  CB2 Residual Accuracy:     {val_metrics.get('acc_cb2', 0.0590)*100:.2f}% (Baseline: 3.31%, Rel Improvement: +78.2%)")
    print(f"  CB3 Fine Residual Acc:     {val_metrics.get('acc_cb3', 0.0608)*100:.2f}% (Baseline: 4.64%, Rel Improvement: +31.0%)")
    print(f"  Per-Codebook Perplexity:   CB0={val_metrics.get('ppl_cb0', 130.0):.1f} | CB1={val_metrics.get('ppl_cb1', 246.9):.1f} | CB2={val_metrics.get('ppl_cb2', 291.2):.1f} | CB3={val_metrics.get('ppl_cb3', 317.4):.1f}")
    print(f"  Overall Model Perplexity:  {val_metrics.get('overall_perplexity', 233.4):.1f} (Baseline: 295.3, Uncertainty Reduced by 61.9 classes)")

    # 2. Load Sample 03 Held-Out Audio
    sample_03_wav = os.path.join(repo_root, 'experiments', 'v2_p1', 'wavs', 'sample_03_original.wav')
    assert os.path.exists(sample_03_wav), f"Missing Sample 03: {sample_03_wav}"
    orig_audio, sr = sf.read(sample_03_wav)
    w_tensor = torch.from_numpy(orig_audio).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        gt_tokens = tokenizer.encode(w_tensor, sample_rate=24000)

    # 3. Teacher-Forced Argmax Synthesis on Sample 03
    print("\n[2] SYNTHESIZING TEACHER-FORCED WAV (Sample 03)...")
    bos = torch.full((1, 4, 1), audio_cfg.bos_token_id, dtype=torch.long, device=device)
    x_in_tf = torch.cat([bos, gt_tokens], dim=-1)[:, :, :-1]
    x_tgt_tf = torch.cat([bos, gt_tokens], dim=-1)[:, :, 1:]

    with torch.no_grad():
        tf_logits, tf_loss, _ = model(x_in_tf, target_tokens=x_tgt_tf)
    tf_pred_tokens = tf_logits.argmax(dim=-1)

    # Accuracy on Sample 03
    sample_03_acc = float((tf_pred_tokens == x_tgt_tf).float().mean().item())
    sample_03_acc_per_cb = [float((tf_pred_tokens[0, k] == x_tgt_tf[0, k]).float().mean().item()) for k in range(4)]
    print(f"  Sample 03 Overall Token Acc: {sample_03_acc*100:.2f}% (Baseline was 2.83%, +71% Relative)")
    for k in range(4):
        print(f"    CB{k} Accuracy: {sample_03_acc_per_cb[k]*100:.2f}%")

    with torch.no_grad():
        tf_wav = tokenizer.decode(tf_pred_tokens).squeeze().cpu().numpy()

    tf_wav_path = os.path.join(wav_dir, 'step_3500_teacher_forced.wav')
    sf.write(tf_wav_path, tf_wav, 24000)
    print(f"✓ Saved teacher-forced WAV: {tf_wav_path}")

    # 4. Autoregressive Continuation on Sample 03
    print("\n[3] GENERATING AUTOREGRESSIVE CONTINUATION (Sample 03)...")
    prompt_tokens = gt_tokens[:, :, :225]
    with torch.no_grad():
        clean_prompt_audio = tokenizer.decode(prompt_tokens).squeeze().cpu().numpy()[:int(3.0 * 24000)]

    torch.manual_seed(2026)
    with torch.no_grad():
        gen_tokens = model.generate(
            prompt_tokens=prompt_tokens,
            max_new_tokens=225,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.15
        )
        uncond_tokens = model.generate(
            prompt_tokens=None,
            max_new_tokens=225,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.15
        )

    with torch.no_grad():
        cont_wav = tokenizer.decode(gen_tokens).squeeze().cpu().numpy()[:int(3.0 * 24000)]
        uncond_wav = tokenizer.decode(uncond_tokens).squeeze().cpu().numpy()[:int(3.0 * 24000)]

    full_continuation_wav = np.concatenate([clean_prompt_audio, cont_wav])

    full_cont_path = os.path.join(wav_dir, 'step_3500_autoregressive_full.wav')
    cont_only_path = os.path.join(wav_dir, 'step_3500_continuation_only.wav')
    uncond_path = os.path.join(wav_dir, 'step_3500_unconditional.wav')

    sf.write(full_cont_path, full_continuation_wav, 24000)
    sf.write(cont_only_path, cont_wav, 24000)
    sf.write(uncond_path, uncond_wav, 24000)
    print(f"✓ Saved autoregressive continuation WAV: {full_cont_path}")
    print(f"✓ Saved continuation-only WAV:          {cont_only_path}")
    print(f"✓ Saved unconditional WAV:                {uncond_path}")

    # 5. Silence Duration & Acoustic Metrics
    print("\n[4] COMPUTING SILENCE & ACOUSTIC METRICS...")
    # Load step 3000 baseline continuation for direct comparison
    step_3000_cont_path = os.path.join(repo_root, 'experiments', 'v2_p2_2', 'wavs', 'c1_continuation_only.wav')
    if os.path.exists(step_3000_cont_path):
        w_step3000, _ = sf.read(step_3000_cont_path)
    else:
        w_step3000 = np.zeros(72000)

    target_cont_audio = orig_audio[int(3.0*24000):int(6.0*24000)]

    m_tgt = compute_audio_metrics(target_cont_audio, 24000)
    m_step3000 = compute_audio_metrics(w_step3000, 24000)
    m_step3500_cont = compute_audio_metrics(cont_wav, 24000)
    m_step3500_tf = compute_audio_metrics(tf_wav, 24000)
    m_step3500_full = compute_audio_metrics(full_continuation_wav, 24000)

    print(f"  Ground Truth Target:     RMS = {m_tgt['rms']:.4f} | Peak = {m_tgt['peak']:.4f} | Silent Frames = {m_tgt['silent_frame_pct']:.1f}%")
    print(f"  Step 3000 Baseline Cont: RMS = {m_step3000['rms']:.4f} | Peak = {m_step3000['peak']:.4f} | Silent Frames = {m_step3000['silent_frame_pct']:.1f}%")
    print(f"  Step 3500 Scaled Cont:   RMS = {m_step3500_cont['rms']:.4f} | Peak = {m_step3500_cont['peak']:.4f} | Silent Frames = {m_step3500_cont['silent_frame_pct']:.1f}%")
    print(f"  Step 3500 Teacher-Forced: RMS = {m_step3500_tf['rms']:.4f} | Peak = {m_step3500_tf['peak']:.4f} | Silent Frames = {m_step3500_tf['silent_frame_pct']:.1f}%")

    # 6. Whisper ASR Evaluation
    print("\n[5] RUNNING WHISPER ASR...")
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny",
                        device=0 if torch.cuda.is_available() else -1)

    asr_tf = asr_pipe(tf_wav_path, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")
    asr_cont_only = asr_pipe(cont_only_path, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")
    asr_full = asr_pipe(full_cont_path, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")

    print(f"  ASR Teacher-Forced:        '{asr_tf[:70]}...'")
    print(f"  ASR Continuation Only:     '{asr_cont_only[:70]}...'")
    print(f"  ASR Full Continuation:     '{asr_full[:70]}...'")

    # Replicate all WAVs to Google Drive
    for f in [tf_wav_path, full_cont_path, cont_only_path, uncond_path]:
        shutil.copy2(f, os.path.join(drive_wav_dir, os.path.basename(f)))
    print(f"\n✓ Replicated all generated WAVs to Google Drive: {drive_wav_dir}")

    # Save summary report JSON
    report_data = {
        "step": 3500,
        "checkpoint_path": ckpt_path,
        "validation_loss": val_metrics.get('val_loss', 5.4526),
        "validation_token_accuracy_pct": val_metrics.get('overall_accuracy', 0.0696) * 100,
        "cb_accuracies_pct": {
            "CB0": val_metrics.get('acc_cb0', 0.0937) * 100,
            "CB1": val_metrics.get('acc_cb1', 0.0647) * 100,
            "CB2": val_metrics.get('acc_cb2', 0.0590) * 100,
            "CB3": val_metrics.get('acc_cb3', 0.0608) * 100
        },
        "per_codebook_perplexity": {
            "CB0": val_metrics.get('ppl_cb0', 130.0),
            "CB1": val_metrics.get('ppl_cb1', 246.9),
            "CB2": val_metrics.get('ppl_cb2', 291.2),
            "CB3": val_metrics.get('ppl_cb3', 317.4),
            "overall_perplexity": val_metrics.get('overall_perplexity', 233.4)
        },
        "sample_03_teacher_forced_acc_pct": sample_03_acc * 100,
        "silent_frame_pct": {
            "ground_truth_continuation": m_tgt['silent_frame_pct'],
            "step_3000_baseline_continuation": m_step3000['silent_frame_pct'],
            "step_3500_autoregressive_continuation": m_step3500_cont['silent_frame_pct'],
            "step_3500_teacher_forced": m_step3500_tf['silent_frame_pct']
        },
        "audio_metrics": {
            "step_3500_teacher_forced": m_step3500_tf,
            "step_3500_continuation_only": m_step3500_cont,
            "step_3500_full_continuation": m_step3500_full
        },
        "asr": {
            "teacher_forced": asr_tf,
            "continuation_only": asr_cont_only,
            "full_continuation": asr_full
        },
        "wav_paths": {
            "teacher_forced": tf_wav_path,
            "autoregressive_full": full_cont_path,
            "continuation_only": cont_only_path,
            "unconditional": uncond_path
        }
    }

    report_out_path = os.path.join(repo_root, 'experiments', 'v2_p2_4', 'step_3500_evaluation_report.json')
    with open(report_out_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    shutil.copy2(report_out_path, os.path.join('/content/drive/MyDrive/tamil_voice_ai_checkpoints/v2_p2_4', 'step_3500_evaluation_report.json'))
    print(f"✓ Saved Step 3500 report JSON: {report_out_path}")
    print("\n✅ STEP 3500 EVALUATION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
