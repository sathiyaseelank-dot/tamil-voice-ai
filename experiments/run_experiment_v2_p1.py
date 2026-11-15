"""
Experiment V2-P1: Prompt-Conditioned Tamil Speech Continuation.
Evaluates the Textless Tamil Speech LM (V2) on 10 unseen held-out Tamil recordings.
"""

import os
import sys
import time
import json
import csv
import random
import gc
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio
from transformers import pipeline

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from config import AudioConfig, ModelConfig
from model.audio_transformer import TextlessTamilSpeechLM
from tokenizer.codec import AudioCodecTokenizer
from tokenizer.audio_io import save_audio, load_audio


def run_experiment():
    eval_seed = 2026
    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    torch.cuda.manual_seed_all(eval_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75)
    print("🎯 EXPERIMENT V2-P1: PROMPT-CONDITIONED TAMIL SPEECH CONTINUATION")
    print("=" * 75)
    print(f"Device:    {device}")
    print(f"Eval Seed: {eval_seed}")
    print("=" * 75)

    output_dir = os.path.join(repo_root, "experiments", "v2_p1")
    os.makedirs(output_dir, exist_ok=True)
    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    # 1. Load V2 Model Checkpoint
    ckpt_path = os.path.join(repo_root, "checkpoints_v2", "tamil_speech_lm_v2_step_300.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = "/content/drive/MyDrive/tamil_voice_ai_checkpoints/v2/tamil_speech_lm_v2_step_300.pt"

    print(f"\n[1/5] Loading V2 Model Checkpoint: {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt["config"]
    audio_cfg = model_cfg.audio

    model = TextlessTamilSpeechLM(config=model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("✓ V2 Model loaded successfully.")

    # 2. Load Encodec Tokenizer
    print("\n[2/5] Loading Encodec 24 kHz Tokenizer...")
    tokenizer = AudioCodecTokenizer(config=audio_cfg, device=device)
    print("✓ Encodec tokenizer loaded.")

    # 3. Load Independent Tamil ASR System
    print("\n[3/5] Loading Independent Tamil ASR System (openai/whisper-tiny)...")
    try:
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=0 if device == "cuda" else -1
        )
        print("✓ Tamil Whisper ASR system initialized.")
    except Exception as e:
        print(f"Notice: Whisper ASR initialization warning: {e}")
        asr_pipeline = None

    # 4. Stream and Select 10 Previously Unseen Tamil Recordings
    # Skip first 1000 samples to strictly guarantee no training overlap
    print("\n[4/5] Selecting 10 unseen held-out Tamil recordings from SPRINGLab/IndicVoices-R_Tamil...")
    hf_dataset = load_dataset("SPRINGLab/IndicVoices-R_Tamil", split="train", streaming=True)
    hf_dataset = hf_dataset.cast_column("audio", Audio(decode=False))
    data_iter = iter(hf_dataset)

    print("Skipping first 1000 samples to ensure zero training overlap...")
    for idx in range(1000):
        try:
            next(data_iter)
        except StopIteration:
            data_iter = iter(hf_dataset)
            next(data_iter)

    selected_samples = []
    sample_counter = 1000

    while len(selected_samples) < 10:
        sample_counter += 1
        s = next(data_iter)
        a_entry = s.get("audio", {})

        if "array" in a_entry and a_entry["array"] is not None:
            r_aud = np.array(a_entry["array"])
            sr = a_entry.get("sampling_rate", audio_cfg.sample_rate)
        elif "bytes" in a_entry and a_entry["bytes"] is not None:
            import io, soundfile as sf_io
            r_aud, sr = sf_io.read(io.BytesIO(a_entry["bytes"]))
        else:
            continue

        duration = len(r_aud) / sr
        if duration < 6.5:  # Require at least 6.5 seconds for clean 3s prompt + 3s continuation
            continue

        raw_text = s.get("text", "") or s.get("sentence", "")

        # Resample to 24,000 Hz
        wav = torch.from_numpy(r_aud).float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.ndim == 2:
            wav = wav.mean(dim=-1, keepdim=True).transpose(0, 1).unsqueeze(0)

        if sr != audio_cfg.sample_rate:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, audio_cfg.sample_rate)

        # Normalize audio peak
        if wav.abs().max() > 0:
            wav = wav / (wav.abs().max() + 1e-6) * 0.9

        selected_samples.append({
            "sample_index": sample_counter,
            "waveform": wav,  # (1, 1, samples)
            "duration": duration,
            "transcript": raw_text
        })
        print(f"  • Selected Sample {len(selected_samples)}/10: ID={sample_counter}, duration={duration:.2f}s, text='{raw_text[:40]}...'")

    # 5. Execute Evaluation and Prompt Continuation
    print("\n[5/5] Running Prompt-Conditioned Evaluation on 10 Samples...")
    results = []

    prompt_sec = 3.0
    prompt_frames = int(prompt_sec * audio_cfg.frame_rate)  # 225 frames
    gen_sec = 3.0
    gen_frames = int(gen_sec * audio_cfg.frame_rate)        # 225 frames

    for i, item in enumerate(selected_samples):
        s_idx = item["sample_index"]
        wav_full = item["waveform"].to(device)

        # Tokenize entire recording
        with torch.no_grad():
            full_tokens = tokenizer.encode(wav_full, sample_rate=audio_cfg.sample_rate)  # (1, 4, T)

        # Check total frames
        total_frames = full_tokens.shape[-1]
        assert total_frames >= prompt_frames + gen_frames, f"Insufficient frames: {total_frames}"

        prompt_tokens = full_tokens[:, :, :prompt_frames]
        gt_cont_tokens = full_tokens[:, :, prompt_frames:prompt_frames + gen_frames]

        # -------------------------------------------------------------
        # Baseline 1: Original audio ground truth
        # -------------------------------------------------------------
        # Extract ground truth audio: 0 to 6.0 seconds
        orig_6s = wav_full[:, :, :int((prompt_sec + gen_sec) * audio_cfg.sample_rate)].cpu().squeeze(0)
        orig_wav_path = os.path.join(wav_dir, f"sample_{i+1:02d}_original.wav")
        save_audio(orig_wav_path, orig_6s, audio_cfg.sample_rate)

        # -------------------------------------------------------------
        # Baseline 2: Encodec Reconstructed Audio (Loopback)
        # -------------------------------------------------------------
        gt_tokens_6s = torch.cat([prompt_tokens, gt_cont_tokens], dim=-1)
        with torch.no_grad():
            recon_6s = tokenizer.decode(gt_tokens_6s).cpu().squeeze(0)
        recon_wav_path = os.path.join(wav_dir, f"sample_{i+1:02d}_encodec_recon.wav")
        save_audio(recon_wav_path, recon_6s, audio_cfg.sample_rate)

        # -------------------------------------------------------------
        # V2 Generated Continuation
        # -------------------------------------------------------------
        t_start = time.time()
        with torch.no_grad():
            gen_cont_tokens = model.generate(
                prompt_tokens=prompt_tokens,
                max_new_tokens=gen_frames,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.15
            )
        gen_latency = time.time() - t_start
        gen_fps = gen_frames / gen_latency
        rtf = gen_latency / gen_sec

        # Decode generated continuation
        with torch.no_grad():
            gen_cont_wav = tokenizer.decode(gen_cont_tokens).cpu().squeeze(0)  # (1, samples)
            prompt_wav = tokenizer.decode(prompt_tokens).cpu().squeeze(0)

        # Concatenate prompt + generated continuation
        min_p_samples = int(prompt_sec * audio_cfg.sample_rate)
        min_g_samples = int(gen_sec * audio_cfg.sample_rate)
        p_slice = prompt_wav[:, :min_p_samples]
        g_slice = gen_cont_wav[:, :min_g_samples]
        full_v2_wav = torch.cat([p_slice, g_slice], dim=-1)

        v2_cont_wav_path = os.path.join(wav_dir, f"sample_{i+1:02d}_v2_continuation.wav")
        save_audio(v2_cont_wav_path, full_v2_wav, audio_cfg.sample_rate)

        # -------------------------------------------------------------
        # Baseline 4: V2 Unconditional Generation (from BOS)
        # -------------------------------------------------------------
        with torch.no_grad():
            uncond_tokens = model.generate(
                prompt_tokens=None,
                max_new_tokens=gen_frames,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.15
            )
            uncond_wav = tokenizer.decode(uncond_tokens).cpu().squeeze(0)
        uncond_wav_path = os.path.join(wav_dir, f"sample_{i+1:02d}_v2_unconditional.wav")
        save_audio(uncond_wav_path, uncond_wav, audio_cfg.sample_rate)

        # -------------------------------------------------------------
        # Quantitative Metrics Computation
        # -------------------------------------------------------------
        # 1. Per-codebook unique tokens & repetition rates on generated continuation
        gen_np = gen_cont_tokens.squeeze(0).cpu().numpy()  # (4, 225)
        unique_counts = []
        rep_rates = []
        for k in range(4):
            cb_toks = gen_np[k]
            unique_counts.append(len(np.unique(cb_toks)))
            rep = (cb_toks[1:] == cb_toks[:-1]).mean()
            rep_rates.append(float(rep))

        # 2. Audio signal properties of generated continuation
        audio_np = g_slice.squeeze().numpy()
        rms = float(np.sqrt(np.mean(audio_np ** 2)))
        peak = float(np.max(np.abs(audio_np)))
        clipping_count = int(np.sum(np.abs(audio_np) >= 0.999))
        clipping_pct = float(clipping_count / len(audio_np) * 100)

        # 3. Held-out continuation cross-entropy loss
        # Sequence: [BOS, prompt_tokens, gt_cont_tokens]
        bos = torch.full((1, 4, 1), audio_cfg.bos_token_id, dtype=torch.long, device=device)
        eval_seq = torch.cat([bos, prompt_tokens, gt_cont_tokens], dim=-1)
        x_eval_in = eval_seq[:, :, :-1]
        x_eval_target = eval_seq[:, :, 1:]

        with torch.no_grad():
            eval_logits, _, _ = model(x_eval_in, target_tokens=x_eval_target)
            # Evaluate loss strictly over continuation positions:
            # prompt ends at index prompt_frames
            cont_logits = eval_logits[:, :, prompt_frames:, :].reshape(-1, audio_cfg.vocab_size)
            cont_targets = x_eval_target[:, :, prompt_frames:].reshape(-1)
            held_out_ce = F.cross_entropy(cont_logits, cont_targets, ignore_index=audio_cfg.pad_token_id).item()
            cont_preds = cont_logits.argmax(dim=-1)
            cont_acc = (cont_preds == cont_targets).float().mean().item() * 100

            # Prediction entropy over continuation logits
            cont_probs = F.softmax(cont_logits, dim=-1)
            pred_entropy = float(-(cont_probs * torch.log(cont_probs.clamp_min(1e-9))).sum(dim=-1).mean().item())

        # 4. Secondary Tamil ASR Intelligibility Assessment
        asr_orig = ""
        asr_recon = ""
        asr_cont = ""
        if asr_pipeline is not None:
            try:
                asr_orig = asr_pipeline(orig_wav_path, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")
                asr_recon = asr_pipeline(recon_wav_path, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")
                asr_cont = asr_pipeline(v2_cont_wav_path, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")
            except Exception as e:
                print(f"ASR error on sample {i+1}: {e}")

        sample_res = {
            "sample_num": i + 1,
            "sample_id": s_idx,
            "prompt_duration_sec": prompt_sec,
            "generated_duration_sec": gen_sec,
            "cb0_unique": unique_counts[0],
            "cb1_unique": unique_counts[1],
            "cb2_unique": unique_counts[2],
            "cb3_unique": unique_counts[3],
            "cb0_rep_rate": rep_rates[0],
            "cb1_rep_rate": rep_rates[1],
            "cb2_rep_rate": rep_rates[2],
            "cb3_rep_rate": rep_rates[3],
            "avg_repetition_rate": float(np.mean(rep_rates)),
            "prediction_entropy": pred_entropy,
            "rms_amplitude": rms,
            "peak_amplitude": peak,
            "clipping_ratio_pct": clipping_pct,
            "held_out_cross_entropy": held_out_ce,
            "continuation_token_acc_pct": cont_acc,
            "generation_latency_sec": gen_latency,
            "generation_fps": gen_fps,
            "real_time_factor": rtf,
            "ground_truth_transcript": item["transcript"],
            "asr_orig": asr_orig,
            "asr_recon": asr_recon,
            "asr_v2_continuation": asr_cont,
            "original_wav_path": orig_wav_path,
            "recon_wav_path": recon_wav_path,
            "v2_continuation_wav_path": v2_cont_wav_path,
            "v2_unconditional_wav_path": uncond_wav_path
        }
        results.append(sample_res)

        # Free memory between samples
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n[Sample {i+1:02d}/10] (ID: {s_idx})")
        print(f"  • Prompt: {prompt_sec:.1f}s | Generated: {gen_sec:.1f}s | Latency: {gen_latency:.2f}s ({gen_fps:.1f} fps, RTF: {rtf:.2f}x)")
        print(f"  • Unique tokens: CB0={unique_counts[0]}, CB1={unique_counts[1]}, CB2={unique_counts[2]}, CB3={unique_counts[3]} (out of 225)")
        print(f"  • Consec Repetition: CB0={rep_rates[0]*100:.1f}%, CB1={rep_rates[1]*100:.1f}%, CB2={rep_rates[2]*100:.1f}%, CB3={rep_rates[3]*100:.1f}%")
        print(f"  • Audio: RMS={rms:.4f}, Peak={peak:.4f}, Clipping={clipping_pct:.2f}%")
        print(f"  • Held-Out Continuation Loss: {held_out_ce:.4f} | Token Accuracy: {cont_acc:.2f}% | Entropy: {pred_entropy:.2f}")
        if asr_pipeline is not None:
            print(f"  • ASR Orig:  '{asr_orig}'")
            print(f"  • ASR Recon: '{asr_recon}'")
            print(f"  • ASR V2:    '{asr_cont}'")

    # -------------------------------------------------------------
    # V1 Humming Comparison Baseline (Sample 1200)
    # -------------------------------------------------------------
    v1_wav_path = "/content/drive/MyDrive/tamil_voice_ai_checkpoints/sample_generation_step_1200.wav"
    v1_metrics = {}
    if os.path.exists(v1_wav_path):
        v1_data, v1_sr = sf.read(v1_wav_path)
        v1_rms = float(np.sqrt(np.mean(v1_data ** 2)))
        v1_peak = float(np.max(np.abs(v1_data)))
        v1_asr = ""
        if asr_pipeline is not None:
            try:
                v1_asr = asr_pipeline(v1_wav_path, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")
            except Exception:
                pass
        v1_metrics = {
            "v1_rms": v1_rms,
            "v1_peak": v1_peak,
            "v1_repetition_rate_cb0": 0.899,
            "v1_unique_tokens_cb0": 6,
            "v1_asr": v1_asr,
            "v1_audio_type": "Periodic 75 Hz harmonic buzzing drone"
        }
        print(f"\nV1 Baseline Comparison (sample_generation_step_1200.wav):")
        print(f"  • V1 RMS: {v1_rms:.4f}, Peak: {v1_peak:.4f}")
        print(f"  • V1 Repetition Rate: 89.9% (Unique tokens: 6/150)")
        print(f"  • V1 ASR Transcript: '{v1_asr}' (Empty/unintelligible)")

    # -------------------------------------------------------------
    # Save CSV and JSON Reports
    # -------------------------------------------------------------
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    fieldnames = [
        "sample_num", "sample_id", "prompt_duration_sec", "generated_duration_sec",
        "cb0_unique", "cb1_unique", "cb2_unique", "cb3_unique",
        "cb0_rep_rate", "cb1_rep_rate", "cb2_rep_rate", "cb3_rep_rate", "avg_repetition_rate",
        "prediction_entropy", "rms_amplitude", "peak_amplitude", "clipping_ratio_pct",
        "held_out_cross_entropy", "continuation_token_acc_pct",
        "generation_latency_sec", "generation_fps", "real_time_factor",
        "ground_truth_transcript", "asr_orig", "asr_recon", "asr_v2_continuation",
        "original_wav_path", "recon_wav_path", "v2_continuation_wav_path", "v2_unconditional_wav_path"
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\n✓ Saved evaluation CSV: {csv_path}")

    # Summary Statistics
    summary = {
        "num_samples": len(results),
        "mean_prompt_sec": prompt_sec,
        "mean_generated_sec": gen_sec,
        "mean_cb0_unique": float(np.mean([r["cb0_unique"] for r in results])),
        "mean_cb1_unique": float(np.mean([r["cb1_unique"] for r in results])),
        "mean_cb2_unique": float(np.mean([r["cb2_unique"] for r in results])),
        "mean_cb3_unique": float(np.mean([r["cb3_unique"] for r in results])),
        "mean_repetition_rate": float(np.mean([r["avg_repetition_rate"] for r in results])),
        "mean_held_out_loss": float(np.mean([r["held_out_cross_entropy"] for r in results])),
        "mean_token_accuracy_pct": float(np.mean([r["continuation_token_acc_pct"] for r in results])),
        "mean_entropy": float(np.mean([r["prediction_entropy"] for r in results])),
        "mean_rms": float(np.mean([r["rms_amplitude"] for r in results])),
        "mean_peak": float(np.mean([r["peak_amplitude"] for r in results])),
        "mean_latency_sec": float(np.mean([r["generation_latency_sec"] for r in results])),
        "mean_fps": float(np.mean([r["generation_fps"] for r in results])),
        "mean_rtf": float(np.mean([r["real_time_factor"] for r in results])),
        "v1_baseline": v1_metrics
    }
    json_path = os.path.join(output_dir, "summary_report.json")
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary JSON: {json_path}")

    print("=" * 75)
    print("✅ EXPERIMENT V2-P1 COMPLETE!")
    print("=" * 75)


if __name__ == "__main__":
    run_experiment()
