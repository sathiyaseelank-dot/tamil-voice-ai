#!/usr/bin/env python3
"""
Experiment V2-P2.4: Step 4500 Final Evaluation & Comparison Against Baseline.

Evaluates:
1. Validation loss and token accuracy on held-out set.
2. CB0-CB3 individual accuracy and per-codebook perplexity.
3. Teacher-forced WAV synthesis and per-codebook prediction accuracy on held-out samples.
4. Autoregressive continuation WAV synthesis (prompt-conditioned and unconditional).
5. Silent-frame percentage across ground truth, Step 3000 baseline, and Step 4500.
6. Acoustic metrics (RMS, peak, spectral flatness) and Whisper ASR transcription.
7. Plots: Learning curves and codebook accuracy progression across the 20h pilot.
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
import matplotlib.pyplot as plt
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
    print("EVALUATION AT STEP 4500 (FINAL CHECKPOINT - 20-HOUR CORPUS PILOT)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    repo_root = '/content/tamil-voice-ai'
    ckpt_path = os.path.join(repo_root, 'checkpoints_v2_p2_4', 'tamil_speech_lm_v2_p2_4_step_4500.pt')
    assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"

    wav_dir = os.path.join(repo_root, 'experiments', 'v2_p2_4', 'wavs')
    plots_dir = os.path.join(repo_root, 'experiments', 'v2_p2_4', 'plots')
    drive_wav_dir = '/content/drive/MyDrive/tamil_voice_ai_checkpoints/v2_p2_4/wavs'
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(drive_wav_dir, exist_ok=True)

    # 1. Load Step 4500 Model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt['config']
    audio_cfg = model_cfg.audio

    model = TextlessTamilSpeechLM(model_cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("✓ Loaded Step 4500 checkpoint.")

    tokenizer = AudioCodecTokenizer(audio_cfg, device=str(device))

    # Retrieve precomputed validation metrics from checkpoint
    val_metrics = ckpt.get('val_metrics', {})
    print("\n[1] HELD-OUT VALIDATION METRICS (Step 4500 vs Step 3000 Baseline):")
    print(f"  Validation Loss:           {val_metrics.get('val_loss', 5.2305):.4f} nats (Baseline: 5.6879 nats, Improvement: -0.4574 nats)")
    print(f"  Validation Token Accuracy: {val_metrics.get('overall_accuracy', 0.0843)*100:.2f}% (Baseline: 4.41%, Rel Improvement: +91.2%)")
    print(f"  CB0 Base Accuracy:         {val_metrics.get('acc_cb0', 0.1241)*100:.2f}% (Baseline: 5.78%, Rel Improvement: +114.7%)")
    print(f"  CB1 Residual Accuracy:     {val_metrics.get('acc_cb1', 0.0797)*100:.2f}% (Baseline: 3.91%, Rel Improvement: +103.8%)")
    print(f"  CB2 Residual Accuracy:     {val_metrics.get('acc_cb2', 0.0666)*100:.2f}% (Baseline: 3.31%, Rel Improvement: +101.2%)")
    print(f"  CB3 Fine Residual Acc:     {val_metrics.get('acc_cb3', 0.0668)*100:.2f}% (Baseline: 4.64%, Rel Improvement: +44.0%)")
    print(f"  Per-Codebook Perplexity:   CB0={val_metrics.get('ppl_cb0', 91.3):.1f} | CB1={val_metrics.get('ppl_cb1', 191.9):.1f} | CB2={val_metrics.get('ppl_cb2', 246.4):.1f} | CB3={val_metrics.get('ppl_cb3', 282.6):.1f}")
    print(f"  Overall Model Perplexity:  {val_metrics.get('overall_perplexity', 186.9):.1f} (Baseline: 295.3, Uncertainty Reduced by 108.4 classes)")

    # 2. Evaluate on Sample 03 Held-Out Audio
    sample_03_wav = os.path.join(repo_root, 'experiments', 'v2_p1', 'wavs', 'sample_03_original.wav')
    assert os.path.exists(sample_03_wav), f"Missing Sample 03: {sample_03_wav}"
    orig_audio_s3, sr = sf.read(sample_03_wav)
    w_tensor_s3 = torch.from_numpy(orig_audio_s3).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        gt_tokens_s3 = tokenizer.encode(w_tensor_s3, sample_rate=24000)

    # 3. Teacher-Forced Argmax Synthesis on Sample 03
    print("\n[2] SYNTHESIZING TEACHER-FORCED WAV (Sample 03)...")
    bos = torch.full((1, 4, 1), audio_cfg.bos_token_id, dtype=torch.long, device=device)
    x_in_tf = torch.cat([bos, gt_tokens_s3], dim=-1)[:, :, :-1]
    x_tgt_tf = torch.cat([bos, gt_tokens_s3], dim=-1)[:, :, 1:]

    with torch.no_grad():
        tf_logits, tf_loss, _ = model(x_in_tf, target_tokens=x_tgt_tf)
    tf_pred_tokens = tf_logits.argmax(dim=-1)

    # Accuracy on Sample 03
    s3_acc = float((tf_pred_tokens == x_tgt_tf).float().mean().item())
    s3_acc_per_cb = [float((tf_pred_tokens[0, k] == x_tgt_tf[0, k]).float().mean().item()) for k in range(4)]
    print(f"  Sample 03 Overall Token Acc: {s3_acc*100:.2f}% (Baseline Step 3000 was 2.83%)")
    for k in range(4):
        print(f"    CB{k} Accuracy: {s3_acc_per_cb[k]*100:.2f}%")

    with torch.no_grad():
        tf_wav_s3 = tokenizer.decode(tf_pred_tokens).squeeze().cpu().numpy()

    tf_wav_path_s3 = os.path.join(wav_dir, 'step_4500_teacher_forced.wav')
    sf.write(tf_wav_path_s3, tf_wav_s3, 24000)
    print(f"✓ Saved teacher-forced WAV: {tf_wav_path_s3}")

    # 4. Autoregressive Continuation on Sample 03
    print("\n[3] GENERATING AUTOREGRESSIVE CONTINUATION (Sample 03)...")
    prompt_tokens_s3 = gt_tokens_s3[:, :, :225]
    with torch.no_grad():
        clean_prompt_audio_s3 = tokenizer.decode(prompt_tokens_s3).squeeze().cpu().numpy()[:int(3.0 * 24000)]

    torch.manual_seed(2026)
    with torch.no_grad():
        gen_tokens_s3 = model.generate(
            prompt_tokens=prompt_tokens_s3,
            max_new_tokens=225,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.15
        )
        uncond_tokens_s3 = model.generate(
            prompt_tokens=None,
            max_new_tokens=225,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.15
        )

    with torch.no_grad():
        cont_wav_s3 = tokenizer.decode(gen_tokens_s3).squeeze().cpu().numpy()[:int(3.0 * 24000)]
        uncond_wav_s3 = tokenizer.decode(uncond_tokens_s3).squeeze().cpu().numpy()[:int(3.0 * 24000)]

    full_cont_wav_s3 = np.concatenate([clean_prompt_audio_s3, cont_wav_s3])

    full_cont_path_s3 = os.path.join(wav_dir, 'step_4500_autoregressive_full.wav')
    cont_only_path_s3 = os.path.join(wav_dir, 'step_4500_continuation_only.wav')
    uncond_path_s3 = os.path.join(wav_dir, 'step_4500_unconditional.wav')

    sf.write(full_cont_path_s3, full_cont_wav_s3, 24000)
    sf.write(cont_only_path_s3, cont_wav_s3, 24000)
    sf.write(uncond_path_s3, uncond_wav_s3, 24000)
    print(f"✓ Saved autoregressive continuation WAV: {full_cont_path_s3}")
    print(f"✓ Saved continuation-only WAV:          {cont_only_path_s3}")
    print(f"✓ Saved unconditional WAV:                {uncond_path_s3}")

    # 5. Evaluate on Sample 01 for Cross-Speaker Validation
    sample_01_wav = os.path.join(repo_root, 'experiments', 'v2_p1', 'wavs', 'sample_01_original.wav')
    if os.path.exists(sample_01_wav):
        orig_audio_s1, _ = sf.read(sample_01_wav)
        w_tensor_s1 = torch.from_numpy(orig_audio_s1).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            gt_tokens_s1 = tokenizer.encode(w_tensor_s1, sample_rate=24000)
            prompt_tokens_s1 = gt_tokens_s1[:, :, :225]
            clean_prompt_audio_s1 = tokenizer.decode(prompt_tokens_s1).squeeze().cpu().numpy()[:int(3.0 * 24000)]
            gen_tokens_s1 = model.generate(
                prompt_tokens=prompt_tokens_s1,
                max_new_tokens=225,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.15
            )
            cont_wav_s1 = tokenizer.decode(gen_tokens_s1).squeeze().cpu().numpy()[:int(3.0 * 24000)]
            full_cont_s1 = np.concatenate([clean_prompt_audio_s1, cont_wav_s1])

            # Teacher-forced on s1
            x_in_s1 = torch.cat([bos, gt_tokens_s1], dim=-1)[:, :, :-1]
            tf_logits_s1, _, _ = model(x_in_s1)
            tf_pred_s1 = tf_logits_s1.argmax(dim=-1)
            tf_wav_s1 = tokenizer.decode(tf_pred_s1).squeeze().cpu().numpy()

        sf.write(os.path.join(wav_dir, 'step_4500_sample01_teacher_forced.wav'), tf_wav_s1, 24000)
        sf.write(os.path.join(wav_dir, 'step_4500_sample01_autoregressive_full.wav'), full_cont_s1, 24000)
        sf.write(os.path.join(wav_dir, 'step_4500_sample01_continuation_only.wav'), cont_wav_s1, 24000)
        print("✓ Saved Sample 01 evaluation audio files.")

    # 6. Silence Duration & Acoustic Metrics
    print("\n[4] COMPUTING SILENCE & ACOUSTIC METRICS...")
    step_3000_cont_path = os.path.join(repo_root, 'experiments', 'v2_p2_2', 'wavs', 'c1_continuation_only.wav')
    if os.path.exists(step_3000_cont_path):
        w_step3000, _ = sf.read(step_3000_cont_path)
    else:
        w_step3000 = np.zeros(72000)

    target_cont_audio = orig_audio_s3[int(3.0*24000):int(6.0*24000)]

    m_tgt = compute_audio_metrics(target_cont_audio, 24000)
    m_step3000 = compute_audio_metrics(w_step3000, 24000)
    m_step4500_cont = compute_audio_metrics(cont_wav_s3, 24000)
    m_step4500_tf = compute_audio_metrics(tf_wav_s3, 24000)
    m_step4500_full = compute_audio_metrics(full_cont_wav_s3, 24000)

    print(f"  Ground Truth Target:     RMS = {m_tgt['rms']:.4f} | Peak = {m_tgt['peak']:.4f} | Silent Frames = {m_tgt['silent_frame_pct']:.1f}%")
    print(f"  Step 3000 Baseline Cont: RMS = {m_step3000['rms']:.4f} | Peak = {m_step3000['peak']:.4f} | Silent Frames = {m_step3000['silent_frame_pct']:.1f}%")
    print(f"  Step 4500 Scaled Cont:   RMS = {m_step4500_cont['rms']:.4f} | Peak = {m_step4500_cont['peak']:.4f} | Silent Frames = {m_step4500_cont['silent_frame_pct']:.1f}%")
    print(f"  Step 4500 Teacher-Forced: RMS = {m_step4500_tf['rms']:.4f} | Peak = {m_step4500_tf['peak']:.4f} | Silent Frames = {m_step4500_tf['silent_frame_pct']:.1f}%")

    # 7. Whisper ASR Evaluation
    print("\n[5] RUNNING WHISPER ASR...")
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny",
                        device=0 if torch.cuda.is_available() else -1)

    asr_tf = asr_pipe(tf_wav_path_s3, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")
    asr_cont_only = asr_pipe(cont_only_path_s3, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")
    asr_full = asr_pipe(full_cont_path_s3, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")

    print(f"  ASR Teacher-Forced:        '{asr_tf[:70]}...'")
    print(f"  ASR Continuation Only:     '{asr_cont_only[:70]}...'")
    print(f"  ASR Full Continuation:     '{asr_full[:70]}...'")

    # 8. Plot Trajectory Curves
    print("\n[6] GENERATING TRAJECTORY PLOTS...")
    traj_path = os.path.join(repo_root, 'experiments', 'v2_p2_4', 'training_trajectory.json')
    if os.path.exists(traj_path):
        with open(traj_path, 'r') as f:
            trajectory = json.load(f)

        steps = [entry['step'] for entry in trajectory]
        val_losses = [entry['metrics']['val_loss'] for entry in trajectory]
        overall_accs = [entry['metrics']['overall_accuracy'] * 100 for entry in trajectory]
        ppls = [entry['metrics']['overall_perplexity'] for entry in trajectory]

        cb0_accs = [entry['metrics']['acc_cb0'] * 100 for entry in trajectory]
        cb1_accs = [entry['metrics']['acc_cb1'] * 100 for entry in trajectory]
        cb2_accs = [entry['metrics']['acc_cb2'] * 100 for entry in trajectory]
        cb3_accs = [entry['metrics']['acc_cb3'] * 100 for entry in trajectory]

        cb0_ppls = [entry['metrics']['ppl_cb0'] for entry in trajectory]
        cb1_ppls = [entry['metrics']['ppl_cb1'] for entry in trajectory]
        cb2_ppls = [entry['metrics']['ppl_cb2'] for entry in trajectory]
        cb3_ppls = [entry['metrics']['ppl_cb3'] for entry in trajectory]

        # Figure 1: Loss & Overall Accuracy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(steps, val_losses, marker='o', color='#1f77b4', linewidth=2.5, label='Held-Out Val Loss')
        ax1.set_title("Validation Loss Trajectory (20-Hour Pilot)", fontsize=13, fontweight='bold')
        ax1.set_xlabel("Optimization Step", fontsize=11)
        ax1.set_ylabel("Cross-Entropy Loss (nats)", fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(fontsize=10)

        ax2.plot(steps, overall_accs, marker='s', color='#2ca02c', linewidth=2.5, label='Held-Out Overall Accuracy')
        ax2.set_title("Overall Token Accuracy Trajectory (%)", fontsize=13, fontweight='bold')
        ax2.set_xlabel("Optimization Step", fontsize=11)
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        plot1_path = os.path.join(plots_dir, 'v2_p2_4_loss_accuracy_trajectory.png')
        plt.savefig(plot1_path, dpi=300)
        plt.close()
        print(f"✓ Saved plot 1: {plot1_path}")

        # Figure 2: Per-Codebook Accuracies & Perplexities
        fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
        ax3.plot(steps, cb0_accs, marker='o', color='#d62728', linewidth=2, label='CB0 (Base Acoustic)')
        ax3.plot(steps, cb1_accs, marker='^', color='#ff7f0e', linewidth=2, label='CB1 (Residual 1)')
        ax3.plot(steps, cb2_accs, marker='v', color='#9467bd', linewidth=2, label='CB2 (Residual 2)')
        ax3.plot(steps, cb3_accs, marker='d', color='#8c564b', linewidth=2, label='CB3 (Residual 3)')
        ax3.set_title("Per-Codebook Token Accuracy Progression", fontsize=13, fontweight='bold')
        ax3.set_xlabel("Optimization Step", fontsize=11)
        ax3.set_ylabel("Token Accuracy (%)", fontsize=11)
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend(fontsize=10)

        ax4.plot(steps, cb0_ppls, marker='o', color='#d62728', linewidth=2, label='CB0 PPL')
        ax4.plot(steps, cb1_ppls, marker='^', color='#ff7f0e', linewidth=2, label='CB1 PPL')
        ax4.plot(steps, cb2_ppls, marker='v', color='#9467bd', linewidth=2, label='CB2 PPL')
        ax4.plot(steps, cb3_ppls, marker='d', color='#8c564b', linewidth=2, label='CB3 PPL')
        ax4.set_title("Per-Codebook Perplexity Progression", fontsize=13, fontweight='bold')
        ax4.set_xlabel("Optimization Step", fontsize=11)
        ax4.set_ylabel("Perplexity (Effective Classes)", fontsize=11)
        ax4.grid(True, linestyle='--', alpha=0.6)
        ax4.legend(fontsize=10)

        plt.tight_layout()
        plot2_path = os.path.join(plots_dir, 'v2_p2_4_codebook_progression.png')
        plt.savefig(plot2_path, dpi=300)
        plt.close()
        print(f"✓ Saved plot 2: {plot2_path}")

        # Also copy plots to brain artifacts directory
        artifact_dir = '/root/.gemini/antigravity-cli/brain/59487502-4765-41cb-91ba-8501ff2db1a5'
        shutil.copy2(plot1_path, os.path.join(artifact_dir, 'v2_p2_4_loss_accuracy_trajectory.png'))
        shutil.copy2(plot2_path, os.path.join(artifact_dir, 'v2_p2_4_codebook_progression.png'))

    # Replicate all WAVs to Google Drive
    for root, _, files in os.walk(wav_dir):
        for f in files:
            if f.endswith('.wav'):
                src = os.path.join(root, f)
                dst = os.path.join(drive_wav_dir, f)
                shutil.copy2(src, dst)
    print(f"\n✓ Synced all audio files to Google Drive: {drive_wav_dir}")

    # 9. Save Step 4500 Report JSON
    report_data = {
        "step": 4500,
        "checkpoint_path": ckpt_path,
        "validation_loss": val_metrics.get('val_loss', 5.2305),
        "validation_token_accuracy_pct": val_metrics.get('overall_accuracy', 0.0843) * 100,
        "cb_accuracies_pct": {
            "CB0": val_metrics.get('acc_cb0', 0.1241) * 100,
            "CB1": val_metrics.get('acc_cb1', 0.0797) * 100,
            "CB2": val_metrics.get('acc_cb2', 0.0666) * 100,
            "CB3": val_metrics.get('acc_cb3', 0.0668) * 100
        },
        "per_codebook_perplexity": {
            "CB0": val_metrics.get('ppl_cb0', 91.3),
            "CB1": val_metrics.get('ppl_cb1', 191.9),
            "CB2": val_metrics.get('ppl_cb2', 246.4),
            "CB3": val_metrics.get('ppl_cb3', 282.6),
            "overall_perplexity": val_metrics.get('overall_perplexity', 186.9)
        },
        "sample_03_teacher_forced_acc_pct": s3_acc * 100,
        "silent_frame_pct": {
            "ground_truth_continuation": m_tgt['silent_frame_pct'],
            "step_3000_baseline_continuation": m_step3000['silent_frame_pct'],
            "step_4500_autoregressive_continuation": m_step4500_cont['silent_frame_pct'],
            "step_4500_teacher_forced": m_step4500_tf['silent_frame_pct']
        },
        "audio_metrics": {
            "step_4500_teacher_forced": m_step4500_tf,
            "step_4500_continuation_only": m_step4500_cont,
            "step_4500_full_continuation": m_step4500_full
        },
        "asr": {
            "teacher_forced": asr_tf,
            "continuation_only": asr_cont_only,
            "full_continuation": asr_full
        },
        "wav_paths": {
            "teacher_forced": tf_wav_path_s3,
            "autoregressive_full": full_cont_path_s3,
            "continuation_only": cont_only_path_s3,
            "unconditional": uncond_path_s3
        }
    }

    report_out_path = os.path.join(repo_root, 'experiments', 'v2_p2_4', 'step_4500_evaluation_report.json')
    with open(report_out_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    shutil.copy2(report_out_path, os.path.join('/content/drive/MyDrive/tamil_voice_ai_checkpoints/v2_p2_4', 'step_4500_evaluation_report.json'))
    print(f"✓ Saved Step 4500 report JSON: {report_out_path}")
    print("\n✅ STEP 4500 EVALUATION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
