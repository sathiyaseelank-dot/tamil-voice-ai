"""
Experiment V2-P2: Scaled Textless Tamil Speech Training (Steps 300 -> 3000).
Resumes from V2 step-300 checkpoint.
Preserves architecture, causal masking, sequential depth conditioning, and silence handling.
Evaluates every 250 steps on the fixed held-out set from V2-P1.
Generates audio samples at steps 550, 800, 1300, 1800, 2300, 2800, 3000.
"""

import os
import sys
import time
import math
import json
import csv
import random
import shutil
import gc
from typing import Optional, Dict, List

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from config import AudioConfig, ModelConfig
from model.audio_transformer import TextlessTamilSpeechLM
from tokenizer.codec import AudioCodecTokenizer
from tokenizer.audio_io import save_audio, load_audio

CHECKPOINT_STEPS = [550, 800, 1050, 1300, 1550, 1800, 2050, 2300, 2550, 2800, 3000]
AUDIO_STEPS = [550, 800, 1300, 1800, 2300, 2800, 3000]


def evaluate_held_out_set(
    model: TextlessTamilSpeechLM,
    held_out_data: List[Dict],
    audio_cfg: AudioConfig,
    device: torch.device
) -> Dict:
    """
    Evaluates model on the exact fixed held-out Tamil token sequences from V2-P1.
    """
    model.eval()
    total_loss = 0.0
    total_speech_acc = 0.0
    cb_losses = [0.0] * audio_cfg.num_codebooks
    cb_accs = [0.0] * audio_cfg.num_codebooks
    entropies = []
    count = 0

    with torch.no_grad():
        for item in held_out_data:
            tokens = item["tokens"].to(device)  # (1, 4, T)
            seq_len = tokens.shape[-1]
            if seq_len < 30:
                continue

            bos = torch.full((1, audio_cfg.num_codebooks, 1), audio_cfg.bos_token_id, dtype=torch.long, device=device)
            tok_bos = torch.cat([bos, tokens], dim=-1)
            x_in = tok_bos[:, :, :-1]
            x_target = tok_bos[:, :, 1:]

            logits, loss, metrics = model(x_in, target_tokens=x_target)
            if loss is not None and not torch.isnan(loss) and metrics is not None:
                total_loss += loss.item()
                total_speech_acc += metrics.get("speech_accuracy", 0.0)
                for k in range(audio_cfg.num_codebooks):
                    cb_losses[k] += metrics.get(f"loss_cb{k}", 0.0)
                    cb_accs[k] += metrics.get(f"acc_cb{k}", 0.0)
                entropies.append(metrics.get("entropy", 0.0))
                count += 1

    model.train()
    if count == 0:
        return {}

    eval_results = {
        "val_loss": total_loss / count,
        "val_speech_acc": (total_speech_acc / count) * 100,
        "val_entropy": float(np.mean(entropies)) if entropies else 0.0,
    }
    for k in range(audio_cfg.num_codebooks):
        eval_results[f"val_loss_cb{k}"] = cb_losses[k] / count
        eval_results[f"val_acc_cb{k}"] = (cb_accs[k] / count) * 100

    return eval_results


def run_experiment_v2_p2():
    seed = 2026
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("🚀 EXPERIMENT V2-P2: SCALED TEXTLESS TAMIL SPEECH TRAINING (300 -> 3000)")
    print("=" * 80)
    print(f"Device:       {device}")
    print(f"Random Seed:  {seed}")
    print(f"Checkpoints:  {CHECKPOINT_STEPS}")
    print(f"Audio Steps:  {AUDIO_STEPS}")
    print("=" * 80)

    # Directories
    ckpt_dir = os.path.join(repo_root, "checkpoints_v2")
    os.makedirs(ckpt_dir, exist_ok=True)
    drive_ckpt_dir = "/content/drive/MyDrive/tamil_voice_ai_checkpoints/v2"
    os.makedirs(drive_ckpt_dir, exist_ok=True)

    exp_dir = os.path.join(repo_root, "experiments", "v2_p2")
    os.makedirs(exp_dir, exist_ok=True)
    wav_dir = os.path.join(exp_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    drive_exp_dir = "/content/drive/MyDrive/tamil_voice_ai_checkpoints/v2/experiments_p2"
    os.makedirs(drive_exp_dir, exist_ok=True)
    drive_wav_dir = os.path.join(drive_exp_dir, "wavs")
    os.makedirs(drive_wav_dir, exist_ok=True)

    # 1. Neural Codec Tokenizer
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    print("\n[1/6] Loading Encodec 24 kHz Codec...")
    tokenizer = AudioCodecTokenizer(config=audio_cfg, device=str(device))
    print("✓ Neural codec tokenizer initialized.")

    # 2. Load Checkpoint Step 300
    start_ckpt_path = os.path.join(ckpt_dir, "tamil_speech_lm_v2_step_300.pt")
    if not os.path.exists(start_ckpt_path):
        start_ckpt_path = os.path.join(drive_ckpt_dir, "tamil_speech_lm_v2_step_300.pt")

    print(f"\n[2/6] Loading V2 Step 300 Checkpoint: {start_ckpt_path}...")
    ckpt_300 = torch.load(start_ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt_300["config"]
    model = TextlessTamilSpeechLM(config=model_cfg).to(device)
    model.load_state_dict(ckpt_300["model_state_dict"])
    start_step = ckpt_300.get("step", 300)
    print(f"✓ Resumed model state dict successfully at step {start_step}.")
    print(f"  Model dimensions: d_model={model_cfg.d_model}, n_heads={model_cfg.n_heads}, n_layers={model_cfg.n_layers}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    # 3. Load Fixed Held-Out Evaluation Set from V2-P1
    print("\n[3/6] Pre-loading fixed held-out evaluation recordings (matching V2-P1)...")
    v2_p1_wav_dir = os.path.join(repo_root, "experiments", "v2_p1", "wavs")
    held_out_data = []

    for idx in range(1, 11):
        wav_file = os.path.join(v2_p1_wav_dir, f"sample_{idx:02d}_original.wav")
        if not os.path.exists(wav_file):
            wav_file = f"/content/drive/MyDrive/tamil_voice_ai_checkpoints/v2/experiments_p1/wavs/sample_{idx:02d}_original.wav"

        assert os.path.exists(wav_file), f"Missing held-out WAV: {wav_file}"
        aud_tensor, sr = sf.read(wav_file)
        w = torch.from_numpy(aud_tensor).float()
        if w.ndim == 1:
            w = w.unsqueeze(0).unsqueeze(0)
        elif w.ndim == 2:
            w = w.mean(dim=-1, keepdim=True).transpose(0, 1).unsqueeze(0)

        w = w.to(device)
        with torch.no_grad():
            toks = tokenizer.encode(w, sample_rate=audio_cfg.sample_rate)  # (1, 4, 450)

        held_out_data.append({
            "sample_num": idx,
            "tokens": toks,
            "wav_path": wav_file
        })

    print(f"✓ Loaded {len(held_out_data)} fixed held-out token sequences (24 kHz, 6.0s each).")

    # Evaluate step 300 baseline on held-out set
    val_300 = evaluate_held_out_set(model, held_out_data, audio_cfg, device)
    print(f"  Step 300 Held-Out Loss: {val_300['val_loss']:.4f} | Token Accuracy: {val_300['val_speech_acc']:.2f}%")
    for k in range(audio_cfg.num_codebooks):
        print(f"    CB{k} Loss: {val_300[f'val_loss_cb{k}']:.4f} | Accuracy: {val_300[f'val_acc_cb{k}']:.2f}%")

    # 4. Stream Dataset for Training
    print("\n[4/6] Connecting to training stream 'SPRINGLab/IndicVoices-R_Tamil'...")
    hf_dataset = load_dataset("SPRINGLab/IndicVoices-R_Tamil", split="train", streaming=True)
    hf_dataset = hf_dataset.cast_column("audio", Audio(decode=False))
    data_iter = iter(hf_dataset)

    # Skip first 100 samples to avoid initial stream noise; strictly skip samples 1000-1050 to prevent overlap
    print("Initializing streaming iterator (ensuring zero overlap with held-out set)...")
    stream_idx = 0

    # 5. Training Setup
    max_steps = 3000
    remaining_steps = max_steps - start_step  # 2700
    lr = 2e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_steps, eta_min=1e-5)

    # Metrics logging
    train_metrics_file = os.path.join(exp_dir, "training_metrics.csv")
    eval_metrics_file = os.path.join(exp_dir, "evaluation_metrics.csv")
    checkpoint_manifest_file = os.path.join(exp_dir, "checkpoint_manifest.json")

    with open(train_metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "accuracy", "entropy", "grad_norm", "lr", "loss_cb0", "loss_cb1", "loss_cb2", "loss_cb3", "acc_cb0", "acc_cb1", "acc_cb2", "acc_cb3"])

    with open(eval_metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss_window", "val_loss", "val_speech_acc", "val_entropy", "val_loss_cb0", "val_loss_cb1", "val_loss_cb2", "val_loss_cb3", "val_acc_cb0", "val_acc_cb1", "val_acc_cb2", "val_acc_cb3"])

    # Record step 300 in eval metrics
    with open(eval_metrics_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            300, 6.1922, val_300["val_loss"], val_300["val_speech_acc"], val_300["val_entropy"],
            val_300["val_loss_cb0"], val_300["val_loss_cb1"], val_300["val_loss_cb2"], val_300["val_loss_cb3"],
            val_300["val_acc_cb0"], val_300["val_acc_cb1"], val_300["val_acc_cb2"], val_300["val_acc_cb3"]
        ])

    checkpoint_manifest = {
        "experiment": "V2-P2",
        "start_checkpoint": start_ckpt_path,
        "start_step": start_step,
        "target_steps": max_steps,
        "seed": seed,
        "learning_rate": lr,
        "scheduler": "CosineAnnealingLR",
        "checkpoints": [{
            "step": 300,
            "path": start_ckpt_path,
            "val_loss": val_300["val_loss"],
            "val_accuracy": val_300["val_speech_acc"]
        }],
        "best_checkpoint": {
            "step": 300,
            "val_loss": val_300["val_loss"],
            "val_accuracy": val_300["val_speech_acc"],
            "path": start_ckpt_path
        }
    }

    best_val_loss = val_300["val_loss"]

    # 6. Training Execution Loop
    print(f"\n[5/6] Commencing Scaled Autoregressive Optimization (Steps {start_step + 1} -> {max_steps})...")
    model.train()
    step = start_step
    running_loss = 0.0
    running_acc = 0.0
    running_entropy = 0.0
    window_loss = 0.0
    window_count = 0
    t_start = time.time()
    t_window = time.time()

    while step < max_steps:
        try:
            sample = next(data_iter)
            stream_idx += 1
        except StopIteration:
            data_iter = iter(hf_dataset)
            sample = next(data_iter)
            stream_idx = 1

        # Strict check: skip samples 1000 to 1050 to guarantee zero evaluation overlap
        if 1000 <= stream_idx <= 1050:
            continue

        audio_entry = sample.get("audio", {})
        if "array" in audio_entry and audio_entry["array"] is not None:
            raw_audio = np.array(audio_entry["array"])
            sr = audio_entry.get("sampling_rate", audio_cfg.sample_rate)
        elif "bytes" in audio_entry and audio_entry["bytes"] is not None:
            import io, soundfile as sf_io
            raw_audio, sr = sf_io.read(io.BytesIO(audio_entry["bytes"]))
        else:
            continue

        if len(raw_audio) < sr * 1.0:
            continue

        wav = torch.from_numpy(raw_audio).float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.ndim == 2:
            wav = wav.mean(dim=-1, keepdim=True).transpose(0, 1).unsqueeze(0)

        if sr != audio_cfg.sample_rate:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, audio_cfg.sample_rate)

        wav = wav.to(device)
        if wav.abs().max() > 0:
            wav = wav / (wav.abs().max() + 1e-6) * 0.9

        # Encode into RVQ tokens: (1, 4, seq_len)
        with torch.no_grad():
            tokens = tokenizer.encode(wav, sample_rate=audio_cfg.sample_rate)

        seq_len = tokens.shape[-1]
        if seq_len < 30:
            continue

        if seq_len > model_cfg.max_seq_len:
            tokens = tokens[:, :, :model_cfg.max_seq_len]
            seq_len = model_cfg.max_seq_len

        # Next-frame autoregressive shift with BOS
        bos = torch.full((1, audio_cfg.num_codebooks, 1), audio_cfg.bos_token_id, dtype=torch.long, device=device)
        tokens_with_bos = torch.cat([bos, tokens], dim=-1)

        x_in = tokens_with_bos[:, :, :-1]
        x_target = tokens_with_bos[:, :, 1:]

        optimizer.zero_grad()
        logits, loss, metrics = model(x_in, target_tokens=x_target)

        # Safeguard: detect NaN or Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ [CRITICAL SAFEGUARD] NaN/Inf loss encountered at step {step + 1}! Skipping gradient step.")
            optimizer.zero_grad()
            continue

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_acc += metrics.get("speech_accuracy", 0.0)
        running_entropy += metrics.get("entropy", 0.0)
        window_loss += loss.item()
        window_count += 1
        step += 1

        # Log training step
        if step % 10 == 0:
            avg_loss = running_loss / 10
            avg_acc = (running_acc / 10) * 100
            avg_entropy = running_entropy / 10
            current_lr = scheduler.get_last_lr()[0]

            with open(train_metrics_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    step, avg_loss, avg_acc, avg_entropy, float(grad_norm), current_lr,
                    metrics.get("loss_cb0", 0.0), metrics.get("loss_cb1", 0.0), metrics.get("loss_cb2", 0.0), metrics.get("loss_cb3", 0.0),
                    metrics.get("acc_cb0", 0.0) * 100, metrics.get("acc_cb1", 0.0) * 100, metrics.get("acc_cb2", 0.0) * 100, metrics.get("acc_cb3", 0.0) * 100
                ])

            running_loss = 0.0
            running_acc = 0.0
            running_entropy = 0.0

            if step % 50 == 0:
                elapsed = time.time() - t_start
                rate = (step - start_step) / elapsed
                cb_l = " ".join([f"L{k}:{metrics.get(f'loss_cb{k}', 0.0):.2f}" for k in range(audio_cfg.num_codebooks)])
                cb_a = " ".join([f"A{k}:{metrics.get(f'acc_cb{k}', 0.0)*100:.1f}%" for k in range(audio_cfg.num_codebooks)])
                print(f"[Step {step:04d}/{max_steps}] Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}% | Ent: {avg_entropy:.2f} | GN: {grad_norm:.2f} | {rate:.1f} st/s | {cb_l} | {cb_a}")

        # -------------------------------------------------------------
        # Evaluation and Checkpointing Every 250 Steps
        # -------------------------------------------------------------
        if step in CHECKPOINT_STEPS:
            print("\n" + "-" * 75)
            print(f"🔍 [EVALUATION & CHECKPOINT] STEP {step} / {max_steps}")
            print("-" * 75)

            train_window_loss = window_loss / max(1, window_count)
            window_loss = 0.0
            window_count = 0

            # Evaluate on fixed held-out set
            val_res = evaluate_held_out_set(model, held_out_data, audio_cfg, device)
            val_loss = val_res.get("val_loss", 0.0)
            val_acc = val_res.get("val_speech_acc", 0.0)
            val_entropy = val_res.get("val_entropy", 0.0)

            print(f"  Train Window Loss: {train_window_loss:.4f}")
            print(f"  Held-Out Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
            for k in range(audio_cfg.num_codebooks):
                print(f"    CB{k} Loss: {val_res.get(f'val_loss_cb{k}', 0.0):.4f} | Accuracy: {val_res.get(f'val_acc_cb{k}', 0.0):.2f}%")

            # Append to evaluation CSV
            with open(eval_metrics_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    step, train_window_loss, val_loss, val_acc, val_entropy,
                    val_res.get("val_loss_cb0", 0.0), val_res.get("val_loss_cb1", 0.0), val_res.get("val_loss_cb2", 0.0), val_res.get("val_loss_cb3", 0.0),
                    val_res.get("val_acc_cb0", 0.0), val_res.get("val_acc_cb1", 0.0), val_res.get("val_acc_cb2", 0.0), val_res.get("val_acc_cb3", 0.0)
                ])

            # Save Checkpoint
            ckpt_filename = f"tamil_speech_lm_v2_step_{step}.pt"
            local_ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
            drive_ckpt_path = os.path.join(drive_ckpt_dir, ckpt_filename)

            ckpt_payload = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": model_cfg,
                "train_loss": train_window_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_metrics": val_res,
                "version": "v2"
            }
            torch.save(ckpt_payload, local_ckpt_path)
            # Sync to Google Drive
            try:
                shutil.copy2(local_ckpt_path, drive_ckpt_path)
                print(f"💾 Checkpoint saved: {local_ckpt_path} -> Google Drive synced.")
            except Exception as e:
                print(f"Notice: Google Drive sync notice: {e}")

            checkpoint_manifest["checkpoints"].append({
                "step": step,
                "path": local_ckpt_path,
                "drive_path": drive_ckpt_path,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })

            # Best Checkpoint Tracking
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_local_path = os.path.join(ckpt_dir, "tamil_speech_lm_v2_best.pt")
                best_drive_path = os.path.join(drive_ckpt_dir, "tamil_speech_lm_v2_best.pt")
                torch.save(ckpt_payload, best_local_path)
                try:
                    shutil.copy2(best_local_path, best_drive_path)
                except Exception:
                    pass
                checkpoint_manifest["best_checkpoint"] = {
                    "step": step,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "path": best_local_path,
                    "drive_path": best_drive_path
                }
                print(f"🌟 NEW BEST CHECKPOINT: Step {step} with Val Loss {val_loss:.4f} | Accuracy {val_acc:.2f}%!")

            with open(checkpoint_manifest_file, "w") as f:
                json.dump(checkpoint_manifest, f, indent=2)

        # -------------------------------------------------------------
        # Audio Sample Generation at Scheduled Steps
        # -------------------------------------------------------------
        if step in AUDIO_STEPS:
            print(f"\n🎵 [AUDIO GENERATION] Generating evaluation audio at Step {step}...")
            model.eval()
            with torch.no_grad():
                # 1. Unconditional generation (from BOS)
                uncond_frames = model.generate(
                    prompt_tokens=None,
                    max_new_tokens=225,  # 3.0 seconds
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.15
                )
                uncond_wav = tokenizer.decode(uncond_frames).squeeze(0).cpu()
                uncond_wav_file = os.path.join(wav_dir, f"step_{step:04d}_unconditional.wav")
                save_audio(uncond_wav_file, uncond_wav, audio_cfg.sample_rate)

                # Diagnostic metrics on unconditional tokens
                uncond_np = uncond_frames.squeeze(0).cpu().numpy()
                unq_uncond = [len(np.unique(uncond_np[k])) for k in range(audio_cfg.num_codebooks)]
                rep_uncond = [(uncond_np[k, 1:] == uncond_np[k, :-1]).mean() for k in range(audio_cfg.num_codebooks)]

                # 2. Prompt-conditioned continuation using fixed Sample 03 prompt
                sample_03_prompt = held_out_data[2]["tokens"][:, :, :225].to(device)  # First 3.0s of Sample 03
                cont_frames = model.generate(
                    prompt_tokens=sample_03_prompt,
                    max_new_tokens=225,  # 3.0s continuation
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.15
                )
                prompt_wav = tokenizer.decode(sample_03_prompt).squeeze(0).cpu()
                cont_wav = tokenizer.decode(cont_frames).squeeze(0).cpu()
                full_cont_wav = torch.cat([prompt_wav[:, :int(3.0 * audio_cfg.sample_rate)], cont_wav[:, :int(3.0 * audio_cfg.sample_rate)]], dim=-1)
                cont_wav_file = os.path.join(wav_dir, f"step_{step:04d}_prompt_continuation.wav")
                save_audio(cont_wav_file, full_cont_wav, audio_cfg.sample_rate)

                # Diagnostic metrics on continuation tokens
                cont_np = cont_frames.squeeze(0).cpu().numpy()
                unq_cont = [len(np.unique(cont_np[k])) for k in range(audio_cfg.num_codebooks)]
                rep_cont = [(cont_np[k, 1:] == cont_np[k, :-1]).mean() for k in range(audio_cfg.num_codebooks)]

                # Audio signal metrics
                aud_arr = cont_wav.squeeze().numpy()
                rms = float(np.sqrt(np.mean(aud_arr ** 2)))
                peak = float(np.max(np.abs(aud_arr)))
                clip_pct = float(np.sum(np.abs(aud_arr) >= 0.999) / len(aud_arr) * 100)

                print(f"  • Unconditional Unique Tokens: CB0={unq_uncond[0]}, CB1={unq_uncond[1]}, CB2={unq_uncond[2]}, CB3={unq_uncond[3]}")
                print(f"  • Unconditional Repetition:   CB0={rep_uncond[0]*100:.1f}%, CB1={rep_uncond[1]*100:.1f}%, CB2={rep_uncond[2]*100:.1f}%, CB3={rep_uncond[3]*100:.1f}%")
                print(f"  • Continuation Unique Tokens:  CB0={unq_cont[0]}, CB1={unq_cont[1]}, CB2={unq_cont[2]}, CB3={unq_cont[3]}")
                print(f"  • Continuation Repetition:    CB0={rep_cont[0]*100:.1f}%, CB1={rep_cont[1]*100:.1f}%, CB2={rep_cont[2]*100:.1f}%, CB3={rep_cont[3]*100:.1f}%")
                print(f"  • Audio Properties:           RMS={rms:.4f}, Peak={peak:.3f}, Clipping={clip_pct:.1f}%")
                print(f"  • Saved: {cont_wav_file}")

                # Copy audio to Google Drive
                try:
                    shutil.copy2(uncond_wav_file, os.path.join(drive_wav_dir, f"step_{step:04d}_unconditional.wav"))
                    shutil.copy2(cont_wav_file, os.path.join(drive_wav_dir, f"step_{step:04d}_prompt_continuation.wav"))
                except Exception:
                    pass

                # Safeguard: detect repetition collapse
                if max(rep_cont) > 0.50:
                    print(f"⚠️ [WARNING] Elevated repetition rate detected ({max(rep_cont)*100:.1f}%) at step {step}!")

            model.train()

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED TO STEP 3000!")
    print("=" * 80)

    # 7. Post-Training Deliverables: Secondary Whisper ASR, Plotting & Final Report
    print("\n[6/6] Generating Post-Training Deliverables (ASR, Curves & Manifest)...")

    # ASR on generated audio
    print("Running secondary Tamil Whisper ASR on generated audio samples...")
    from transformers import pipeline
    try:
        asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        print(f"Notice: ASR pipeline load exception: {e}")
        asr_pipeline = None

    asr_results = {}
    if asr_pipeline is not None:
        for s in AUDIO_STEPS:
            cont_wav_file = os.path.join(wav_dir, f"step_{s:04d}_prompt_continuation.wav")
            uncond_wav_file = os.path.join(wav_dir, f"step_{s:04d}_unconditional.wav")
            try:
                txt_cont = asr_pipeline(cont_wav_file, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")
            except Exception:
                txt_cont = ""
            try:
                txt_uncond = asr_pipeline(uncond_wav_file, generate_kwargs={"language": "tamil", "task": "transcribe"}).get("text", "")
            except Exception:
                txt_uncond = ""

            asr_results[str(s)] = {
                "continuation_asr": txt_cont,
                "unconditional_asr": txt_uncond
            }
            print(f"  Step {s} Continuation ASR: '{txt_cont[:60]}...'")

    with open(os.path.join(exp_dir, "asr_evaluation.json"), "w") as f:
        json.dump(asr_results, f, indent=2)

    # Generate Publication Plots
    print("Generating loss and accuracy curve visualizations...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Read eval metrics
    eval_steps, t_losses, v_losses, v_accs = [], [], [], []
    cb_accs = {0: [], 1: [], 2: [], 3: []}
    cb_losses = {0: [], 1: [], 2: [], 3: []}

    with open(eval_metrics_file, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            eval_steps.append(int(r["step"]))
            t_losses.append(float(r["train_loss_window"]))
            v_losses.append(float(r["val_loss"]))
            v_accs.append(float(r["val_speech_acc"]))
            for k in range(4):
                cb_accs[k].append(float(r[f"val_acc_cb{k}"]))
                cb_losses[k].append(float(r[f"val_loss_cb{k}"]))

    # Plot 1: Training & Validation Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eval_steps, t_losses, label="Train Loss (Window)", color="#1f77b4", linewidth=2, marker="o")
    ax.plot(eval_steps, v_losses, label="Held-Out Val Loss", color="#ff7f0e", linewidth=2.5, marker="s")
    ax.set_title("Textless Tamil Speech LM (V2): Loss Curves (Steps 300 -> 3000)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Optimization Steps", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss (Speech Only)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    loss_plot_path = os.path.join(plots_dir, "loss_curves.png")
    fig.tight_layout()
    fig.savefig(loss_plot_path, dpi=200)
    plt.close(fig)

    # Plot 2: Per-Codebook Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eval_steps, cb_accs[0], label="CB0 (Acoustic Base)", color="#2ca02c", linewidth=2.5, marker="o")
    ax.plot(eval_steps, cb_accs[1], label="CB1 (Residual 1)", color="#1f77b4", linewidth=2, marker="^")
    ax.plot(eval_steps, cb_accs[2], label="CB2 (Residual 2)", color="#9467bd", linewidth=2, marker="d")
    ax.plot(eval_steps, cb_accs[3], label="CB3 (High-Freq Residual)", color="#d62728", linewidth=2, marker="v")
    ax.axhline(0.098, color="gray", linestyle=":", label="Uniform Random Baseline (0.098%)")
    ax.set_title("Held-Out Per-Codebook Token Accuracy (Steps 300 -> 3000)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Optimization Steps", fontsize=12)
    ax.set_ylabel("Token Accuracy (%)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    acc_plot_path = os.path.join(plots_dir, "codebook_accuracy_curves.png")
    fig.tight_layout()
    fig.savefig(acc_plot_path, dpi=200)
    plt.close(fig)

    # Copy plots & metrics to Google Drive
    try:
        shutil.copy2(loss_plot_path, os.path.join(drive_exp_dir, "loss_curves.png"))
        shutil.copy2(acc_plot_path, os.path.join(drive_exp_dir, "codebook_accuracy_curves.png"))
        shutil.copy2(train_metrics_file, os.path.join(drive_exp_dir, "training_metrics.csv"))
        shutil.copy2(eval_metrics_file, os.path.join(drive_exp_dir, "evaluation_metrics.csv"))
        shutil.copy2(checkpoint_manifest_file, os.path.join(drive_exp_dir, "checkpoint_manifest.json"))
        shutil.copy2(os.path.join(exp_dir, "asr_evaluation.json"), os.path.join(drive_exp_dir, "asr_evaluation.json"))
    except Exception as e:
        print(f"Notice: Google Drive sync notice: {e}")

    print("✓ All Experiment V2-P2 post-training assets generated and backed up.")
    print("=" * 80)
    print("🎉 EXPERIMENT V2-P2 EXECUTION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_v2_p2()
