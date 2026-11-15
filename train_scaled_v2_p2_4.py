#!/usr/bin/env python3
"""
Experiment V2-P2.4: Phase 2 — Controlled Training on 20-Hour Scaled Tamil Corpus.

Requirements:
- Start from preserved V2 step-3000 checkpoint (c1a3165f...).
- Keep existing architecture (87.17M params, 12 layers, 16 heads, d_model=1024).
- Keep standard cross-entropy loss across 4 codebooks.
- Effective batch size = 16 via gradient accumulation (micro_batch=2, accum=8).
- Learning-rate schedule with 100-step warmup and cosine decay.
- Checkpoints saved every 250 steps (3250, 3500, 3750, 4000, 4250, 4500).
- Strict gradient clipping and numerical stability checks.
- Zero scheduled sampling or hierarchical loss weighting (pure data-scale pilot).
"""

import os
import sys
import time
import math
import json
import random
import hashlib
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import librosa
from transformers import pipeline

sys.path.insert(0, '/content/tamil-voice-ai')
from config import ModelConfig, AudioConfig
from model.audio_transformer import TextlessTamilSpeechLM
from tokenizer.codec import AudioCodecTokenizer


class PreEncodedTokenDataset(Dataset):
    def __init__(self, data_path: str, max_seq_len: int = 750):
        print(f"Loading pre-encoded tokens from: {data_path}...")
        self.items = torch.load(data_path, weights_only=False)
        self.max_seq_len = max_seq_len
        print(f"✓ Loaded {len(self.items)} recordings.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        tokens = item['tokens']  # (4, seq_len) int16
        if tokens.shape[-1] > self.max_seq_len:
            tokens = tokens[:, :self.max_seq_len]
        return {
            "tokens": tokens,
            "speaker_id": item['speaker_id'],
            "duration": item['duration'],
            "text": item['text']
        }


def make_collate_fn(pad_token_id: int = 1024, bos_token_id: int = 1025, max_seq_len: int = 750):
    def collate_fn(batch):
        max_len = max(item['tokens'].shape[-1] for item in batch)
        max_len = min(max_len, max_seq_len)
        batch_in = []
        batch_tgt = []
        for item in batch:
            toks = item['tokens'][:, :max_len].long()
            curr_len = toks.shape[-1]
            bos = torch.full((4, 1), bos_token_id, dtype=torch.long)
            toks_bos = torch.cat([bos, toks], dim=-1)
            x_in = toks_bos[:, :-1]
            x_tgt = toks_bos[:, 1:]
            if curr_len < max_len:
                pad_len = max_len - curr_len
                x_in = torch.cat([x_in, torch.full((4, pad_len), pad_token_id, dtype=torch.long)], dim=-1)
                x_tgt = torch.cat([x_tgt, torch.full((4, pad_len), pad_token_id, dtype=torch.long)], dim=-1)
            batch_in.append(x_in)
            batch_tgt.append(x_tgt)
        return torch.stack(batch_in, dim=0), torch.stack(batch_tgt, dim=0)
    return collate_fn


def get_lr(step: int, warmup_steps: int = 100, max_steps: int = 1500, min_lr: float = 1e-5, max_lr: float = 1.5e-4) -> float:
    if step < warmup_steps:
        return min_lr + (max_lr - min_lr) * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def evaluate_val_set(model: nn.Module, val_loader: DataLoader, device: torch.device, pad_token_id: int = 1024, max_batches: int = 100):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    cb_correct = [0] * 4
    cb_tokens = [0] * 4
    cb_losses = [0.0] * 4
    all_entropies = []
    batch_count = 0

    with torch.no_grad():
        for b_in, b_tgt in val_loader:
            if batch_count >= max_batches:
                break
            b_in, b_tgt = b_in.to(device), b_tgt.to(device)
            logits, loss, _ = model(b_in, target_tokens=b_tgt)
            if loss is None:
                continue

            # Per-codebook calculations
            valid_mask = (b_tgt != pad_token_id)
            total_loss += loss.item() * valid_mask.sum().item()
            total_tokens += valid_mask.sum().item()

            pred_tokens = logits.argmax(dim=-1)
            total_correct += ((pred_tokens == b_tgt) & valid_mask).sum().item()

            for k in range(4):
                val_k = valid_mask[:, k, :]
                num_k = val_k.sum().item()
                if num_k > 0:
                    corr_k = ((pred_tokens[:, k, :] == b_tgt[:, k, :]) & val_k).sum().item()
                    cb_correct[k] += corr_k
                    cb_tokens[k] += num_k

                    loss_k = F.cross_entropy(logits[:, k, :, :].reshape(-1, logits.size(-1)),
                                             b_tgt[:, k, :].reshape(-1),
                                             ignore_index=pad_token_id)
                    cb_losses[k] += loss_k.item() * num_k

            # Entropy
            probs = F.softmax(logits[valid_mask], dim=-1)
            ent = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1).mean().item()
            all_entropies.append(ent)
            batch_count += 1

    mean_loss = total_loss / max(1, total_tokens)
    mean_acc = total_correct / max(1, total_tokens)
    acc_per_cb = [cb_correct[k] / max(1, cb_tokens[k]) for k in range(4)]
    loss_per_cb = [cb_losses[k] / max(1, cb_tokens[k]) for k in range(4)]
    ppl_per_cb = [math.exp(l) for l in loss_per_cb]
    mean_entropy = float(np.mean(all_entropies)) if all_entropies else 0.0

    return {
        "val_loss": mean_loss,
        "overall_accuracy": mean_acc,
        "overall_perplexity": math.exp(mean_loss),
        "acc_cb0": acc_per_cb[0],
        "acc_cb1": acc_per_cb[1],
        "acc_cb2": acc_per_cb[2],
        "acc_cb3": acc_per_cb[3],
        "loss_cb0": loss_per_cb[0],
        "loss_cb1": loss_per_cb[1],
        "loss_cb2": loss_per_cb[2],
        "loss_cb3": loss_per_cb[3],
        "ppl_cb0": ppl_per_cb[0],
        "ppl_cb1": ppl_per_cb[1],
        "ppl_cb2": ppl_per_cb[2],
        "ppl_cb3": ppl_per_cb[3],
        "entropy": mean_entropy
    }


def compute_audio_metrics(y: np.ndarray, sr: int = 24000) -> dict:
    if len(y) == 0:
        return {"rms": 0.0, "peak": 0.0, "spectral_flatness": 0.0, "silence_duration_sec": 0.0}
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    try:
        flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    except Exception:
        flat = 0.0
    # Silence: frames where local RMS < 0.01
    frame_len = int(sr * 0.05)
    silence_frames = 0
    total_frames = 0
    for i in range(0, len(y) - frame_len, frame_len):
        f_rms = np.sqrt(np.mean(y[i:i+frame_len]**2))
        if f_rms < 0.008:
            silence_frames += 1
        total_frames += 1
    silence_duration = (silence_frames / max(1, total_frames)) * (len(y) / sr)

    return {
        "rms": rms,
        "peak": peak,
        "spectral_flatness": flat,
        "silence_duration_sec": float(silence_duration)
    }


def main():
    print("=" * 80)
    print("EXPERIMENT V2-P2.4: CONTROLLED SCALED TRAINING (20-HOUR CORPUS)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Reproducibility
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)

    repo_root = '/content/tamil-voice-ai'
    baseline_ckpt_path = os.path.join(repo_root, 'checkpoints_v2', 'tamil_speech_lm_v2_best.pt')
    assert os.path.exists(baseline_ckpt_path), f"Missing baseline: {baseline_ckpt_path}"
    with open(baseline_ckpt_path, 'rb') as f:
        baseline_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Baseline Checkpoint SHA256: {baseline_hash}")

    output_dir = os.path.join(repo_root, 'checkpoints_v2_p2_4')
    drive_dir = '/content/drive/MyDrive/tamil_voice_ai_checkpoints/v2_p2_4'
    exp_dir = os.path.join(repo_root, 'experiments', 'v2_p2_4')
    wav_dir = os.path.join(exp_dir, 'wavs')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(drive_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    # 1. Model Initialization
    model_cfg = ModelConfig()
    audio_cfg = AudioConfig()
    model_cfg.audio = audio_cfg

    model = TextlessTamilSpeechLM(model_cfg).to(device)
    ckpt = torch.load(baseline_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"✓ Loaded baseline weights from Step {ckpt.get('step', 3000)}.")

    tokenizer = AudioCodecTokenizer(audio_cfg, device=str(device))

    # 2. Dataset Loaders
    train_tokens_path = os.path.join(repo_root, 'data', 'v2_p2_4', 'v2_p2_4_train_tokens.pt')
    val_tokens_path = os.path.join(repo_root, 'data', 'v2_p2_4', 'v2_p2_4_val_tokens.pt')

    train_dataset = PreEncodedTokenDataset(train_tokens_path, max_seq_len=model_cfg.max_seq_len)
    val_dataset = PreEncodedTokenDataset(val_tokens_path, max_seq_len=model_cfg.max_seq_len)

    collate_fn = make_collate_fn(pad_token_id=audio_cfg.pad_token_id,
                                 bos_token_id=audio_cfg.bos_token_id,
                                 max_seq_len=model_cfg.max_seq_len)

    micro_batch_size = 2
    grad_accum_steps = 8  # 2 * 8 = 16 effective batch size
    effective_batch_size = micro_batch_size * grad_accum_steps

    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.5e-4,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )

    start_step = 3000
    total_training_steps = 1500  # Step 3000 -> Step 4500 (3+ epochs of 20h corpus)
    target_end_step = start_step + total_training_steps
    warmup_steps = 100

    print(f"Training Plan: Step {start_step} -> Step {target_end_step} ({total_training_steps} optimization steps)")
    print(f"Effective Batch Size: {effective_batch_size} (micro-batch: {micro_batch_size}, accum: {grad_accum_steps})")
    print(f"Dataset Size: {len(train_dataset)} train samples (~19.0h) | {len(val_dataset)} val samples (~1.8h)")

    # Initial Baseline Validation
    print("\n[Baseline Evaluation] Benchmarking starting checkpoint on held-out validation set...")
    initial_val_metrics = evaluate_val_set(model, val_loader, device, pad_token_id=audio_cfg.pad_token_id)
    print(f"✓ Baseline Val Loss: {initial_val_metrics['val_loss']:.4f} nats | Accuracy: {initial_val_metrics['overall_accuracy']*100:.2f}% | PPL: {initial_val_metrics['overall_perplexity']:.1f}")
    print(f"  CB0: {initial_val_metrics['acc_cb0']*100:.2f}% | CB1: {initial_val_metrics['acc_cb1']*100:.2f}% | CB2: {initial_val_metrics['acc_cb2']*100:.2f}% | CB3: {initial_val_metrics['acc_cb3']*100:.2f}%")

    best_val_loss = initial_val_metrics['val_loss']
    best_checkpoint_path = os.path.join(output_dir, 'tamil_speech_lm_v2_p2_4_best.pt')

    # Training Loop
    model.train()
    optimizer.zero_grad()

    step = 0
    global_step = start_step
    running_train_loss = 0.0
    running_train_acc = 0.0
    micro_step = 0
    t0 = time.time()

    train_history = []
    checkpoint_records = [{
        "step": start_step,
        "metrics": initial_val_metrics,
        "type": "baseline_step_3000"
    }]

    print("\n" + "=" * 80)
    print(f"STARTING SCALED TRAINING LOOP: STEP {start_step} -> {target_end_step}")
    print("=" * 80)

    train_iter = iter(train_loader)

    while step < total_training_steps:
        try:
            b_in, b_tgt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            b_in, b_tgt = next(train_iter)

        b_in, b_tgt = b_in.to(device, non_blocking=True), b_tgt.to(device, non_blocking=True)

        logits, loss, metrics = model(b_in, target_tokens=b_tgt)
        assert not torch.isnan(loss), f"FATAL: NaN loss encountered at step {global_step}!"
        assert not torch.isinf(loss), f"FATAL: Inf loss encountered at step {global_step}!"

        # Scale loss for gradient accumulation
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        running_train_loss += loss.item()
        running_train_acc += metrics.get("speech_accuracy", 0.0)
        micro_step += 1

        if micro_step % grad_accum_steps == 0:
            # Gradient clipping & optimizer step
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            assert not math.isnan(grad_norm), f"FATAL: NaN gradient norm at step {global_step}!"

            # Update learning rate
            current_lr = get_lr(step, warmup_steps=warmup_steps, max_steps=total_training_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.step()
            optimizer.zero_grad()

            step += 1
            global_step = start_step + step

            # Periodic logging
            if step % 25 == 0:
                avg_loss = running_train_loss / (25 * grad_accum_steps)
                avg_acc = (running_train_acc / (25 * grad_accum_steps)) * 100
                elapsed = time.time() - t0
                steps_per_sec = step / max(elapsed, 1e-5)
                print(f"[Step {global_step:4d}/{target_end_step}] Train Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}% | GradNorm: {grad_norm:.3f} | LR: {current_lr:.2e} | Speed: {steps_per_sec:.2f} step/s")
                running_train_loss = 0.0
                running_train_acc = 0.0

            # Periodic Checkpointing & Validation Evaluation
            if step % 250 == 0 or step == total_training_steps:
                print("\n" + "-" * 70)
                print(f"EVALUATION AT STEP {global_step} (Trained on {step * effective_batch_size} samples / {(step * effective_batch_size * 8.1)/3600.0:.2f}h audio)")
                print("-" * 70)

                val_metrics = evaluate_val_set(model, val_loader, device, pad_token_id=audio_cfg.pad_token_id)
                print(f"Val Loss: {val_metrics['val_loss']:.4f} nats | Acc: {val_metrics['overall_accuracy']*100:.2f}% | PPL: {val_metrics['overall_perplexity']:.1f}")
                print(f"CB Accuracies: CB0={val_metrics['acc_cb0']*100:.2f}% | CB1={val_metrics['acc_cb1']*100:.2f}% | CB2={val_metrics['acc_cb2']*100:.2f}% | CB3={val_metrics['acc_cb3']*100:.2f}%")
                print(f"CB Perplexities: CB0={val_metrics['ppl_cb0']:.1f} | CB1={val_metrics['ppl_cb1']:.1f} | CB2={val_metrics['ppl_cb2']:.1f} | CB3={val_metrics['ppl_cb3']:.1f}")

                # Save periodic checkpoint
                ckpt_data = {
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model_cfg,
                    "val_metrics": val_metrics,
                    "baseline_hash": baseline_hash,
                    "corpus_hours": 20.73
                }
                periodic_path = os.path.join(output_dir, f"tamil_speech_lm_v2_p2_4_step_{global_step}.pt")
                torch.save(ckpt_data, periodic_path)
                print(f"✓ Saved periodic checkpoint to: {periodic_path}")

                # Replicate to Drive
                drive_ckpt_path = os.path.join(drive_dir, f"tamil_speech_lm_v2_p2_4_step_{global_step}.pt")
                shutil.copy2(periodic_path, drive_ckpt_path)

                is_best = val_metrics['val_loss'] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['val_loss']
                    torch.save(ckpt_data, best_checkpoint_path)
                    shutil.copy2(best_checkpoint_path, os.path.join(drive_dir, 'tamil_speech_lm_v2_p2_4_best.pt'))
                    print(f"★ NEW BEST MODEL SAVED! (Val Loss: {best_val_loss:.4f} nats)")

                checkpoint_records.append({
                    "step": global_step,
                    "metrics": val_metrics,
                    "checkpoint_path": periodic_path,
                    "is_best": is_best
                })
                print("-" * 70 + "\n")
                model.train()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Final Best Val Loss: {best_val_loss:.4f} nats")
    print("=" * 80)

    # Save training trajectory log
    traj_path = os.path.join(exp_dir, "training_trajectory.json")
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_records, f, indent=2)

    print(f"✓ Saved training history to: {traj_path}")


if __name__ == "__main__":
    main()
