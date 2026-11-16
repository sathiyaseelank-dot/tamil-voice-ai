#!/usr/bin/env python3
"""
Experiment V2-P2.5-A: Standalone Reproducible Training Script for Single-Codebook (CB0) LM.

Architecture:
- TextlessTamilSpeechLM_CB0 (77.69M parameters, 6 layers, d_model=1024, 16 heads, d_ff=4096)
- Input: CB0 acoustic tokens (vocabulary size: 1029)
- Output: CB0 next-frame predictions (ignore_index=1024)
- Native 1-codebook Encodec 24 kHz reconstruction

Requirements:
- Warm-starts trunk and CB0 head from V2-P2.4 best checkpoint without modifying source weights.
- Preserves speaker-independent evaluation set.
- Cosine LR schedule with 100-step warmup.
- Effective batch size = 16 (micro-batch 2 * grad_accum 8).
- Strictly reproducible random seed.
"""

import os
import sys
import time
import math
import json
import argparse
import random
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio

REPO_ROOT = '/content/tamil-voice-ai' if os.path.exists('/content/tamil-voice-ai') else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from config import ModelConfig, AudioConfig
from model.audio_transformer_cb0 import TextlessTamilSpeechLM_CB0
from tokenizer.codec import AudioCodecTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="V2-P2.5-A: Single-Codebook (CB0) Training (Dual-Mode: A1 & A2)")
    parser.add_argument("--mode", type=str, choices=["a1", "a2", "random", "transferred"], default="a2",
                        help="Experiment mode: 'a1'/'random' (random initialization) or 'a2'/'transferred' (transferred weights from checkpoint)")
    parser.add_argument("--warm_start_checkpoint", type=str,
                        default=os.path.join(REPO_ROOT, "checkpoints_v2_p2_4", "tamil_speech_lm_v2_p2_4_best.pt"),
                        help="Path to trained 4-codebook checkpoint to warm-start trunk and CB0 head (for mode a2)")
    parser.add_argument("--val_tokens_path", type=str, default=os.path.join(REPO_ROOT, "data", "v2_p2_4", "v2_p2_4_val_tokens.pt"),
                        help="Path to pre-encoded full held-out validation tokens dataset")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output checkpoint directory (defaults to checkpoints_v2_p2_5_a1 or a2)")
    parser.add_argument("--drive_dir", type=str, default=None,
                        help="Google Drive backup directory (defaults based on mode)")
    parser.add_argument("--max_steps", type=int, default=2000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=1.0e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--micro_batch_size", type=int, default=2, help="Micro batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--log_every", type=int, default=10, help="Logging step interval")
    parser.add_argument("--eval_every", type=int, default=250, help="Validation step interval")
    parser.add_argument("--save_every", type=int, default=250, help="Checkpoint save interval")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step: int, warmup_steps: int, max_steps: int, min_lr: float, max_lr: float) -> float:
    if step < warmup_steps:
        return min_lr + (max_lr - min_lr) * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def load_validation_corpus(val_tokens_path: str, tokenizer: AudioCodecTokenizer, device: torch.device):
    """
    Loads evaluation corpus:
    1. Full held-out validation set (if available from pre-encoded tokens file).
    2. 10 fixed diagnostic audio recordings for standardized continuation synthesis and listening.
    """
    diag_samples = []
    base_dir = os.path.join(REPO_ROOT, 'experiments', 'v2_p1', 'wavs')
    for i in range(1, 11):
        wav_path = os.path.join(base_dir, f'sample_{i:02d}_original.wav')
        if os.path.exists(wav_path):
            y, sr = sf.read(wav_path)
            w_t = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                toks_4cb = tokenizer.encode(w_t, 24000)
                toks_cb0 = toks_4cb[:, 0, :]
            diag_samples.append({
                "sample_id": f"sample_{i:02d}",
                "tokens": toks_cb0,
                "audio": y
            })

    full_val_samples = []
    if val_tokens_path and os.path.exists(val_tokens_path):
        raw_items = torch.load(val_tokens_path, weights_only=False)
        print(f"✓ Loaded {len(raw_items)} recordings from full held-out validation dataset: {val_tokens_path}")
        for item in raw_items:
            toks = item['tokens']
            if toks.ndim == 2 and toks.shape[0] == 4:
                toks_cb0 = toks[0:1, :].to(device)
            elif toks.ndim == 1:
                toks_cb0 = toks.unsqueeze(0).to(device)
            else:
                toks_cb0 = toks.to(device)
            full_val_samples.append({"tokens": toks_cb0.long()})
    else:
        full_val_samples = diag_samples

    print(f"✓ Evaluation setup: {len(full_val_samples)} validation recordings ({len(diag_samples)} diagnostic audio references).")
    return full_val_samples, diag_samples


def evaluate_held_out(model: nn.Module, full_val_samples: list, diag_samples: list, tokenizer: AudioCodecTokenizer, device: torch.device, out_dir: str, step: int):
    """
    Evaluates CB0 loss, accuracy, perplexity across full validation set and produces:
    1. Autoregressive continuation
    2. Unconditional generation
    3. Teacher-forced reconstruction
    4. Quantitative token and audio diagnostics (RMS, silence %, repetition rate, unique tokens)
    """
    model.eval()
    losses = []
    accuracies = []
    entropies = []

    with torch.no_grad():
        for s in full_val_samples:
            toks = s['tokens'].to(device)  # (1, T)
            bos = torch.tensor([[1025]], dtype=torch.long, device=device)
            toks_bos = torch.cat([bos, toks], dim=-1)
            x_in = toks_bos[:, :-1]
            x_tgt = toks_bos[:, 1:]

            _, loss, metrics = model(x_in, target_tokens=x_tgt)
            if metrics:
                losses.append(metrics['cb0_loss'])
                accuracies.append(metrics['cb0_accuracy'])
                entropies.append(metrics['cb0_entropy'])

    mean_loss = float(np.mean(losses)) if losses else 0.0
    mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
    mean_ent = float(np.mean(entropies)) if entropies else 0.0
    mean_ppl = float(math.exp(min(mean_loss, 20.0)))

    # Select reference sample 03 for standardized generation
    synth_samples = diag_samples if diag_samples else full_val_samples
    s3 = synth_samples[2] if len(synth_samples) > 2 else synth_samples[0]
    prompt_75 = s3['tokens'][:, :75].to(device)

    with torch.no_grad():
        # 1. Autoregressive Continuation (prompt=75, gen=150)
        gen_tokens = model.generate(prompt_tokens=prompt_75, max_new_tokens=150, temperature=0.8, top_k=50, top_p=0.9)
        emb = tokenizer.model.quantizer.decode(gen_tokens)
        synth_wav = tokenizer.model.decoder(emb).squeeze().cpu().numpy()

        # 2. Unconditional Generation (from pure BOS)
        uncond_tokens = model.generate(prompt_tokens=None, max_new_tokens=150, temperature=0.8, top_k=50, top_p=0.9)
        emb_uncond = tokenizer.model.quantizer.decode(uncond_tokens)
        uncond_wav = tokenizer.model.decoder(emb_uncond).squeeze().cpu().numpy()

        # 3. Teacher-Forced Argmax Reconstruction
        tf_bos = torch.cat([torch.tensor([[1025]], dtype=torch.long, device=device), prompt_75], dim=-1)
        tf_logits, _, _ = model(tf_bos[:, :-1])
        tf_tokens = tf_logits.argmax(dim=-1).unsqueeze(1)
        emb_tf = tokenizer.model.quantizer.decode(tf_tokens)
        tf_wav = tokenizer.model.decoder(emb_tf).squeeze().cpu().numpy()

    # Audio metrics
    rms = float(np.sqrt(np.mean(synth_wav ** 2))) if len(synth_wav) > 0 else 0.0
    silence_pct = float(np.mean(np.abs(synth_wav) < 0.01)) * 100.0 if len(synth_wav) > 0 else 100.0

    # Token diagnostics
    gen_arr = gen_tokens.squeeze().cpu().numpy()
    uniq_tokens = int(len(np.unique(gen_arr)))
    consec_reps = int(np.sum(gen_arr[1:] == gen_arr[:-1])) if len(gen_arr) > 1 else 0
    repetition_rate = float(consec_reps / max(1, len(gen_arr) - 1)) * 100.0

    # Save audio files
    wav_out = os.path.join(out_dir, f"sample03_p2_5_a2_step_{step}.wav")
    wav_uncond_out = os.path.join(out_dir, f"sample_uncond_p2_5_a2_step_{step}.wav")
    wav_tf_out = os.path.join(out_dir, f"sample03_tf_p2_5_a2_step_{step}.wav")

    sf.write(wav_out, synth_wav, 24000)
    sf.write(wav_uncond_out, uncond_wav, 24000)
    sf.write(wav_tf_out, tf_wav, 24000)

    return {
        "step": step,
        "val_loss": mean_loss,
        "val_accuracy_pct": mean_acc * 100.0,
        "val_perplexity": mean_ppl,
        "val_entropy": mean_ent,
        "val_sequences_evaluated": len(full_val_samples),
        "ar_rms": rms,
        "ar_silence_pct": silence_pct,
        "ar_unique_tokens": uniq_tokens,
        "ar_repetition_rate_pct": repetition_rate,
        "synth_wav": wav_out,
        "uncond_wav": wav_uncond_out,
        "tf_wav": wav_tf_out
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    is_random = args.mode in ["a1", "random"]
    exp_code = "P2.5-A1" if is_random else "P2.5-A2"
    exp_name = f"V2-P2.5-{'A1 (Random Initialization)' if is_random else 'A2 (Transferred Weights from V2-P2.4)'}"

    if args.output_dir is None:
        args.output_dir = os.path.join(REPO_ROOT, f"checkpoints_v2_p2_5_{'a1' if is_random else 'a2'}")
    if args.drive_dir is None:
        args.drive_dir = f"/content/drive/MyDrive/tamil_voice_ai_checkpoints/v2_p2_5_{'a1' if is_random else 'a2'}"

    print("=" * 80)
    print(f"STARTING EXPERIMENT {exp_name}")
    print("=" * 80)
    print(f"Mode:                    {exp_code} ({'Random from scratch' if is_random else 'Transferred weights'})")
    print(f"Device:                  {args.device}")
    print(f"Max Steps:               {args.max_steps}")
    print(f"Effective Batch Size:    {args.micro_batch_size * args.grad_accum} (Micro={args.micro_batch_size}, Accum={args.grad_accum})")
    print(f"Learning Rate:           {args.lr} -> {args.min_lr} (Cosine with {args.warmup_steps} step warmup)")
    if not is_random:
        print(f"Warm-Start Checkpoint:   {args.warm_start_checkpoint}")
    print(f"Output Directory:        {args.output_dir}")
    print(f"Drive Backup Directory:  {args.drive_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        os.makedirs(args.drive_dir, exist_ok=True)
    except Exception:
        pass

    device = torch.device(args.device)

    # 1. Initialize Tokenizer & Model
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=1)
    tokenizer = AudioCodecTokenizer(audio_cfg, device=args.device)

    model_cfg = ModelConfig(d_model=1024, n_heads=16, n_layers=6, d_ff=4096, audio=audio_cfg)

    if is_random:
        print(f"\nInitializing {exp_code}: Random parameter initialization (Normal/Xavier, seed={args.seed}).")
        model = TextlessTamilSpeechLM_CB0(model_cfg).to(device)
    else:
        print(f"\nInitializing {exp_code}: Transferring compatible weights from {args.warm_start_checkpoint}...")
        if not os.path.exists(args.warm_start_checkpoint):
            raise FileNotFoundError(f"Warm-start checkpoint not found: {args.warm_start_checkpoint}")
        model = TextlessTamilSpeechLM_CB0.from_pretrained_multicodebook(args.warm_start_checkpoint, model_cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model Architecture: {model.__class__.__name__} ({total_params/1e6:.2f}M params, {trainable_params/1e6:.2f}M trainable)")

    # 2. Setup Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.min_lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )

    # 3. Load Validation Corpus (Full Held-Out Split + Diagnostic References)
    full_val_samples, diag_samples = load_validation_corpus(args.val_tokens_path, tokenizer, device)

    # Initial Validation Baseline
    print("\nEvaluating initial zero-step validation metrics...")
    init_metrics = evaluate_held_out(model, full_val_samples, diag_samples, tokenizer, device, args.output_dir, step=0)
    print(f"  [Step 0 Baseline] Val Loss: {init_metrics['val_loss']:.4f} | CB0 Acc: {init_metrics['val_accuracy_pct']:.2f}% | PPL: {init_metrics['val_perplexity']:.1f} (Evaluated on {init_metrics['val_sequences_evaluated']} sequences)")
    print(f"  [Step 0 Baseline Audio] RMS: {init_metrics['ar_rms']:.4f} | Silence: {init_metrics['ar_silence_pct']:.1f}% | Repetition: {init_metrics['ar_repetition_rate_pct']:.1f}% | Unique: {init_metrics['ar_unique_tokens']}/150")

    history = [init_metrics]
    history_json_path = os.path.join(args.output_dir, "evaluation_history.json")
    with open(history_json_path, 'w') as f:
        json.dump(history, f, indent=2)

    # 4. Data Pipeline Setup (Streaming IndicVoices-R_Tamil)
    print("\nConnecting to streaming Tamil training corpus...")
    dataset = load_dataset("SPRINGLab/IndicVoices-R_Tamil", split="train", streaming=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))
    data_iter = iter(dataset)

    # Mandatory held-out evaluation speakers to strictly avoid training contamination
    held_out_speaker_ids = {'S4258098500393504', 'S4259865400376309'}

    print("\n" + "=" * 80)
    print("BEGINNING TRAINING LOOP")
    print("=" * 80)

    model.train()
    step = 0
    accum_loss = 0.0
    accum_acc = 0.0
    micro_step = 0
    t0 = time.time()
    best_val_loss = init_metrics['val_loss']

    while step < args.max_steps:
        # Fetch valid sample
        try:
            item = next(data_iter)
        except StopIteration:
            data_iter = iter(dataset)
            item = next(data_iter)

        spk = str(item.get('speaker_id', ''))
        if spk in held_out_speaker_ids:
            continue

        audio_arr = item['audio']['array']
        dur = len(audio_arr) / 24000.0
        if dur < 1.0 or dur > 20.0:
            continue

        # Encode CB0 on-the-fly
        w_t = torch.from_numpy(audio_arr).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            toks_4cb = tokenizer.encode(w_t, 24000)
            toks_cb0 = toks_4cb[:, 0, :]  # (1, T)

        seq_len = toks_cb0.shape[-1]
        if seq_len < 30:
            continue
        if seq_len > model_cfg.max_seq_len:
            toks_cb0 = toks_cb0[:, :model_cfg.max_seq_len]

        # Construct causal inputs: BOS + toks[:-1] -> toks
        bos = torch.tensor([[model.bos_token_id]], dtype=torch.long, device=device)
        toks_bos = torch.cat([bos, toks_cb0], dim=-1)
        x_in = toks_bos[:, :-1]
        x_tgt = toks_bos[:, 1:]

        # Forward pass
        logits, loss, metrics = model(x_in, target_tokens=x_tgt)
        if loss is None or torch.isnan(loss):
            continue

        loss_scaled = loss / args.grad_accum
        loss_scaled.backward()

        accum_loss += loss.item()
        accum_acc += metrics['cb0_accuracy']
        micro_step += 1

        # Optimization step after accumulating gradients
        if micro_step % args.grad_accum == 0:
            step += 1

            # Update learning rate
            cur_lr = get_lr(step, args.warmup_steps, args.max_steps, args.min_lr, args.lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            mean_step_loss = accum_loss / args.grad_accum
            mean_step_acc = (accum_acc / args.grad_accum) * 100.0
            accum_loss = 0.0
            accum_acc = 0.0

            # Logging
            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - t0
                steps_per_sec = step / max(1.0, elapsed)
                ppl = math.exp(min(mean_step_loss, 20.0))
                print(f"Step {step:4d}/{args.max_steps} | Loss: {mean_step_loss:.4f} | Acc: {mean_step_acc:.2f}% | PPL: {ppl:.1f} | LR: {cur_lr:.2e} | Speed: {steps_per_sec:.2f} step/s")

            # Periodic Held-Out Validation
            if step % args.eval_every == 0 or step == args.max_steps:
                print(f"\n--- [Validation at Step {step}] ---")
                val_res = evaluate_held_out(model, full_val_samples, diag_samples, tokenizer, device, args.output_dir, step)
                print(f"  Val Loss: {val_res['val_loss']:.4f} | CB0 Acc: {val_res['val_accuracy_pct']:.2f}% | PPL: {val_res['val_perplexity']:.1f} ({val_res['val_sequences_evaluated']} sequences)")
                print(f"  Audio Metrics: RMS={val_res['ar_rms']:.4f} | Silence={val_res['ar_silence_pct']:.1f}% | Repetition={val_res['ar_repetition_rate_pct']:.1f}% | Unique={val_res['ar_unique_tokens']}/150")
                print(f"  Synthesized audio: {val_res['synth_wav']}")

                history.append(val_res)
                with open(history_json_path, 'w') as f:
                    json.dump(history, f, indent=2)

                is_best = val_res['val_loss'] < best_val_loss
                if is_best:
                    best_val_loss = val_res['val_loss']

                # Save Checkpoint
                ckpt_data = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model_cfg,
                    "val_metrics": val_res,
                    "experiment": exp_code,
                    "init_mode": "random" if is_random else "transferred",
                    "warm_start_checkpoint": None if is_random else args.warm_start_checkpoint
                }
                save_path = os.path.join(args.output_dir, f"tamil_speech_lm_cb0_step_{step}.pt")
                torch.save(ckpt_data, save_path)
                print(f"  ✓ Saved checkpoint: {save_path}")

                if is_best:
                    best_path = os.path.join(args.output_dir, "tamil_speech_lm_cb0_best.pt")
                    torch.save(ckpt_data, best_path)
                    print(f"  ★ New best model saved to: {best_path}")

                # Backup to Google Drive if accessible
                if os.path.exists(args.drive_dir):
                    try:
                        import shutil
                        shutil.copy(save_path, args.drive_dir)
                        if is_best:
                            shutil.copy(best_path, args.drive_dir)
                        shutil.copy(val_res['synth_wav'], args.drive_dir)
                        shutil.copy(val_res['uncond_wav'], args.drive_dir)
                        shutil.copy(val_res['tf_wav'], args.drive_dir)
                        shutil.copy(history_json_path, args.drive_dir)
                        print(f"  ✓ Backed up to Google Drive: {args.drive_dir}")
                    except Exception as e:
                        print(f"  Warning: Drive backup failed: {e}")

                model.train()
                print("-" * 60)

    print("\n" + "=" * 80)
    print("EXPERIMENT V2-P2.5-A TRAINING COMPLETE")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
