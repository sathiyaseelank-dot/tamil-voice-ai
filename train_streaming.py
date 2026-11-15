"""
Streaming Training Script for Textless Tamil Autoregressive LM (V2).
Implements full-sequence next-frame audio prediction: X[:, :, 0:T-1] -> X[:, :, 1:T] (with BOS).
Provides detailed per-codebook loss, accuracy, entropy, and gradient norm tracking.
"""

import os
import sys
import argparse
import time
import math
import random
from typing import Optional, Dict
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset, Audio

from config import AudioConfig, ModelConfig
from model.audio_transformer import TextlessTamilSpeechLM
from tokenizer.codec import AudioCodecTokenizer
from tokenizer.audio_io import save_audio


def evaluate_held_out(
    model: TextlessTamilSpeechLM,
    held_out_tokens: list,
    device: torch.device
) -> dict:
    """
    Evaluates model on a fixed held-out set of real Tamil speech token sequences.
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    cb_accs = [0.0] * model.num_codebooks
    cb_losses = [0.0] * model.num_codebooks
    count = 0

    with torch.no_grad():
        for tok in held_out_tokens:
            tok = tok.to(device)
            seq_len = tok.shape[-1]
            if seq_len < 20:
                continue

            bos = torch.full((1, model.num_codebooks, 1), model.bos_token_id, dtype=torch.long, device=device)
            tok_bos = torch.cat([bos, tok], dim=-1)
            x_in = tok_bos[:, :, :-1]
            x_target = tok_bos[:, :, 1:]

            _, loss, metrics = model(x_in, target_tokens=x_target)
            if loss is not None and metrics is not None:
                total_loss += loss.item()
                total_acc += metrics.get("speech_accuracy", 0.0)
                for k in range(model.num_codebooks):
                    cb_losses[k] += metrics.get(f"loss_cb{k}", 0.0)
                    cb_accs[k] += metrics.get(f"acc_cb{k}", 0.0)
                count += 1

    model.train()
    if count == 0:
        return {}

    eval_results = {
        "val_loss": total_loss / count,
        "val_speech_acc": total_acc / count,
    }
    for k in range(model.num_codebooks):
        eval_results[f"val_loss_cb{k}"] = cb_losses[k] / count
        eval_results[f"val_acc_cb{k}"] = cb_accs[k] / count

    return eval_results


def train_streaming(
    dataset_name: str = "SPRINGLab/IndicVoices-R_Tamil",
    split: str = "train",
    checkpoint_path: Optional[str] = None,
    output_dir: str = "checkpoints_v2",
    max_steps: int = 300,
    lr: float = 2e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_every: int = 10,
    save_every: int = 50,
    seed: int = 42
):
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)

    print("=" * 75)
    print("🎧 TEXTLESS TAMIL SPEECH LM V2: AUTOREGRESSIVE TRAINING")
    print("=" * 75)
    print(f"Device:       {device}")
    print(f"Dataset:      {dataset_name} (split: {split}, streaming: True)")
    print(f"Output Dir:   {output_dir}")
    print(f"Target Steps: {max_steps}")
    print(f"Seed:         {seed}")
    print("=" * 75)

    # 1. Neural Codec
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    print("\n[1/4] Loading Neural Audio Codec (Encodec 24kHz)...")
    tokenizer = AudioCodecTokenizer(config=audio_cfg, device=str(device))

    # 2. Model Initialization
    print("\n[2/4] Initializing V2 Autoregressive Speech Language Model...")
    model_cfg = ModelConfig(
        d_model=1024,
        n_heads=16,
        n_layers=6,
        d_ff=4096,
        max_seq_len=750,
        audio=audio_cfg
    )
    model = TextlessTamilSpeechLM(config=model_cfg).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading base weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # Filter matching keys
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        model_state.update(filtered)
        model.load_state_dict(model_state)
        print(f"✓ Loaded {len(filtered)}/{len(model_state)} matching parameter tensors.")
    else:
        print("Initializing weights from configuration.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params / 1e6:.2f}M")

    # 3. Stream Hugging Face Audio Dataset
    print(f"\n[3/4] Connecting to dataset stream '{dataset_name}'...")
    hf_dataset = load_dataset(dataset_name, split=split, streaming=True)
    hf_dataset = hf_dataset.cast_column("audio", Audio(decode=False))
    data_iter = iter(hf_dataset)
    print("✓ Dataset stream connected.")

    # Cache fixed held-out set for consistent evaluation
    print("Caching fixed held-out audio validation samples...")
    held_out_tokens = []
    held_out_transcripts = []
    while len(held_out_tokens) < 5:
        s = next(data_iter)
        a_entry = s.get("audio", {})
        if "array" in a_entry and a_entry["array"] is not None:
            r_aud = np.array(a_entry["array"])
            sr_h = a_entry.get("sampling_rate", audio_cfg.sample_rate)
        elif "bytes" in a_entry and a_entry["bytes"] is not None:
            import io, soundfile as sf
            r_aud, sr_h = sf.read(io.BytesIO(a_entry["bytes"]))
        else:
            continue

        if len(r_aud) < sr_h * 1.5:
            continue

        w = torch.from_numpy(r_aud).float()
        if w.ndim == 1:
            w = w.unsqueeze(0).unsqueeze(0)
        elif w.ndim == 2:
            w = w.mean(dim=-1, keepdim=True).transpose(0, 1).unsqueeze(0)

        if sr_h != audio_cfg.sample_rate:
            import torchaudio
            w = torchaudio.functional.resample(w, sr_h, audio_cfg.sample_rate)

        w = w.to(device)
        if w.abs().max() > 0:
            w = w / (w.abs().max() + 1e-6) * 0.9

        with torch.no_grad():
            tok = tokenizer.encode(w, sample_rate=audio_cfg.sample_rate)
        if tok.shape[-1] > model_cfg.max_seq_len:
            tok = tok[:, :, :model_cfg.max_seq_len]

        held_out_tokens.append(tok)
        held_out_transcripts.append(s.get("text", "") or s.get("sentence", ""))

    print(f"✓ Cached {len(held_out_tokens)} fixed held-out validation sequences.")

    # 4. Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-5)

    print("\n[4/4] Starting Autoregressive Training Loop...")
    model.train()
    step = 0
    running_loss = 0.0
    running_acc = 0.0
    running_entropy = 0.0
    t0 = time.time()

    while step < max_steps:
        try:
            sample = next(data_iter)
        except StopIteration:
            data_iter = iter(hf_dataset)
            sample = next(data_iter)

        audio_entry = sample.get("audio", {})
        if "array" in audio_entry and audio_entry["array"] is not None:
            raw_audio = np.array(audio_entry["array"])
            sr = audio_entry.get("sampling_rate", audio_cfg.sample_rate)
        elif "bytes" in audio_entry and audio_entry["bytes"] is not None:
            import io, soundfile as sf
            raw_audio, sr = sf.read(io.BytesIO(audio_entry["bytes"]))
        else:
            continue

        if len(raw_audio) < sr * 1.0:
            continue

        transcript_preview = sample.get("text", "") or sample.get("sentence", "")

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

        # Encode full Tamil audio into discrete RVQ tokens: (1, 4, seq_len)
        with torch.no_grad():
            tokens = tokenizer.encode(wav, sample_rate=audio_cfg.sample_rate)

        seq_len = tokens.shape[-1]
        if seq_len < 30:
            continue

        if seq_len > model_cfg.max_seq_len:
            tokens = tokens[:, :, :model_cfg.max_seq_len]
            seq_len = model_cfg.max_seq_len

        # Full-sequence autoregressive shift with BOS:
        # Input:  [BOS, tokens[:, :, 0:T-1]]
        # Target: [tokens[:, :, 0:T]]
        bos = torch.full((1, audio_cfg.num_codebooks, 1), audio_cfg.bos_token_id, dtype=torch.long, device=device)
        tokens_with_bos = torch.cat([bos, tokens], dim=-1)

        x_in = tokens_with_bos[:, :, :-1]     # (1, 4, seq_len)
        x_target = tokens_with_bos[:, :, 1:]  # (1, 4, seq_len) (the exact real audio tokens)

        optimizer.zero_grad()
        logits, loss, metrics = model(x_in, target_tokens=x_target)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_acc += metrics.get("speech_accuracy", 0.0)
        running_entropy += metrics.get("entropy", 0.0)
        step += 1

        if step % log_every == 0:
            avg_loss = running_loss / log_every
            avg_acc = (running_acc / log_every) * 100
            avg_entropy = running_entropy / log_every
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed

            running_loss = 0.0
            running_acc = 0.0
            running_entropy = 0.0

            cb_losses_str = " ".join([f"L{k}:{metrics.get(f'loss_cb{k}', 0.0):.2f}" for k in range(audio_cfg.num_codebooks)])
            cb_accs_str = " ".join([f"A{k}:{metrics.get(f'acc_cb{k}', 0.0)*100:.1f}%" for k in range(audio_cfg.num_codebooks)])

            print(
                f"[Step {step:04d}/{max_steps}] Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}% | "
                f"Entropy: {avg_entropy:.2f} | GradNorm: {grad_norm:.2f} | {cb_losses_str} | {cb_accs_str}"
            )

        # Evaluation, checkpointing, and generation preview
        if step % save_every == 0 or step == max_steps:
            val_results = evaluate_held_out(model, held_out_tokens, device)
            val_loss = val_results.get("val_loss", 0.0)
            val_acc = val_results.get("val_speech_acc", 0.0) * 100

            print(f"\n--- [Step {step}] Held-Out Validation ---")
            print(f"  Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}% (Random chance: ~0.098%)")
            for k in range(audio_cfg.num_codebooks):
                print(f"    CB{k} Loss: {val_results.get(f'val_loss_cb{k}', 0.0):.4f} | Accuracy: {val_results.get(f'val_acc_cb{k}', 0.0)*100:.2f}%")

            # Save checkpoint
            save_path = os.path.join(output_dir, f"tamil_speech_lm_v2_step_{step}.pt")
            latest_path = os.path.join(output_dir, "tamil_speech_lm_v2_latest.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": model_cfg,
                "train_loss": avg_loss if 'avg_loss' in locals() else loss.item(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "version": "v2"
            }, save_path)
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": model_cfg,
                "version": "v2"
            }, latest_path)
            print(f"💾 Checkpoint saved: {save_path}")

            # Audio generation preview
            with torch.no_grad():
                model.eval()
                gen_frames = model.generate(
                    prompt_tokens=None,  # Unconditional from BOS
                    max_new_tokens=150,  # 2.0s
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.15
                )
                gen_wav = tokenizer.decode(gen_frames).squeeze(0).cpu()
                preview_wav_path = os.path.join(output_dir, f"sample_v2_step_{step}.wav")
                save_audio(preview_wav_path, gen_wav, audio_cfg.sample_rate)

                # Token diagnostics
                gen_np = gen_frames.squeeze(0).cpu().numpy()
                for k in range(audio_cfg.num_codebooks):
                    cb_t = gen_np[k]
                    unq = len(np.unique(cb_t))
                    rep = (cb_t[1:] == cb_t[:-1]).mean() if len(cb_t) > 1 else 0.0
                    print(f"  Preview CB{k}: {unq}/150 unique tokens | Consec Repetition: {rep*100:.1f}%")

                print(f"🎵 Generated sample saved to: {preview_wav_path}\n")
                model.train()

    print("=" * 75)
    print("✅ V2 CONTROLLED VALIDATION RUN COMPLETED!")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Training V2")
    parser.add_argument("--dataset", type=str, default="SPRINGLab/IndicVoices-R_Tamil")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints_v2")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_streaming(
        dataset_name=args.dataset,
        split=args.split,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        lr=args.lr,
        device=args.device,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed
    )
