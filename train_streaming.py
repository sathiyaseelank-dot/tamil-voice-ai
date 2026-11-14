"""
Streaming Training Script for Textless Tamil Speech-to-Speech AI Model.
Streams real Tamil audio on-the-fly from Hugging Face (e.g. SPRINGLab/IndicVoices-R_Tamil)
without downloading heavy datasets to disk. Designed for Google Colab and local training.
"""

import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset, Audio

from config import AudioConfig, ModelConfig
from model.audio_transformer import TextlessTamilSpeechLM
from tokenizer.codec import AudioCodecTokenizer
from tokenizer.audio_io import save_audio


def train_streaming(
    dataset_name: str = "SPRINGLab/IndicVoices-R_Tamil",
    split: str = "train",
    checkpoint_path: str = "checkpoints/tamil_speech_lm_wsp_grafted.pt",
    output_dir: str = "checkpoints_streaming",
    max_steps: int = 1000,
    lr: float = 2e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_every: int = 10,
    save_every: int = 200
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    print("=" * 70)
    print("🎧 STREAMING TEXTLESS TAMIL SPEECH TRAINING")
    print("=" * 70)
    print(f"Device:       {device}")
    print(f"Dataset:      {dataset_name} (split: {split}, streaming: True)")
    print(f"Base Weights: {checkpoint_path}")
    print(f"Target Steps: {max_steps}")
    print("=" * 70)

    # 1. Initialize Neural Codec
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    print("\n[1/4] Loading Neural Audio Codec (Encodec 24kHz)...")
    tokenizer = AudioCodecTokenizer(config=audio_cfg, device=str(device))

    # 2. Load Model & WSP Grafted Weights
    print("\n[2/4] Loading Speech Language Model...")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_cfg = ckpt["config"]
        model = TextlessTamilSpeechLM(config=model_cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found, initializing from config.")
        model_cfg = ModelConfig(
            d_model=1024,
            n_heads=16,
            n_layers=6,
            d_ff=4096,
            max_seq_len=750,
            audio=audio_cfg
        )
        model = TextlessTamilSpeechLM(config=model_cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model trainable parameters: {total_params / 1e6:.2f}M")

    # 3. Stream Hugging Face Audio Dataset
    print(f"\n[3/4] Connecting to Hugging Face Audio Stream: '{dataset_name}'...")
    hf_dataset = load_dataset(dataset_name, split=split, streaming=True)
    # Resample on-the-fly to 24,000 Hz
    hf_dataset = hf_dataset.cast_column("audio", Audio(sampling_rate=audio_cfg.sample_rate))
    data_iter = iter(hf_dataset)
    print("✓ Dataset stream connected with 24,000 Hz on-the-fly resampling.")

    # 4. Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-5)

    print("\n[4/4] Starting Streaming Training Loop...")
    model.train()
    step = 0
    running_loss = 0.0
    t0 = time.time()

    while step < max_steps:
        try:
            sample = next(data_iter)
        except StopIteration:
            print("Reached end of stream; restarting iterator...")
            data_iter = iter(hf_dataset)
            sample = next(data_iter)

        raw_audio = sample["audio"]["array"]
        if len(raw_audio) < audio_cfg.sample_rate * 1.0:
            # Skip very short audio (< 1 sec)
            continue

        transcript_preview = sample.get("text", "") or sample.get("sentence", "")

        # Convert to torch tensor: (1, 1, samples)
        wav = torch.from_numpy(raw_audio).float().unsqueeze(0).to(device)
        if wav.abs().max() > 0:
            wav = wav / (wav.abs().max() + 1e-6) * 0.9

        # Tokenize on-the-fly using Encodec (Pure audio tokens, zero text)
        with torch.no_grad():
            tokens = tokenizer.encode(wav, sample_rate=audio_cfg.sample_rate)  # (1, 4, seq_len)

        seq_len = tokens.shape[-1]
        if seq_len < 30:
            continue

        # Truncate if exceeds max_seq_len
        if seq_len > model_cfg.max_seq_len:
            tokens = tokens[:, :, :model_cfg.max_seq_len]
            seq_len = model_cfg.max_seq_len

        # Conversational continuation split:
        # First half of audio = User speech context
        # Second half of audio = Assistant response target
        split_idx = max(seq_len // 2, 10)
        u_tok = tokens[:, :, :split_idx]
        a_target_tok = tokens[:, :, split_idx:]

        u_len = u_tok.shape[-1]
        a_len = a_target_tok.shape[-1]
        total_len = u_len + a_len

        # Construct user stream (speaking during first half, silent during second half)
        user_stream = torch.full((1, audio_cfg.num_codebooks, total_len), audio_cfg.silence_token_id, dtype=torch.long, device=device)
        user_stream[:, :, :u_len] = u_tok

        # Construct assistant stream
        asst_full = torch.full((1, audio_cfg.num_codebooks, total_len), audio_cfg.silence_token_id, dtype=torch.long, device=device)
        asst_full[:, :, u_len:] = a_target_tok

        # Assistant shifted input (with BOS token at start)
        bos = torch.full((1, audio_cfg.num_codebooks, 1), audio_cfg.bos_token_id, dtype=torch.long, device=device)
        asst_in = torch.cat([bos, asst_full[:, :, :-1]], dim=-1)
        asst_target = asst_full

        # Forward pass & loss
        optimizer.zero_grad()
        logits, loss = model(user_stream, asst_in, target_assistant_tokens=asst_target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        step += 1

        if step % log_every == 0:
            avg_loss = running_loss / log_every
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            running_loss = 0.0
            preview = transcript_preview[:35] + "..." if len(transcript_preview) > 35 else transcript_preview
            print(f"[Step {step:04d}/{max_steps}] Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Speed: {steps_per_sec:.2f} steps/s | Tamil Audio: '{preview}'")

        if step % save_every == 0 or step == max_steps:
            save_path = os.path.join(output_dir, f"tamil_speech_lm_stream_step_{step}.pt")
            latest_path = os.path.join(output_dir, "tamil_speech_lm_stream_latest.pt")
            
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": model_cfg,
                "loss": avg_loss if 'avg_loss' in locals() else loss.item()
            }, save_path)
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": model_cfg
            }, latest_path)
            print(f"\n💾 Checkpoint saved: {save_path}")

            # Quick audio generation preview
            with torch.no_grad():
                model.eval()
                gen_frames = model.generate(u_tok, max_new_tokens=75, temperature=0.8, top_k=50)
                gen_wav = tokenizer.decode(gen_frames)
                preview_wav_path = os.path.join(output_dir, f"sample_generation_step_{step}.wav")
                save_audio(preview_wav_path, gen_wav, audio_cfg.sample_rate)
                print(f"🎵 Generated sample Tamil speech saved to: {preview_wav_path}\n")
                model.train()

    print("=" * 70)
    print("✅ STREAMING TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Training for Textless Tamil Voice AI")
    parser.add_argument("--dataset", type=str, default="SPRINGLab/IndicVoices-R_Tamil")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tamil_speech_lm_wsp_grafted.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints_streaming")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
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
        save_every=args.save_every
    )
