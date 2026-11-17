#!/usr/bin/env python3
"""
Streaming Training Script for S2S V2 Milestone A:
Genuine Textless Tamil Speech-to-Speech Model.

Optimized for:
- Google Colab GPU (T4, V100, A100, L4)
- Hugging Face Streaming Dataset (`SPRINGLab/IndicVoices-R_Tamil`)
- Continuous MMS-to-Encodec timestamp alignment
- Two-Stage Codebook Training Strategy (Stage 1: CB0 Phonetics, Stage 2: Residual Harmonics)
- Google Drive Checkpointing
"""

import os
import sys
import math
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import numpy as np
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Model

# Add repository root to path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from config import AudioConfig, ModelConfig
from tokenizer.codec import AudioCodecTokenizer
from model.s2s_transformer import S2STokenPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Train S2S V2 Model on Streaming Tamil Audio")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Training Stage: 1 = CB0 Phonetic Backbone, 2 = Residual RVQ Depth Head")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Linear warmup steps")
    parser.add_argument("--eval_interval", type=int, default=250,
                        help="Evaluation interval in steps")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Checkpoint save interval in steps")
    parser.add_argument("--save_dir", type=str, default="checkpoints_s2s_v2",
                        help="Directory to save checkpoints")
    parser.add_argument("--init_type", type=str, default="mms", choices=["mms", "wsp", "random"],
                        help="Backbone initialization strategy")
    parser.add_argument("--wsp_checkpoint", type=str, default="checkpoints/tamil_speech_lm_v2_p2_4_wsp_corrected.pt",
                        help="Path to WSP checkpoint for WSP initialization")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--dataset_name", type=str, default="SPRINGLab/IndicVoices-R_Tamil",
                        help="Hugging Face streaming dataset name")
    parser.add_argument("--clip_dur_sec", type=float, default=4.0,
                        help="Audio chunk duration in seconds per training step")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 mixed precision if available")
    return parser.parse_args()


def load_mms_frontend(device: torch.device):
    """Loads frozen MMS-300M SSL frontend."""
    print("Loading frozen facebook/mms-300m frontend...")
    mms = Wav2Vec2Model.from_pretrained("facebook/mms-300m")
    mms.eval()
    for p in mms.parameters():
        p.requires_grad = False
    mms = mms.to(device)
    return mms


def initialize_weights(model: S2STokenPredictor, init_type: str, wsp_checkpoint_path: str, mms_model: Wav2Vec2Model):
    """Initializes model parameters based on specified strategy."""
    if init_type == "mms":
        print("Initializing S2S Transformer backbone from MMS-300M layers 0-5...")
        with torch.no_grad():
            for i in range(min(len(model.transformer.layers), len(mms_model.encoder.layers))):
                src_layer = mms_model.encoder.layers[i]
                dst_layer = model.transformer.layers[i]
                # QKV weights
                dst_layer.self_attn.in_proj_weight.copy_(
                    torch.cat([
                        src_layer.attention.q_proj.weight,
                        src_layer.attention.k_proj.weight,
                        src_layer.attention.v_proj.weight
                    ], dim=0)
                )
                dst_layer.self_attn.in_proj_bias.copy_(
                    torch.cat([
                        src_layer.attention.q_proj.bias,
                        src_layer.attention.k_proj.bias,
                        src_layer.attention.v_proj.bias
                    ], dim=0)
                )
                # Out projection
                dst_layer.self_attn.out_proj.weight.copy_(src_layer.attention.out_proj.weight)
                dst_layer.self_attn.out_proj.bias.copy_(src_layer.attention.out_proj.bias)
                # Feed-forward
                dst_layer.linear1.weight.copy_(src_layer.feed_forward.intermediate_dense.weight)
                dst_layer.linear1.bias.copy_(src_layer.feed_forward.intermediate_dense.bias)
                dst_layer.linear2.weight.copy_(src_layer.feed_forward.output_dense.weight)
                dst_layer.linear2.bias.copy_(src_layer.feed_forward.output_dense.bias)
                # LayerNorms
                dst_layer.norm1.weight.copy_(src_layer.layer_norm.weight)
                dst_layer.norm1.bias.copy_(src_layer.layer_norm.bias)
                dst_layer.norm2.weight.copy_(src_layer.final_layer_norm.weight)
                dst_layer.norm2.bias.copy_(src_layer.final_layer_norm.bias)
        print("✓ MMS backbone weights transferred successfully.")

    elif init_type == "wsp":
        if os.path.exists(wsp_checkpoint_path):
            print(f"Loading WSP donor checkpoint from {wsp_checkpoint_path}...")
            ckpt = torch.load(wsp_checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            with torch.no_grad():
                # Transfer embeddings to depth embeddings if compatible
                for k in range(min(model.head.num_codebooks - 1, 3)):
                    key = f"emb.{k}.weight"
                    if key in state_dict and state_dict[key].shape == model.head.depth_embeddings[k].weight.shape:
                        model.head.depth_embeddings[k].weight.copy_(state_dict[key])
                        print(f"✓ Transferred WSP depth embedding CB{k+1}")
            print("✓ WSP weights transferred.")
        else:
            print(f"⚠️ WSP checkpoint not found at {wsp_checkpoint_path}. Using standard init.")
    else:
        print("Using standard random initialization (Kaiming / Xavier).")


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    """Linear warmup followed by cosine decay."""
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    out_samples_dir = os.path.join(args.save_dir, "samples")
    os.makedirs(out_samples_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"================================================================================")
    print(f"🎙️ S2S V2 MILESTONE A STREAMING TRAINING")
    print(f"Device: {device} | Stage: {args.stage} | Max Steps: {args.max_steps} | Init: {args.init_type}")
    print(f"Save Directory: {args.save_dir}")
    print(f"================================================================================")

    # 1. Initialize Tokenizer & MMS
    tokenizer = AudioCodecTokenizer().to(device)
    mms_model = load_mms_frontend(device)

    # 2. Build S2STokenPredictor Model
    model = S2STokenPredictor(
        d_model=1024,
        n_heads=16,
        n_layers=6,
        d_ff=4096,
        num_codebooks=4,
        vocab_size=1029
    ).to(device)

    # 3. Initialize weights
    initialize_weights(model, args.init_type, args.wsp_checkpoint, mms_model)

    # 4. Stage-specific loss weights
    if args.stage == 1:
        # Stage 1: CB0 Phonetic Backbone Focus
        # Dedicate 100% of loss to CB0, zeroing residual distraction
        loss_weights = [1.0, 0.0, 0.0, 0.0]
        print("✓ Stage 1 Mode: Loss weights = [CB0: 1.0, CB1: 0.0, CB2: 0.0, CB3: 0.0]")
    else:
        # Stage 2: Residual RVQ Harmonics
        loss_weights = [0.2, 1.0, 0.5, 0.25]
        print("✓ Stage 2 Mode: Loss weights = [CB0: 0.2, CB1: 1.0, CB2: 0.5, CB3: 0.25]")

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay
    )
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, args.max_steps)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"✓ Resumed at step {start_step}")

    # 6. Stream Hugging Face Dataset
    print(f"Connecting to streaming dataset: {args.dataset_name}...")
    try:
        ds = load_dataset(args.dataset_name, split="train", streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=24000))
        ds_iter = iter(ds)
        print("✓ Dataset stream connected.")
    except Exception as e:
        print(f"❌ Failed to connect to Hugging Face streaming dataset: {e}")
        print("Please check your internet connection or Hugging Face access.")
        sys.exit(1)

    # Prepare validation sample for audio monitoring
    val_sample_wav = None
    try:
        sample_item = next(ds_iter)
        val_sample_wav = sample_item["audio"]["array"]
        if val_sample_wav.ndim > 1:
            val_sample_wav = val_sample_wav.mean(axis=0)
        print(f"✓ Fixed validation audio sample acquired ({len(val_sample_wav)/24000.0:.2f}s).")
    except Exception as e:
        print(f"Warning: Could not fetch initial sample for validation: {e}")

    # 7. Training Loop
    model.train()
    step = start_step
    running_loss = 0.0
    running_cb0_acc = 0.0
    start_time = time.time()
    clip_samples_24k = int(args.clip_dur_sec * 24000)

    print(f"\n🚀 Commencing Milestone A Training...")
    while step < args.max_steps:
        try:
            item = next(ds_iter)
        except StopIteration:
            ds_iter = iter(ds)
            item = next(ds_iter)

        raw_audio = item["audio"]["array"]
        if len(raw_audio) < 24000:  # Skip clips shorter than 1.0 second
            continue

        if raw_audio.ndim > 1:
            raw_audio = raw_audio.mean(axis=0)

        # Slice to fixed window if longer
        if len(raw_audio) > clip_samples_24k:
            max_start = len(raw_audio) - clip_samples_24k
            st = np.random.randint(0, max_start)
            raw_audio = raw_audio[st:st + clip_samples_24k]

        t_24k = torch.tensor(raw_audio, dtype=torch.float32, device=device).unsqueeze(0)
        # Resample to 16 kHz for MMS
        t_16k = torchaudio.functional.resample(t_24k, 24000, 16000)

        # Normalize audio peak
        if t_24k.abs().max() > 0:
            t_24k = t_24k / t_24k.abs().max() * 0.95
        if t_16k.abs().max() > 0:
            t_16k = t_16k / t_16k.abs().max() * 0.95

        # 1. Extract Target Encodec Tokens (Frozen)
        with torch.no_grad():
            tgt_tokens = tokenizer.encode(t_24k)  # (1, 4, T_codec)
            T_codec = tgt_tokens.shape[-1]
            if T_codec < 10:
                continue

            # 2. Extract MMS Representations (Frozen)
            mms_out = mms_model(t_16k)
            h_mms = mms_out.last_hidden_state  # (1, T_mms, 1024)

        # 3. Forward S2STokenPredictor
        optimizer.zero_grad()
        logits = model(h_mms, target_len=T_codec, target_tokens=tgt_tokens)  # (1, 4, T_codec, 1029)

        # 4. Compute Loss with Stage Weights
        loss = 0.0
        cb_losses = []
        cb_accs = []
        for k in range(4):
            l_k = logits[:, k, :, :].reshape(-1, model.vocab_size)
            t_k = tgt_tokens[:, k, :].reshape(-1)
            ce_k = F.cross_entropy(l_k, t_k)
            cb_losses.append(ce_k.item())
            if loss_weights[k] > 0:
                loss = loss + loss_weights[k] * ce_k
            
            with torch.no_grad():
                acc_k = (torch.argmax(l_k, dim=-1) == t_k).float().mean().item() * 100.0
                cb_accs.append(acc_k)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        step += 1
        running_loss += loss.item()
        running_cb0_acc += cb_accs[0]

        # Logging
        if step % 50 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = 50.0 / elapsed
            cur_lr = scheduler.get_last_lr()[0]
            print(f"Step {step:6d}/{args.max_steps} | Loss: {running_loss/50.0:.4f} | "
                  f"CB0 Acc: {running_cb0_acc/50.0:.2f}% | CB1: {cb_accs[1]:.1f}% | "
                  f"LR: {cur_lr:.2e} | Speed: {steps_per_sec:.2f} it/s")
            running_loss = 0.0
            running_cb0_acc = 0.0
            start_time = time.time()

        # Evaluation & Sample Audio Generation
        if step % args.eval_interval == 0 and val_sample_wav is not None:
            model.eval()
            with torch.no_grad():
                v_24k = torch.tensor(val_sample_wav[:clip_samples_24k], dtype=torch.float32, device=device).unsqueeze(0)
                v_16k = torchaudio.functional.resample(v_24k, 24000, 16000)
                v_tokens = tokenizer.encode(v_24k)
                v_h_mms = mms_model(v_16k).last_hidden_state

                # Generate tokens
                v_pred_4cb = model.generate_tokens(v_h_mms, target_len=v_tokens.shape[-1], cb0_only=False)
                v_pred_cb0 = model.generate_tokens(v_h_mms, target_len=v_tokens.shape[-1], cb0_only=True)

                # Decode audio via frozen Encodec decoder
                wav_cb0 = tokenizer.decode(v_pred_cb0).squeeze().cpu().numpy()
                wav_4cb = tokenizer.decode(v_pred_4cb).squeeze().cpu().numpy()

                # Save audio samples
                sf.write(os.path.join(out_samples_dir, f"step_{step}_cb0_only.wav"), wav_cb0, 24000)
                sf.write(os.path.join(out_samples_dir, f"step_{step}_4cb.wav"), wav_4cb, 24000)

                # Compute accuracy on validation sample
                val_cb0_acc = (v_pred_4cb[:, 0, :] == v_tokens[:, 0, :]).float().mean().item() * 100.0
                print(f"  📊 [EVAL STEP {step}] Val CB0 Acc: {val_cb0_acc:.2f}% | "
                      f"Saved audio: step_{step}_cb0_only.wav & step_{step}_4cb.wav")
            model.train()

        # Checkpointing
        if step % args.save_interval == 0 or step == args.max_steps:
            save_path = os.path.join(args.save_dir, f"s2s_v2_stage{args.stage}_step_{step}.pt")
            torch.save({
                "step": step,
                "stage": args.stage,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args)
            }, save_path)
            # Also save latest pointer
            latest_path = os.path.join(args.save_dir, f"s2s_v2_stage{args.stage}_latest.pt")
            torch.save({
                "step": step,
                "stage": args.stage,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args)
            }, latest_path)
            print(f"  💾 Saved checkpoint: {save_path}")

    print("\n🏁 MILESTONE A STREAMING TRAINING FINISHED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
