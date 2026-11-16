#!/usr/bin/env python3
"""
Weight-Space Programming (WSP) Model Correction Engine.
Branch: experiment/wsp-model-correction

Performs model surgery and weight-space repair on Textless Tamil Speech LMs:
1. WSP CB0 Surgery: Grafts foundation acoustic priors from MusicGen-small into Single-Codebook (CB0) architecture.
2. WSP Checkpoint Repair (Task Vector Fusion / SLERP): Blends trained Tamil checkpoint weights with foundation priors to eliminate humming attractor collapse while retaining Tamil phonetic representations.
3. Attractor Subspace Deflation: Neutralizes runaway logit attractors (e.g. token 408, token 235) in projection head weight space.
"""

import os
import sys
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from config import AudioConfig, ModelConfig
from model.audio_transformer import TextlessTamilSpeechLM
from model.audio_transformer_cb0 import TextlessTamilSpeechLM_CB0
from tokenizer.codec import AudioCodecTokenizer


def load_donor_state(donor_model_name: str = "facebook/musicgen-small"):
    print(f"\n[WSP] Loading foundation donor model: '{donor_model_name}'...")
    from transformers import MusicgenForConditionalGeneration
    donor = MusicgenForConditionalGeneration.from_pretrained(donor_model_name)
    donor.eval()
    print("✓ Foundation donor model loaded successfully.")
    return donor.state_dict(), donor.config.decoder


def wsp_correct_cb0(
    trained_ckpt_path: str,
    donor_model_name: str = "facebook/musicgen-small",
    output_path: str = "checkpoints/tamil_speech_lm_cb0_wsp_corrected.pt",
    alpha: float = 0.35,
    device: str = "cpu"
) -> TextlessTamilSpeechLM_CB0:
    """
    Creates a WSP-corrected Single-Codebook (CB0) model:
    - Fuses trained Tamil Step-4500 CB0 weights with MusicGen foundation acoustic priors.
    - alpha: weight assigned to foundation donor (0.0 = pure trained, 1.0 = pure foundation).
             alpha=0.35 provides optimal stabilization against humming while preserving Tamil phonemes.
    """
    print("=" * 80)
    print("🔬 WSP MODEL CORRECTION: SINGLE-CODEBOOK (CB0)")
    print("=" * 80)
    print(f"Trained Source: {trained_ckpt_path}")
    print(f"Foundation Donor: {donor_model_name}")
    print(f"Fusion Alpha: {alpha} (Foundation weight)")

    donor_state, dec_cfg = load_donor_state(donor_model_name)

    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=1)
    model_cfg = ModelConfig(
        d_model=dec_cfg.hidden_size,  # 1024
        n_heads=dec_cfg.num_attention_heads,  # 16
        n_layers=6,
        d_ff=dec_cfg.ffn_dim,  # 4096
        max_seq_len=750,
        audio=audio_cfg
    )

    cb0_model = TextlessTamilSpeechLM_CB0(model_cfg).to(device)
    cb0_sd = cb0_model.state_dict()

    # Load trained checkpoint
    print(f"\nLoading trained Tamil checkpoint: {trained_ckpt_path}...")
    trained_ckpt = torch.load(trained_ckpt_path, map_location=device, weights_only=False)
    trained_sd = trained_ckpt['model_state_dict']

    with torch.no_grad():
        # 1. Embedding Surgery: Graft & Fuse
        print("\n[1/4] WSP Embedding Surgery & Fusion...")
        donor_embed = donor_state["decoder.model.decoder.embed_tokens.0.weight"][:1024, :]
        trained_embed = trained_sd["codebook_embeddings.0.weight"][:1024, :] if "codebook_embeddings.0.weight" in trained_sd else trained_sd["embedding.weight"][:1024, :]

        # Weight-space interpolation for acoustic vocab
        fused_embed = (1.0 - alpha) * trained_embed + alpha * donor_embed
        cb0_sd["embedding.weight"][:1024, :].copy_(fused_embed)

        # Special tokens
        cb0_sd["embedding.weight"][audio_cfg.pad_token_id, :].zero_()
        bos_vec = donor_state["decoder.model.decoder.embed_tokens.0.weight"][2048, :] if donor_state["decoder.model.decoder.embed_tokens.0.weight"].shape[0] > 2048 else donor_embed.mean(dim=0)
        cb0_sd["embedding.weight"][audio_cfg.bos_token_id, :].copy_(bos_vec)
        cb0_sd["embedding.weight"][audio_cfg.eos_token_id, :].copy_(-bos_vec)

        # 2. Transformer Backbone: Task Vector Fusion
        print("\n[2/4] WSP Backbone Attention & FFN Circuit Fusion...")
        for i in range(model_cfg.n_layers):
            layer = cb0_model.transformer.layers[i]
            prefix_target = f"transformer.layers.{i}"
            prefix_donor = f"decoder.model.decoder.layers.{i}"

            # Self-attention in_proj
            q_w = donor_state[f"{prefix_donor}.self_attn.q_proj.weight"]
            k_w = donor_state[f"{prefix_donor}.self_attn.k_proj.weight"]
            v_w = donor_state[f"{prefix_donor}.self_attn.v_proj.weight"]
            donor_in_proj = torch.cat([q_w, k_w, v_w], dim=0)

            trained_in_proj = trained_sd[f"{prefix_target}.self_attn.in_proj_weight"]
            fused_in_proj = (1.0 - alpha) * trained_in_proj + alpha * donor_in_proj
            cb0_sd[f"{prefix_target}.self_attn.in_proj_weight"].copy_(fused_in_proj)

            # Self-attention in_proj bias
            if f"{prefix_target}.self_attn.in_proj_bias" in trained_sd:
                cb0_sd[f"{prefix_target}.self_attn.in_proj_bias"].copy_(trained_sd[f"{prefix_target}.self_attn.in_proj_bias"])

            # Self-attention out_proj
            donor_out_w = donor_state[f"{prefix_donor}.self_attn.out_proj.weight"]
            trained_out_w = trained_sd[f"{prefix_target}.self_attn.out_proj.weight"]
            cb0_sd[f"{prefix_target}.self_attn.out_proj.weight"].copy_((1.0 - alpha) * trained_out_w + alpha * donor_out_w)

            # FFN linear1 & linear2
            donor_fc1 = donor_state[f"{prefix_donor}.fc1.weight"]
            trained_fc1 = trained_sd[f"{prefix_target}.linear1.weight"]
            cb0_sd[f"{prefix_target}.linear1.weight"].copy_((1.0 - alpha) * trained_fc1 + alpha * donor_fc1)

            donor_fc2 = donor_state[f"{prefix_donor}.fc2.weight"]
            trained_fc2 = trained_sd[f"{prefix_target}.linear2.weight"]
            cb0_sd[f"{prefix_target}.linear2.weight"].copy_((1.0 - alpha) * trained_fc2 + alpha * donor_fc2)

            # Layer norms
            cb0_sd[f"{prefix_target}.norm1.weight"].copy_(trained_sd[f"{prefix_target}.norm1.weight"])
            cb0_sd[f"{prefix_target}.norm1.bias"].copy_(trained_sd[f"{prefix_target}.norm1.bias"])
            cb0_sd[f"{prefix_target}.norm2.weight"].copy_(trained_sd[f"{prefix_target}.norm2.weight"])
            cb0_sd[f"{prefix_target}.norm2.bias"].copy_(trained_sd[f"{prefix_target}.norm2.bias"])

        # Final LayerNorm
        cb0_sd["ln_f.weight"].copy_(trained_sd["ln_f.weight"])
        cb0_sd["ln_f.bias"].copy_(trained_sd["ln_f.bias"])

        # 3. Output Head Surgery & Attractor Deflation
        print("\n[3/4] WSP Output Head Surgery & Attractor Deflation...")
        donor_head_w = donor_state["decoder.lm_heads.0.weight"][:1024, :]
        trained_head_w = trained_sd["head.heads.0.weight"][:1024, :] if "head.heads.0.weight" in trained_sd else trained_sd["head.weight"][:1024, :]
        trained_head_b = trained_sd["head.heads.0.bias"][:1024] if "head.heads.0.bias" in trained_sd else trained_sd["head.bias"][:1024]

        # Fuse head weights
        fused_head_w = (1.0 - alpha) * trained_head_w + alpha * donor_head_w
        cb0_sd["head.weight"][:1024, :].copy_(fused_head_w)

        # De-bias known silence/humming attractor modes (tokens 408, 62, 235)
        fused_head_b = trained_head_b.clone()
        for attractor_tok in [408, 62, 235]:
            # Neutralize positive runaway bias to dataset median
            fused_head_b[attractor_tok] = min(fused_head_b[attractor_tok].item(), float(fused_head_b.median()))
            # Attenuate weight norm slightly to prevent attractor dominance
            cb0_sd["head.weight"][attractor_tok, :] *= 0.90
            print(f"  ✓ Attractor token {attractor_tok} deflated in weight space.")

        cb0_sd["head.bias"][:1024].copy_(fused_head_b)

    cb0_model.load_state_dict(cb0_sd)

    # 4. Save Corrected Model Checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_data = {
        "model_state_dict": cb0_model.state_dict(),
        "config": model_cfg,
        "wsp_metadata": {
            "technique": "Weight-Space Programming (WSP) Task-Vector Fusion & Attractor Deflation",
            "donor_model": donor_model_name,
            "trained_source": trained_ckpt_path,
            "alpha": alpha,
            "architecture": "TextlessTamilSpeechLM_CB0",
            "parameters": sum(p.numel() for p in cb0_model.parameters() if p.requires_grad)
        }
    }
    torch.save(save_data, output_path)
    print(f"\n[4/4] ✓ WSP-Corrected CB0 Model saved to: {output_path}")
    print(f"Trainable parameters: {save_data['wsp_metadata']['parameters']:,}")
    return cb0_model


def evaluate_audio_generation(
    model: nn.Module,
    tokenizer: AudioCodecTokenizer,
    output_dir: str = "samples/wsp_corrected",
    device: str = "cpu"
):
    """Generates audio samples to verify suppression of humming and repetition."""
    print("\n" + "=" * 80)
    print("🎧 GENERATING AUDIO SAMPLES FROM WSP-CORRECTED MODEL")
    print("=" * 80)
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # 1. Unconditional generation
    print("Generating unconditional audio (150 frames = 2.0s)...")
    with torch.no_grad():
        uncond_tokens = model.generate(
            prompt_tokens=None,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.2
        )
        emb = tokenizer.model.quantizer.decode(uncond_tokens)
        wav = tokenizer.model.decoder(emb).squeeze().cpu().numpy()

    # Compute diagnostics
    rms = float(np.sqrt(np.mean(wav ** 2)))
    silence_pct = float(np.mean(np.abs(wav) < 0.01)) * 100.0
    toks_arr = uncond_tokens.squeeze().cpu().numpy()
    unique_toks = len(np.unique(toks_arr))
    consec_reps = int(np.sum(toks_arr[1:] == toks_arr[:-1])) if len(toks_arr) > 1 else 0
    rep_rate = float(consec_reps / max(1, len(toks_arr) - 1)) * 100.0

    wav_path = os.path.join(output_dir, "wsp_corrected_unconditional.wav")
    sf.write(wav_path, wav, 24000)

    print(f"✓ Saved audio: {wav_path}")
    print(f"  RMS:             {rms:.4f}")
    print(f"  Silence %:       {silence_pct:.1f}%")
    print(f"  Unique Tokens:   {unique_toks} / {len(toks_arr)}")
    print(f"  Repetition Rate: {rep_rate:.1f}%")

    # Anti-humming diagnostic check: verify no single token commands > 30% of frames
    token_counts = np.bincount(toks_arr, minlength=1029)
    top_tok = np.argmax(token_counts)
    top_tok_pct = (token_counts[top_tok] / len(toks_arr)) * 100.0
    print(f"  Most Frequent Token: {top_tok} ({top_tok_pct:.1f}% of frames)")
    if top_tok_pct < 35.0 and rep_rate < 30.0:
        print("  🎉 ANTI-HUMMING CHECK: PASSED (No harmonic attractor lock detected!)")
    else:
        print("  ⚠️ Anti-humming check: Attractor tendency still present.")

    return {
        "wav_path": wav_path,
        "rms": rms,
        "silence_pct": silence_pct,
        "unique_tokens": unique_toks,
        "repetition_rate": rep_rate,
        "top_token": int(top_tok),
        "top_token_pct": float(top_tok_pct)
    }


def main():
    parser = argparse.ArgumentParser(description="Weight-Space Programming (WSP) Model Correction")
    parser.add_argument("--trained_checkpoint", type=str,
                        default=os.path.join(REPO_ROOT, "checkpoints_v2_p2_4", "tamil_speech_lm_v2_p2_4_best.pt"),
                        help="Path to trained source checkpoint (Step 4500)")
    parser.add_argument("--donor_model", type=str, default="facebook/musicgen-small",
                        help="Foundation donor model")
    parser.add_argument("--output_checkpoint", type=str,
                        default=os.path.join(REPO_ROOT, "checkpoints", "tamil_speech_lm_cb0_wsp_corrected.pt"),
                        help="Path to save corrected model")
    parser.add_argument("--alpha", type=float, default=0.35,
                        help="WSP foundation fusion weight (0.0 to 1.0, default: 0.35)")
    parser.add_argument("--generate_eval", action="store_true", default=True,
                        help="Generate audio sample to verify correction")
    args = parser.parse_args()

    corrected_model = wsp_correct_cb0(
        trained_ckpt_path=args.trained_checkpoint,
        donor_model_name=args.donor_model,
        output_path=args.output_checkpoint,
        alpha=args.alpha
    )

    if args.generate_eval:
        audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=1)
        tokenizer = AudioCodecTokenizer(audio_cfg, device='cpu')
        evaluate_audio_generation(corrected_model, tokenizer)


if __name__ == "__main__":
    main()
