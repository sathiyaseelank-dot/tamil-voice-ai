"""
Weight-Space Programming (WSP) Engine for Textless Tamil Speech LM (V2).
Grafts pre-trained attention circuits, embeddings, and prediction heads from
MusicGen-small directly into the single-stream V2 architecture.
"""

import os
import torch
import torch.nn as nn
from typing import Optional

from config import AudioConfig, ModelConfig
from model.audio_transformer import TextlessTamilSpeechLM
from tokenizer.codec import AudioCodecTokenizer
from tokenizer.audio_io import load_audio, save_audio


def perform_wsp_surgery(
    donor_model_name: str = "facebook/musicgen-small",
    target_layers: int = 6,
    output_checkpoint_path: str = "checkpoints/tamil_speech_lm_wsp_grafted_v2.pt",
    device: str = "cpu"
) -> TextlessTamilSpeechLM:
    """
    Executes WSP surgery for V2 single-stream architecture.
    """
    from transformers import MusicgenForConditionalGeneration

    print("=" * 70)
    print("🔬 STARTING WEIGHT-SPACE PROGRAMMING (WSP) SURGERY (V2)")
    print("=" * 70)
    print(f"Donor Model: {donor_model_name}")

    donor = MusicgenForConditionalGeneration.from_pretrained(donor_model_name)
    donor.eval()
    donor_state = donor.state_dict()
    print("✓ Donor foundation model loaded.")

    dec_cfg = donor.config.decoder
    d_model = dec_cfg.hidden_size  # 1024
    n_heads = dec_cfg.num_attention_heads  # 16
    d_ff = dec_cfg.ffn_dim  # 4096
    num_codebooks = dec_cfg.num_codebooks  # 4

    audio_cfg = AudioConfig(num_codebooks=num_codebooks, codebook_size=1024)
    model_cfg = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=target_layers,
        d_ff=d_ff,
        max_seq_len=750,
        audio=audio_cfg
    )

    target_model = TextlessTamilSpeechLM(config=model_cfg).to(device)

    # 1. Embeddings Surgery
    print("\n[1/4] Grafting codebook embeddings...")
    with torch.no_grad():
        for k in range(num_codebooks):
            donor_embed = donor_state[f"decoder.model.decoder.embed_tokens.{k}.weight"]  # (2049, d_model)
            target_model.codebook_embeddings[k].weight.data[:1024, :] = donor_embed[:1024, :].clone()

            # Special tokens
            target_model.codebook_embeddings[k].weight.data[audio_cfg.pad_token_id, :] = 0.0
            bos_vec = donor_embed[2048, :] if donor_embed.shape[0] > 2048 else donor_embed[:1024, :].mean(dim=0)
            target_model.codebook_embeddings[k].weight.data[audio_cfg.bos_token_id, :] = bos_vec
            target_model.codebook_embeddings[k].weight.data[audio_cfg.eos_token_id, :] = -bos_vec
            target_model.codebook_embeddings[k].weight.data[audio_cfg.silence_token_id, :] = donor_embed[:1024, :].mean(dim=0) * 0.05
            target_model.codebook_embeddings[k].weight.data[audio_cfg.interrupt_token_id, :] = donor_embed[:1024, :].std(dim=0) * 2.0

    # 2. Transformer Layer Surgery
    print("\n[2/4] Grafting temporal causal attention and FFN circuits...")
    with torch.no_grad():
        for i in range(target_layers):
            target_layer = target_model.transformer.layers[i]

            q_w = donor_state[f"decoder.model.decoder.layers.{i}.self_attn.q_proj.weight"]
            k_w = donor_state[f"decoder.model.decoder.layers.{i}.self_attn.k_proj.weight"]
            v_w = donor_state[f"decoder.model.decoder.layers.{i}.self_attn.v_proj.weight"]
            target_layer.self_attn.in_proj_weight.copy_(torch.cat([q_w, k_w, v_w], dim=0))

            target_layer.self_attn.out_proj.weight.copy_(donor_state[f"decoder.model.decoder.layers.{i}.self_attn.out_proj.weight"])
            target_layer.norm1.weight.copy_(donor_state[f"decoder.model.decoder.layers.{i}.self_attn_layer_norm.weight"])
            target_layer.norm1.bias.copy_(donor_state[f"decoder.model.decoder.layers.{i}.self_attn_layer_norm.bias"])
            target_layer.linear1.weight.copy_(donor_state[f"decoder.model.decoder.layers.{i}.fc1.weight"])
            target_layer.linear2.weight.copy_(donor_state[f"decoder.model.decoder.layers.{i}.fc2.weight"])
            target_layer.norm2.weight.copy_(donor_state[f"decoder.model.decoder.layers.{i}.final_layer_norm.weight"])
            target_layer.norm2.bias.copy_(donor_state[f"decoder.model.decoder.layers.{i}.final_layer_norm.bias"])

        target_model.ln_f.weight.copy_(donor_state["decoder.model.decoder.layer_norm.weight"])
        target_model.ln_f.bias.copy_(donor_state["decoder.model.decoder.layer_norm.bias"])

    # 3. Depth Head Surgery
    print("\n[3/4] Grafting depth prediction heads and residual embeddings...")
    with torch.no_grad():
        for k in range(num_codebooks):
            donor_head_w = donor_state[f"decoder.lm_heads.{k}.weight"]  # (2048, d_model)
            target_model.head.heads[k].weight.data[:1024, :].copy_(donor_head_w[:1024, :])
            target_model.head.heads[k].weight.data[1024:, :].copy_(donor_head_w[1024:1029, :])

            if k > 0:
                embed_k = donor_state[f"decoder.model.decoder.embed_tokens.{k}.weight"]
                embed_prev = donor_state[f"decoder.model.decoder.embed_tokens.{k-1}.weight"]
                depth_w = embed_k[:audio_cfg.vocab_size, :] - embed_prev[:audio_cfg.vocab_size, :]
                target_model.head.depth_embeddings[k - 1].weight.copy_(depth_w)

    # 4. Verification & Save
    print("\n[4/4] Verifying parameter health and saving checkpoint...")
    total_params = sum(p.numel() for p in target_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f}M")
    assert not any(torch.isnan(p).any() for p in target_model.parameters()), "NaN in parameters!"

    os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint_path)), exist_ok=True)
    torch.save({
        "model_state_dict": target_model.state_dict(),
        "config": model_cfg,
        "donor": donor_model_name,
        "version": "v2"
    }, output_checkpoint_path)
    print(f"✓ V2 Grafted checkpoint saved: {output_checkpoint_path}")
    return target_model


if __name__ == "__main__":
    perform_wsp_surgery()
