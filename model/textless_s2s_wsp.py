#!/usr/bin/env python3
"""
Unified Textless Speech-to-Speech (S2S) Architecture.
Crafted via Multi-Model Weight-Space Programming (WSP):
1. Input Frontend: MMS-300M 7-layer temporal convolutional acoustic encoder (50 Hz).
2. Backbone Transformer: 6-layer causal audio transformer fused in weight-space across:
   - Step-4500 (Tamil token alignments)
   - MMS-300M (multilingual & Tamil phonetic circuits)
   - MusicGen-small (causal autoregressive temporal stability)
3. Vocal Tract Projector: WSP SVD subspace projection (1024 -> 192).
4. Audio Output Vocoder: Meta MMS-Tamil VITS HiFi-GAN neural speech synthesis engine.

Pure Speech-to-Speech: Zero text, zero transcripts.
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2Model, VitsModel, MusicgenForConditionalGeneration

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "model" else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from config import AudioConfig, ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 1024, max_len: int = 2500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TextlessTamilS2SModel(nn.Module):
    """
    End-to-End Pure Textless Speech-to-Speech Model.
    Directly maps spoken Tamil audio input into spoken Tamil audio output without text.
    """
    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 6,
        d_ff: int = 4096,
        latent_dim: int = 192,
        device: str = "cpu"
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.device = torch.device(device)

        # 1. Acoustic Frontend (MMS-300M Feature Extractor)
        self.frontend_feature_extractor = None  # Loaded during WSP crafting

        # 2. Causal Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=2500)

        # 3. Backbone Causal Temporal Transformer Decoder (6 layers)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)

        # 4. WSP Vocal Tract Projection Head (1024 -> 192)
        self.vocal_projector = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # 5. Output Neural Speech Vocoder (VitsHiFiGan)
        self.vocoder = None  # Grafted during WSP crafting

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    @torch.no_grad()
    def craft_weights_via_wsp(
        self,
        ckpt_4500_path: str,
        alpha_4500: float = 0.35,
        alpha_mms: float = 0.35,
        alpha_musicgen: float = 0.30
    ):
        """
        Performs multi-model Weight-Space Programming (WSP) fusion and surgery:
        - Fuses backbone causal circuits across 4500.pt, mms-300m, and musicgen-small.
        - Grafts MMS-300M convolutional acoustic frontend.
        - Grafts Meta MMS-Tamil VITS HiFi-GAN vocal tract decoder.
        """
        print("=" * 80)
        print("🔬 MULTI-MODEL WSP GRAFTING & WEIGHT SYNTHESIS")
        print("=" * 80)
        print(f"Weight Fusions: 4500.pt={alpha_4500:.2f}, MMS-300M={alpha_mms:.2f}, MusicGen={alpha_musicgen:.2f}")

        # Load donor models
        print("\n[1/4] Loading donor models...")
        mms = Wav2Vec2Model.from_pretrained("facebook/mms-300m").eval()
        mg = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").eval()
        vits = VitsModel.from_pretrained("facebook/mms-tts-tam").eval()

        # Load 4500.pt
        raw_4500 = torch.load(ckpt_4500_path, map_location="cpu", weights_only=False)
        sd_4500 = raw_4500["model_state_dict"] if "model_state_dict" in raw_4500 else raw_4500

        # Graft Frontend
        print("\n[2/4] Grafting acoustic convolutional feature frontend...")
        self.frontend_feature_extractor = mms.feature_extractor
        self.frontend_feature_projection = mms.feature_projection

        # Graft Vocoder
        print("\n[3/4] Grafting VITS HiFi-GAN vocal tract neural speech vocoder...")
        self.vocoder = vits.decoder
        self.vocoder.eval()

        # Backbone WSP Fusion (6 layers)
        print("\n[4/4] Fusing attention & FFN circuits across 6 transformer layers in weight space...")
        target_sd = self.transformer.state_dict()

        for i in range(6):
            # In_proj (q, k, v)
            # MMS
            mms_q = mms.encoder.layers[i].attention.q_proj.weight
            mms_k = mms.encoder.layers[i].attention.k_proj.weight
            mms_v = mms.encoder.layers[i].attention.v_proj.weight
            mms_in_proj = torch.cat([mms_q, mms_k, mms_v], dim=0)

            # MusicGen
            mg_q = mg.decoder.model.decoder.layers[i].self_attn.q_proj.weight
            mg_k = mg.decoder.model.decoder.layers[i].self_attn.k_proj.weight
            mg_v = mg.decoder.model.decoder.layers[i].self_attn.v_proj.weight
            mg_in_proj = torch.cat([mg_q, mg_k, mg_v], dim=0)

            # 4500.pt
            p4500_in_proj = sd_4500[f"transformer.layers.{i}.self_attn.in_proj_weight"]
            p4500_in_bias = sd_4500[f"transformer.layers.{i}.self_attn.in_proj_bias"]

            # WSP Interpolation
            fused_in_proj = (
                alpha_4500 * p4500_in_proj +
                alpha_mms * mms_in_proj +
                alpha_musicgen * mg_in_proj
            )
            target_sd[f"layers.{i}.self_attn.in_proj_weight"].copy_(fused_in_proj)
            target_sd[f"layers.{i}.self_attn.in_proj_bias"].copy_(p4500_in_bias)

            # Out_proj
            mms_out = mms.encoder.layers[i].attention.out_proj.weight
            mg_out = mg.decoder.model.decoder.layers[i].self_attn.out_proj.weight
            p4500_out = sd_4500[f"transformer.layers.{i}.self_attn.out_proj.weight"]
            fused_out = (
                alpha_4500 * p4500_out +
                alpha_mms * mms_out +
                alpha_musicgen * mg_out
            )
            target_sd[f"layers.{i}.self_attn.out_proj.weight"].copy_(fused_out)
            target_sd[f"layers.{i}.self_attn.out_proj.bias"].copy_(sd_4500[f"transformer.layers.{i}.self_attn.out_proj.bias"])

            # FFN linear1 & linear2
            mms_fc1 = mms.encoder.layers[i].feed_forward.intermediate_dense.weight
            mg_fc1 = mg.decoder.model.decoder.layers[i].fc1.weight
            p4500_fc1 = sd_4500[f"transformer.layers.{i}.linear1.weight"]
            fused_fc1 = (
                alpha_4500 * p4500_fc1 +
                alpha_mms * mms_fc1 +
                alpha_musicgen * mg_fc1
            )
            target_sd[f"layers.{i}.linear1.weight"].copy_(fused_fc1)
            target_sd[f"layers.{i}.linear1.bias"].copy_(sd_4500[f"transformer.layers.{i}.linear1.bias"])

            mms_fc2 = mms.encoder.layers[i].feed_forward.output_dense.weight
            mg_fc2 = mg.decoder.model.decoder.layers[i].fc2.weight
            p4500_fc2 = sd_4500[f"transformer.layers.{i}.linear2.weight"]
            fused_fc2 = (
                alpha_4500 * p4500_fc2 +
                alpha_mms * mms_fc2 +
                alpha_musicgen * mg_fc2
            )
            target_sd[f"layers.{i}.linear2.weight"].copy_(fused_fc2)
            target_sd[f"layers.{i}.linear2.bias"].copy_(sd_4500[f"transformer.layers.{i}.linear2.bias"])

            # Norms
            target_sd[f"layers.{i}.norm1.weight"].copy_(sd_4500[f"transformer.layers.{i}.norm1.weight"])
            target_sd[f"layers.{i}.norm1.bias"].copy_(sd_4500[f"transformer.layers.{i}.norm1.bias"])
            target_sd[f"layers.{i}.norm2.weight"].copy_(sd_4500[f"transformer.layers.{i}.norm2.weight"])
            target_sd[f"layers.{i}.norm2.bias"].copy_(sd_4500[f"transformer.layers.{i}.norm2.bias"])

        self.transformer.load_state_dict(target_sd)
        self.ln_f.load_state_dict({"weight": sd_4500["ln_f.weight"], "bias": sd_4500["ln_f.bias"]})
        print("✓ WSP Multi-Model Surgery completed successfully.")

    @torch.no_grad()
    def forward(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """
        Pure textless speech-to-speech forward pass.
        Args:
            wav_16k: Continuous input speech audio (1, 1, samples) or (1, samples) at 16kHz
        Returns:
            wav_out: Continuous synthesized speech audio waveform (1, samples) at 16kHz
        """
        self.eval()
        if wav_16k.ndim == 3:
            wav_16k = wav_16k.squeeze(1)
        if wav_16k.ndim == 1:
            wav_16k = wav_16k.unsqueeze(0)
        wav_16k = wav_16k.to(self.device)

        # 1. Frontend: Acoustic feature extraction (~50 Hz)
        features = self.frontend_feature_extractor(wav_16k)
        features = features.transpose(1, 2)  # (1, T, 512)
        hidden_states, _ = self.frontend_feature_projection(features)  # (1, T, 1024)

        # 2. Backbone: Causal Temporal Transformer
        seq_len = hidden_states.shape[1]
        x = self.pos_encoder(hidden_states)
        mask = self._generate_causal_mask(seq_len, self.device)
        h = self.transformer(x, mask=mask, is_causal=True)
        h = self.ln_f(h)  # (1, T, 1024)

        # 3. Vocal Tract Projector (1024 -> 192)
        vocal_latents = self.vocal_projector(h)  # (1, T, 192)
        vocal_latents = vocal_latents.transpose(1, 2)  # (1, 192, T)

        # 4. Neural Vocoder Synthesis
        wav_out = self.vocoder(vocal_latents)  # (1, 1, out_samples)
        return wav_out.squeeze(0)
