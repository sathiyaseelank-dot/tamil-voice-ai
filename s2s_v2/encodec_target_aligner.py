#!/usr/bin/env python3
"""
Encodec Target Extraction, Continuous Timestamp Aligner, and Codebook Ablation Engine.
Guarantees verified codebook slicing:
- Preserves scale and chunk metadata for faithful reconstruction.
- Keeps training token extraction separate from waveform reconstruction.
- True CB0-only decoding slices token tensor to shape (batch, 1, seq_len).
- Verifies that Encodec RVQ completely removes residual quantized vectors rather
  than looking up codevector index 0.
"""

import os
import torch
import torch.nn as nn
import torchaudio
from typing import Optional, Tuple, List
from encodec import EncodecModel
from encodec.utils import convert_audio

from model.s2s_transformer import TimestampAligner


class EncodecTargetAligner(nn.Module):
    """
    Manages continuous timestamp alignment and verified Encodec token encoding/decoding.
    """
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Continuous Timestamp Aligner
        self.aligner = TimestampAligner(d_in=1024, d_out=1024).to(self.device)

        # 2. Frozen Encodec 24kHz Model
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.set_target_bandwidth(3.0)  # 3.0 kbps = 4 codebooks @ 75 Hz
        self.codec.eval()
        for p in self.codec.parameters():
            p.requires_grad = False
        self.codec.to(self.device)

    @torch.no_grad()
    def extract_targets(self, wav: torch.Tensor, sample_rate: int = 24000) -> torch.Tensor:
        """
        Pure token extraction for training targets (independent of synthesis metadata).
        Args:
            wav: (batch, samples) or (batch, 1, samples)
            sample_rate: sampling rate of input audio
        Returns:
            tokens: (batch, 4, T_codec) in range [0, 1023]
        """
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        wav = wav.to(self.device)

        if sample_rate != 24000:
            wav = convert_audio(wav, sample_rate, 24000, self.codec.channels)

        # Normalize peak
        max_val = wav.abs().max()
        if max_val > 0:
            wav = (wav / max_val) * 0.95

        encoded_frames = self.codec.encode(wav)
        codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)  # (batch, 4, T)
        return codes[:, :4, :].long()

    @torch.no_grad()
    def encode_with_metadata(
        self,
        wav: torch.Tensor,
        sample_rate: int = 24000
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, Optional[torch.Tensor]]]]:
        """
        Extracts tokens AND preserves full Encodec frame/chunk metadata (including scale).
        Returns:
            tokens: (batch, 4, T_codec)
            encoded_frames: List of [(codes_chunk, scale_chunk)]
        """
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        wav = wav.to(self.device)

        if sample_rate != 24000:
            wav = convert_audio(wav, sample_rate, 24000, self.codec.channels)

        max_val = wav.abs().max()
        if max_val > 0:
            wav = (wav / max_val) * 0.95

        encoded_frames = self.codec.encode(wav)
        codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)
        return codes[:, :4, :].long(), encoded_frames

    def align_features(self, h_mms: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Applies mathematical continuous coordinate mapping:
            mu(n) = (2/3) * n - 0.29114583
        """
        return self.aligner(h_mms, target_len)

    @torch.no_grad()
    def decode_tokens(
        self,
        tokens: torch.Tensor,
        num_codebooks: int = 4,
        scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Verified Encodec decoding supporting genuine RVQ depth truncation and optional scale.
        Args:
            tokens: (batch, K, seq_len)
            num_codebooks: number of codebooks to genuinely decode (1, 2, 3, or 4)
            scale: optional scale tensor from encoder metadata
        Returns:
            wav: (batch, 1, time_samples) at 24 kHz
        """
        tokens = tokens.to(self.device)
        clean_tokens = torch.clamp(tokens, 0, 1023)

        # Truncate genuinely to desired depth
        sliced_tokens = clean_tokens[:, :num_codebooks, :]

        # When sliced_tokens has shape (B, 1, T), Encodec's ResidualVectorQuantization
        # executes enumerate(q_indices) for exactly 1 iteration (layer 0).
        # Residual codevectors from layers 1..3 are genuinely omitted.
        frames = [(sliced_tokens, scale)]
        waveform = self.codec.decode(frames)
        return waveform

    @torch.no_grad()
    def reconstruct_from_frames(
        self,
        encoded_frames: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
        num_codebooks: int = 4
    ) -> torch.Tensor:
        """
        Reconstructs waveform preserving original scale and chunk structure across frames.
        """
        truncated_frames = []
        for codes, scale in encoded_frames:
            codes_c = torch.clamp(codes[:, :num_codebooks, :], 0, 1023).to(self.device)
            scale_c = scale.to(self.device) if scale is not None else None
            truncated_frames.append((codes_c, scale_c))
        return self.codec.decode(truncated_frames)

    @torch.no_grad()
    def verify_vector_ablation(self, tokens_4cb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Verifies the mathematical difference between:
        1. True CB0 vector ablation: slicing to (B, 1, T) so residual codevectors are never looked up.
        2. Pseudo-ablation: leaving shape as (B, 4, T) with rows 1..3 set to token ID 0.
        Returns:
            wav_true: (batch, 1, samples)
            wav_pseudo: (batch, 1, samples)
            l2_difference: L2 norm of the difference
        """
        tokens_4cb = tokens_4cb.to(self.device)
        B, K, T = tokens_4cb.shape
        assert K >= 4, "Requires 4-codebook input tokens for ablation verification"

        # 1. True CB0 ablation (n_q = 1)
        wav_true = self.decode_tokens(tokens_4cb, num_codebooks=1)

        # 2. Pseudo-ablation (n_q = 4 with rows 1..3 set to 0)
        tokens_pseudo = tokens_4cb.clone()
        tokens_pseudo[:, 1:, :] = 0
        wav_pseudo = self.decode_tokens(tokens_pseudo, num_codebooks=4)

        l2_diff = torch.norm(wav_true - wav_pseudo).item()
        return wav_true, wav_pseudo, l2_diff
