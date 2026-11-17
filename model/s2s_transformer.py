"""
S2S V2 Model Architecture: Genuine Textless Speech-to-Speech Transformer.
Converts continuous SSL representations (Meta MMS-300M, 50 Hz, 16 kHz)
into discrete acoustic multi-codebook representations (Encodec RVQ, 75 Hz, 24 kHz)
without any text, phonemes, transcripts, or hidden text encoders.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Continuous Sinusoidal Positional Encoding for 75 Hz speech sequences.
    Provides scale-invariant monotonic position information over arbitrary sequence lengths.
    """
    def __init__(self, d_model: int = 1024, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned 1D Absolute Positional Embeddings for ablation benchmarking.
    """
    def __init__(self, d_model: int = 1024, max_len: int = 5000):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        return x + self.pos_emb(positions).unsqueeze(0)


class TimestampAligner(nn.Module):
    """
    Continuous timestamp-based alignment between MMS (16 kHz, 50 Hz)
    and Encodec (24 kHz, 75 Hz).
    
    Temporal Physics:
    - MMS-300M Conv Feature Extractor:
        Cumulative Stride: 320 samples at 16 kHz = 20.0 ms per frame (50 Hz).
        Receptive Field: 400 samples at 16 kHz = 25.0 ms.
        Frame m centre: t_mms(m) = (320 * m + 199.5) / 16000.0 s.
    - Encodec 24kHz:
        Cumulative Stride: 320 samples at 24 kHz = 13.333 ms per frame (75 Hz).
        Frame n centre: t_codec(n) = (320 * n + 159.5) / 24000.0 s.
    - Continuous coordinate of Encodec frame n along MMS frame timeline:
        mu(n) = (t_codec(n) - t_mms(0)) / 0.020
              = (2/3) * n - 0.29114583
    """
    def __init__(self, d_in: int = 1024, d_out: int = 1024):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.smooth_conv = nn.Conv1d(d_out, d_out, kernel_size=3, padding=1)

    def forward(self, h_mms: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Args:
            h_mms: (batch, T_mms, d_in) from MMS-300M
            target_len: target number of 75 Hz Encodec frames
        Returns:
            h_aligned: (batch, target_len, d_out) aligned to Encodec frame centres
        """
        B, T_mms, D = h_mms.shape
        device = h_mms.device

        # Continuous target coordinates
        n_idx = torch.arange(target_len, dtype=torch.float32, device=device)
        mu = (2.0 / 3.0) * n_idx - 0.29114583
        mu = torch.clamp(mu, 0.0, float(T_mms - 1))

        mu_floor = torch.floor(mu).long()
        mu_ceil = torch.clamp(mu_floor + 1, max=T_mms - 1)
        w = (mu - mu_floor.float()).unsqueeze(0).unsqueeze(-1)  # (1, target_len, 1)

        # Gather continuous representations via linear interpolation
        h_floor = h_mms[:, mu_floor, :]
        h_ceil = h_mms[:, mu_ceil, :]
        h_interp = (1.0 - w) * h_floor + w * h_ceil

        # Apply projection, norm, and temporal smoothing conv
        h_proj = self.norm(self.proj(h_interp))
        h_conv = self.smooth_conv(h_proj.transpose(1, 2)).transpose(1, 2)
        return h_proj + h_conv


class DepthAudioHead(nn.Module):
    """
    Hierarchical Multi-Codebook Prediction Head for Encodec RVQ codebooks (e.g. 4 codebooks).
    Predicts CB0 unconditionally from the latent representations.
    Predicts CB_k conditioned on embeddings of preceding codebooks (depth recurrence).
    """
    def __init__(self, d_model: int = 1024, num_codebooks: int = 4, vocab_size: int = 1029):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(num_codebooks)
        ])
        self.depth_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for _ in range(num_codebooks - 1)
        ])

    def forward(self, h: torch.Tensor, target_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h: (batch, seq_len, d_model)
            target_tokens: optional (batch, num_codebooks, seq_len) for teacher-forced conditioning
        Returns:
            logits: (batch, num_codebooks, seq_len, vocab_size)
        """
        B, T, D = h.shape
        logits_list = []

        curr_h = h
        logits_0 = self.heads[0](curr_h)
        logits_list.append(logits_0)

        for k in range(1, self.num_codebooks):
            if target_tokens is not None:
                prev_tokens = target_tokens[:, k - 1, :]
            else:
                prev_tokens = torch.argmax(logits_list[-1], dim=-1)

            # Clamp valid range
            prev_tokens = torch.clamp(prev_tokens, 0, self.vocab_size - 1)
            depth_emb = self.depth_embeddings[k - 1](prev_tokens)
            curr_h = curr_h + depth_emb
            logits_k = self.heads[k](curr_h)
            logits_list.append(logits_k)

        return torch.stack(logits_list, dim=1)


class S2STokenPredictor(nn.Module):
    """
    Textless Speech-to-Speech Acoustic Token Predictor.
    Maps continuous input speech representations (MMS-300M)
    directly to discrete Encodec multi-codebook tokens.
    """
    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 6,
        d_ff: int = 4096,
        num_codebooks: int = 4,
        vocab_size: int = 1029,
        dropout: float = 0.1,
        pos_enc_type: str = "sinusoidal"  # "sinusoidal", "learned", "conv", "none"
    ):
        super().__init__()
        self.d_model = d_model
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.pos_enc_type = pos_enc_type

        # Continuous timestamp aligner
        self.aligner = TimestampAligner(d_in=1024, d_out=d_model)

        # Positional Encoding Mechanism
        if pos_enc_type == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model)
        elif pos_enc_type == "learned":
            self.pos_enc = LearnedPositionalEncoding(d_model=d_model)
        elif pos_enc_type in ["conv", "none"]:
            self.pos_enc = nn.Identity()
        else:
            raise ValueError(f"Unknown pos_enc_type: {pos_enc_type}")

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)

        # Multi-codebook depth head
        self.head = DepthAudioHead(d_model=d_model, num_codebooks=num_codebooks, vocab_size=vocab_size)

    def forward(
        self,
        h_mms: torch.Tensor,
        target_len: int,
        target_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            h_mms: (batch, T_mms, 1024) MMS feature representations
            target_len: integer Encodec sequence length
            target_tokens: optional (batch, num_codebooks, target_len)
        Returns:
            logits: (batch, num_codebooks, target_len, vocab_size)
        """
        # 1. Align MMS timeline (50 Hz) to Encodec timeline (75 Hz)
        h_aligned = self.aligner(h_mms, target_len=target_len)

        # 2. Add Positional Information
        h_pos = self.pos_enc(h_aligned)

        # 3. Deep Transformer processing
        h_trans = self.transformer(h_pos)
        h_norm = self.ln_f(h_trans)

        # 4. Hierarchical codebook logits
        logits = self.head(h_norm, target_tokens=target_tokens)
        return logits

    @torch.no_grad()
    def generate_tokens(
        self,
        h_mms: torch.Tensor,
        target_len: int,
        cb0_only: bool = False
    ) -> torch.Tensor:
        """
        Inference token generation.
        Args:
            h_mms: (batch, T_mms, 1024)
            target_len: target Encodec frames
            cb0_only: if True, returns genuinely sliced 1-codebook tensor (batch, 1, target_len)
        Returns:
            pred_tokens: (batch, K, target_len) where K=1 if cb0_only else num_codebooks
        """
        self.eval()
        logits = self.forward(h_mms, target_len=target_len, target_tokens=None)
        pred_tokens = torch.argmax(logits, dim=-1)  # (batch, num_codebooks, target_len)

        if cb0_only:
            # Genuine CB0-only codebook slicing (n_q = 1)
            # NEVER substitute token 0 into residual codebooks!
            return pred_tokens[:, :1, :]

        return pred_tokens
