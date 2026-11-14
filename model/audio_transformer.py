"""
Textless Full-Duplex Speech-to-Speech Transformer for Conversational Tamil.
Operates directly on discrete multi-codebook audio tokens without intermediate text.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from config import ModelConfig, AudioConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class DepthAudioHead(nn.Module):
    """
    Predicts K residual codebooks hierarchically or autoregressively across depth.
    Codebook 0 (primary acoustic/semantic token) is predicted first, and codebooks 1..K-1
    condition on the temporal hidden state plus embeddings of preceding codebooks.
    """
    def __init__(self, d_model: int, num_codebooks: int, vocab_size: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        
        # Heads for each codebook
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(num_codebooks)
        ])
        
        # Cross-codebook residual conditioning embeddings
        self.depth_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for _ in range(num_codebooks - 1)
        ])

    def forward(self, h: torch.Tensor, target_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h: (batch, seq_len, d_model) from temporal transformer
            target_tokens: optional (batch, num_codebooks, seq_len) during teacher forcing
        Returns:
            logits: (batch, num_codebooks, seq_len, vocab_size)
        """
        batch_size, seq_len, _ = h.shape
        logits_list = []

        # Predict Codebook 0
        h_current = h
        logits_0 = self.heads[0](h_current)
        logits_list.append(logits_0.unsqueeze(1))

        for k in range(1, self.num_codebooks):
            if target_tokens is not None:
                # Teacher forcing: condition on ground truth preceding codebook token
                prev_token = target_tokens[:, k - 1, :]
            else:
                # Inference: condition on argmax/sampled token of preceding codebook
                prev_token = logits_list[-1].squeeze(1).argmax(dim=-1)

            depth_emb = self.depth_embeddings[k - 1](prev_token)
            h_current = h_current + depth_emb
            logits_k = self.heads[k](h_current)
            logits_list.append(logits_k.unsqueeze(1))

        # Shape: (batch, num_codebooks, seq_len, vocab_size)
        return torch.cat(logits_list, dim=1)


class TextlessTamilSpeechLM(nn.Module):
    """
    Dual-stream Audio-to-Audio Language Model for Tamil.
    Processes:
      1. User audio stream tokens (listening)
      2. Assistant audio stream tokens (speaking)
    Enables full-duplex conversational interaction without text.
    """
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        audio_cfg = self.config.audio
        
        self.num_codebooks = audio_cfg.num_codebooks
        self.vocab_size = audio_cfg.vocab_size
        d_model = self.config.d_model

        # Token embeddings for user and assistant streams (separate tables per codebook)
        self.user_embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size, d_model) for _ in range(self.num_codebooks)
        ])
        self.assistant_embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size, d_model) for _ in range(self.num_codebooks)
        ])

        # Channel role embeddings (0 = user channel, 1 = assistant channel)
        self.channel_emb = nn.Embedding(2, d_model)
        
        # Projection layer to merge user & assistant streams into single sequence representation
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.config.max_seq_len)

        # Causal Temporal Transformer Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_ff,
            dropout=self.config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=self.config.n_layers)
        self.ln_f = nn.LayerNorm(d_model)

        # Output prediction head for multi-codebook audio tokens
        self.head = DepthAudioHead(d_model, self.num_codebooks, self.vocab_size)

    def _embed_stream(self, tokens: torch.Tensor, embedding_list: nn.ModuleList) -> torch.Tensor:
        """
        Embeds multi-codebook tokens of shape (batch, num_codebooks, seq_len)
        by summing embeddings across codebooks.
        """
        batch_size, n_q, seq_len = tokens.shape
        embedded = 0
        for k in range(n_q):
            embedded = embedded + embedding_list[k](tokens[:, k, :])
        return embedded  # (batch, seq_len, d_model)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Standard upper triangular causal mask."""
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(
        self,
        user_tokens: torch.Tensor,
        assistant_tokens: torch.Tensor,
        target_assistant_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            user_tokens: (batch, num_codebooks, seq_len)
            assistant_tokens: (batch, num_codebooks, seq_len) (input context for assistant)
            target_assistant_tokens: (batch, num_codebooks, seq_len) (targets for loss computation)
        Returns:
            logits: (batch, num_codebooks, seq_len, vocab_size)
            loss: scalar loss (if targets provided)
        """
        seq_len = user_tokens.shape[-1]
        device = user_tokens.device

        # Embed user and assistant streams
        u_emb = self._embed_stream(user_tokens, self.user_embeddings) + self.channel_emb(torch.tensor(0, device=device))
        a_emb = self._embed_stream(assistant_tokens, self.assistant_embeddings) + self.channel_emb(torch.tensor(1, device=device))

        # Concatenate and fuse channels
        fused = self.fusion_proj(torch.cat([u_emb, a_emb], dim=-1))
        x = self.pos_encoder(fused)

        # Causal attention mask
        mask = self._generate_causal_mask(seq_len, device)
        h = self.transformer(x, mask=mask, is_causal=True)
        h = self.ln_f(h)

        # Predict assistant audio token logits
        logits = self.head(h, target_tokens=target_assistant_tokens)

        loss = None
        if target_assistant_tokens is not None:
            # Flatten codebooks and time steps for cross-entropy loss
            # logits: (batch, num_codebooks, seq_len, vocab_size)
            # targets: (batch, num_codebooks, seq_len)
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                target_assistant_tokens.reshape(-1),
                ignore_index=self.config.audio.pad_token_id
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        user_tokens: torch.Tensor,
        max_new_tokens: int = 150,
        temperature: float = 0.35,
        top_k: int = 40,
        top_p: float = 0.85,
        repetition_penalty: float = 1.25,
        assistant_prefix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Autoregressively generates assistant audio tokens conditioned on user audio tokens.
        Applies top-p nucleus sampling and repetition penalty to prevent monotone humming/drones.
        """
        self.eval()
        device = user_tokens.device
        cfg = self.config.audio

        user_len = user_tokens.shape[-1]
        
        # Start assistant tokens with silence during user speech
        if assistant_prefix is None:
            curr_assistant = torch.full(
                (1, self.num_codebooks, user_len),
                cfg.silence_token_id,
                dtype=torch.long,
                device=device
            )
        else:
            curr_assistant = assistant_prefix.to(device)

        # Extended user stream with silence padding for future frames
        future_user = torch.full(
            (1, self.num_codebooks, max_new_tokens),
            cfg.silence_token_id,
            dtype=torch.long,
            device=device
        )
        full_user = torch.cat([user_tokens, future_user], dim=-1)

        for step in range(max_new_tokens):
            curr_len = curr_assistant.shape[-1]
            if curr_len >= self.config.max_seq_len:
                break

            # Current user slice matching assistant length
            u_slice = full_user[:, :, :curr_len]

            # Forward pass
            logits, _ = self.forward(u_slice, curr_assistant)
            
            # Extract next token logits for the latest time step: (1, num_codebooks, vocab_size)
            next_step_logits = logits[:, :, -1, :]

            # Sample each codebook with repetition penalty & top-p
            sampled_tokens = []
            for k in range(self.num_codebooks):
                logits_k = next_step_logits[:, k, :].clone()

                # Anti-humming: apply repetition penalty on recent frames (last 15 frames)
                if repetition_penalty > 1.0 and curr_assistant.shape[-1] > user_len:
                    recent = curr_assistant[0, k, max(user_len, curr_assistant.shape[-1] - 15):]
                    for prev_tok in set(recent.tolist()):
                        if logits_k[0, prev_tok] > 0:
                            logits_k[0, prev_tok] /= repetition_penalty
                        else:
                            logits_k[0, prev_tok] *= repetition_penalty

                # Apply temperature
                logits_k = logits_k / max(temperature, 1e-5)

                # Top-K filtering
                if top_k > 0:
                    val, _ = torch.topk(logits_k, min(top_k, logits_k.size(-1)))
                    logits_k[logits_k < val[:, [-1]]] = float('-inf')

                # Top-P (Nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits_k, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits_k[indices_to_remove] = float('-inf')

                probs = F.softmax(logits_k, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                sampled_tokens.append(token)

            # Combine codebooks: (1, num_codebooks, 1)
            new_frame = torch.stack(sampled_tokens, dim=1)
            curr_assistant = torch.cat([curr_assistant, new_frame], dim=-1)

        # Return only the newly generated response frames
        return curr_assistant[:, :, user_len:]
