"""
Textless Autoregressive Speech Language Model (V2) for Tamil.
Operates on discrete multi-codebook RVQ audio tokens without text conditioning.
Implements temporal causal attention and teacher-forced / sequential depth prediction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Union

from config import ModelConfig, AudioConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2500):
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
    Predicts K residual codebooks hierarchically across depth.
    Codebook 0 is predicted from temporal hidden state h.
    Codebook k (k >= 1) conditions on h plus residual embeddings of codebook k-1.

    During training: teacher-forced using target tokens.
    During generation: sequentially samples codebook k-1 and conditions on it.
    """
    def __init__(self, d_model: int, num_codebooks: int, vocab_size: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

        # Linear projection heads for each codebook
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(num_codebooks)
        ])

        # Cross-codebook depth conditioning embeddings (k-1 -> k)
        self.depth_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for _ in range(num_codebooks - 1)
        ])

    def forward(self, h: torch.Tensor, target_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for training or greedy evaluation.
        Args:
            h: (batch, seq_len, d_model) from temporal transformer
            target_tokens: optional (batch, num_codebooks, seq_len) for teacher forcing
        Returns:
            logits: (batch, num_codebooks, seq_len, vocab_size)
        """
        batch_size, seq_len, _ = h.shape
        logits_list = []

        # Codebook 0
        h_current = h
        logits_0 = self.heads[0](h_current)
        logits_list.append(logits_0.unsqueeze(1))

        # Codebooks 1 .. K-1
        for k in range(1, self.num_codebooks):
            if target_tokens is not None:
                # Teacher forcing: condition on ground truth preceding codebook token
                prev_token = target_tokens[:, k - 1, :]
            else:
                # Greedy fallback
                prev_token = logits_list[-1].squeeze(1).argmax(dim=-1)

            depth_emb = self.depth_embeddings[k - 1](prev_token)
            h_current = h_current + depth_emb
            logits_k = self.heads[k](h_current)
            logits_list.append(logits_k.unsqueeze(1))

        return torch.cat(logits_list, dim=1)  # (batch, num_codebooks, seq_len, vocab_size)

    def sample_step(
        self,
        h_t: torch.Tensor,
        temperature: Union[float, List[float]] = 0.8,
        top_k: Union[int, List[int]] = 50,
        top_p: Union[float, List[float]] = 0.9,
        repetition_penalty: float = 1.0,
        recent_tokens_by_cb: Optional[List[List[int]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequentially predicts, samples, and conditions across codebooks for a single time step.
        Supports both scalar sampling parameters and per-codebook hierarchical lists.
        Guarantees zero discrepancy between depth conditioning and sampled tokens.

        Args:
            h_t: (batch=1, d_model) hidden state at current time step
            temperature: sampling temperature (scalar or list per codebook)
            top_k: top-k filtering parameter (scalar or list per codebook)
            top_p: nucleus filtering parameter (scalar or list per codebook)
            repetition_penalty: penalty for repeated tokens
            recent_tokens_by_cb: list of recent token ids for each codebook [cb0_list, cb1_list, ...]
        Returns:
            sampled_frame: (1, num_codebooks, 1) tensor of sampled token IDs
            step_logits: (1, num_codebooks, vocab_size) logits for each codebook
        """
        device = h_t.device
        h_current = h_t.unsqueeze(1)  # (1, 1, d_model)
        sampled_tokens = []
        logits_list = []

        for k in range(self.num_codebooks):
            # Compute logits for codebook k
            logits_k = self.heads[k](h_current).squeeze(1)  # (1, vocab_size)
            logits_list.append(logits_k.unsqueeze(1))

            t_k = temperature[k] if isinstance(temperature, (list, tuple)) else temperature
            k_k = top_k[k] if isinstance(top_k, (list, tuple)) else top_k
            p_k = top_p[k] if isinstance(top_p, (list, tuple)) else top_p

            # Apply repetition penalty if specified
            logits_proc = logits_k.clone()
            if repetition_penalty > 1.0 and recent_tokens_by_cb is not None:
                recent = recent_tokens_by_cb[k]
                for prev_tok in set(recent):
                    if prev_tok < self.vocab_size:
                        if logits_proc[0, prev_tok] > 0:
                            logits_proc[0, prev_tok] /= repetition_penalty
                        else:
                            logits_proc[0, prev_tok] *= repetition_penalty

            # Greedy or temperature sampling
            if t_k <= 1e-4:
                token = logits_proc.argmax(dim=-1, keepdim=True)
            else:
                logits_proc = logits_proc / max(t_k, 1e-5)

                # Top-K filtering
                if k_k > 0:
                    val, _ = torch.topk(logits_proc, min(k_k, logits_proc.size(-1)))
                    logits_proc[logits_proc < val[:, [-1]]] = float('-inf')

                # Top-P (Nucleus) filtering
                if p_k < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits_proc, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > p_k
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits_proc[indices_to_remove] = float('-inf')

                probs = F.softmax(logits_proc, dim=-1)
                token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            sampled_tokens.append(token)

            # Sequential conditioning: advance hidden state using sampled token
            if k < self.num_codebooks - 1:
                depth_emb = self.depth_embeddings[k](token)  # (1, 1, d_model)
                h_current = h_current + depth_emb

        sampled_frame = torch.cat(sampled_tokens, dim=1).unsqueeze(-1)  # (1, num_codebooks, 1)
        step_logits = torch.cat(logits_list, dim=1)  # (1, num_codebooks, vocab_size)
        return sampled_frame, step_logits


class TextlessTamilSpeechLM(nn.Module):
    """
    Single-stream Textless Autoregressive Audio Transformer.
    Models temporal dependencies via causal attention and codebook depth dependencies
    via sequential teacher-forcing (training) and sequential sampling (inference).
    """
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        audio_cfg = self.config.audio

        self.num_codebooks = audio_cfg.num_codebooks
        self.vocab_size = audio_cfg.vocab_size
        self.pad_token_id = audio_cfg.pad_token_id
        self.bos_token_id = audio_cfg.bos_token_id
        d_model = self.config.d_model

        # Token embeddings per codebook (summed along codebook dimension)
        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size, d_model) for _ in range(self.num_codebooks)
        ])

        self.pos_encoder = PositionalEncoding(d_model, max_len=self.config.max_seq_len + 500)

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

        # Multi-codebook depth head
        self.head = DepthAudioHead(d_model, self.num_codebooks, self.vocab_size)

    def _embed_frames(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embeds multi-codebook frames by summing embeddings across codebooks.
        Args:
            tokens: (batch, num_codebooks, seq_len)
        Returns:
            embedded: (batch, seq_len, d_model)
        """
        batch_size, n_q, seq_len = tokens.shape
        embedded = 0
        for k in range(n_q):
            embedded = embedded + self.codebook_embeddings[k](tokens[:, k, :])
        return embedded

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Strict upper triangular causal mask."""
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(
        self,
        input_tokens: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, float]]]:
        """
        Forward pass.
        Args:
            input_tokens: (batch, num_codebooks, seq_len)
            target_tokens: optional (batch, num_codebooks, seq_len)
        Returns:
            logits: (batch, num_codebooks, seq_len, vocab_size)
            loss: scalar cross-entropy loss (or None if no targets)
            metrics: dict of per-codebook loss, accuracies, entropy (or None)
        """
        seq_len = input_tokens.shape[-1]
        device = input_tokens.device

        # Embed input frames
        x = self._embed_frames(input_tokens)
        x = self.pos_encoder(x)

        # Temporal causal attention
        mask = self._generate_causal_mask(seq_len, device)
        h = self.transformer(x, mask=mask, is_causal=True)
        h = self.ln_f(h)

        # Depth prediction
        logits = self.head(h, target_tokens=target_tokens)

        loss = None
        metrics = None

        if target_tokens is not None:
            # Flatten codebooks and sequence length for loss
            flat_logits = logits.permute(0, 2, 1, 3).reshape(-1, self.vocab_size)
            flat_targets = target_tokens.permute(0, 2, 1).reshape(-1)

            loss = F.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=self.pad_token_id
            )

            # Detailed metrics calculation
            with torch.no_grad():
                metrics = {}
                # Valid mask (ignore actual padding only, keep legitimate speech and silence)
                valid_mask = (target_tokens != self.pad_token_id)  # (batch, num_codebooks, seq_len)
                total_valid = valid_mask.sum().item()

                if total_valid > 0:
                    pred_tokens = logits.argmax(dim=-1)  # (batch, num_codebooks, seq_len)
                    correct_all = (pred_tokens == target_tokens) & valid_mask
                    metrics["speech_accuracy"] = correct_all.sum().item() / total_valid

                    for k in range(self.num_codebooks):
                        valid_k = valid_mask[:, k, :]
                        total_valid_k = valid_k.sum().item()
                        if total_valid_k > 0:
                            # Per-codebook loss
                            cb_logits = logits[:, k, :, :].reshape(-1, self.vocab_size)
                            cb_targets = target_tokens[:, k, :].reshape(-1)
                            cb_loss = F.cross_entropy(cb_logits, cb_targets, ignore_index=self.pad_token_id)
                            metrics[f"loss_cb{k}"] = cb_loss.item()

                            # Per-codebook accuracy
                            cb_correct = ((pred_tokens[:, k, :] == target_tokens[:, k, :]) & valid_k).sum().item()
                            metrics[f"acc_cb{k}"] = cb_correct / total_valid_k
                        else:
                            metrics[f"loss_cb{k}"] = 0.0
                            metrics[f"acc_cb{k}"] = 0.0

                    # Prediction entropy across vocabulary
                    probs = F.softmax(flat_logits[flat_targets != self.pad_token_id], dim=-1)
                    entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1).mean()
                    metrics["entropy"] = entropy.item()
                else:
                    metrics["speech_accuracy"] = 0.0
                    metrics["entropy"] = 0.0

        return logits, loss, metrics

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: Optional[torch.Tensor] = None,
        max_new_tokens: int = 150,
        temperature: Union[float, List[float]] = 0.8,
        top_k: Union[int, List[int]] = 50,
        top_p: Union[float, List[float]] = 0.9,
        repetition_penalty: float = 1.15
    ) -> torch.Tensor:
        """
        Autoregressively generates next audio frames using sequential depth sampling.
        Matches training token conventions exactly.

        Args:
            prompt_tokens: optional (1, num_codebooks, prompt_len)
            max_new_tokens: number of new frames to generate (75 frames = 1.0 second)
            temperature: sampling temperature
            top_k: top-k filtering parameter
            top_p: nucleus filtering parameter
            repetition_penalty: penalty for recently repeated tokens
        Returns:
            generated_tokens: (1, num_codebooks, max_new_tokens)
        """
        self.eval()
        device = next(self.parameters()).device

        bos = torch.full(
            (1, self.num_codebooks, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=device
        )

        if prompt_tokens is None:
            # Unconditional generation: starts with BOS
            curr_tokens = bos
            prompt_len = 0
        else:
            # Conditional generation: prepend BOS to prompt
            prompt_tokens = prompt_tokens.to(device)
            curr_tokens = torch.cat([bos, prompt_tokens], dim=-1)
            prompt_len = prompt_tokens.shape[-1]

        # Recent tokens history per codebook for repetition penalty
        recent_window = 25

        for step in range(max_new_tokens):
            seq_len = curr_tokens.shape[-1]
            if seq_len >= self.config.max_seq_len:
                break

            # Forward pass through temporal transformer
            x = self._embed_frames(curr_tokens)
            x = self.pos_encoder(x)
            mask = self._generate_causal_mask(seq_len, device)
            h = self.transformer(x, mask=mask, is_causal=True)
            h = self.ln_f(h)

            # Latest temporal hidden state
            h_latest = h[:, -1, :].squeeze(1)  # (1, d_model)

            # Gather recent tokens per codebook
            recent_tokens_by_cb = []
            for k in range(self.num_codebooks):
                history = curr_tokens[0, k, -recent_window:].tolist()
                recent_tokens_by_cb.append(history)

            # Sequentially predict, sample, and depth-condition the 4 codebooks
            next_frame, _ = self.head.sample_step(
                h_t=h_latest,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                recent_tokens_by_cb=recent_tokens_by_cb
            )  # (1, num_codebooks, 1)

            # Append completed frame to sequence
            curr_tokens = torch.cat([curr_tokens, next_frame], dim=-1)

        # Return only newly generated frames (excluding BOS and prompt)
        return curr_tokens[:, :, (1 + prompt_len):]
