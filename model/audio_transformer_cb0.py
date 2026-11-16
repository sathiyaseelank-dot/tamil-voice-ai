"""
Experiment V2-P2.5-A: Single-Codebook (CB0-Only) Speech Language Model Architecture.

Isolates acoustic language modeling to Codebook 0 (base formants, pitch contour, and energy).
Eliminates cross-codebook depth head interference, residual noise, and depth-conditioning complexity.
"""

import math
from typing import Optional, Dict, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class TextlessTamilSpeechLM_CB0(nn.Module):
    """
    Single-Stream Textless Autoregressive Audio Transformer for Codebook 0.
    Directly predicts next CB0 acoustic token using strict causal temporal attention.
    """
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        audio_cfg = self.config.audio

        self.vocab_size = audio_cfg.vocab_size  # 1024 + 5 special tokens = 1029
        self.pad_token_id = audio_cfg.pad_token_id  # 1024
        self.bos_token_id = audio_cfg.bos_token_id  # 1025
        self.d_model = self.config.d_model  # 1024

        # Single CB0 token embedding
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=self.config.max_seq_len + 500)

        # Causal Temporal Transformer Decoder Backbone (6 layers, 16 heads, d_ff=4096)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_ff,
            dropout=self.config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=self.config.n_layers)
        self.ln_f = nn.LayerNorm(self.d_model)

        # Direct projection head to CB0 vocabulary
        self.head = nn.Linear(self.d_model, self.vocab_size)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Strict upper-triangular causal mask."""
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(
        self,
        input_tokens: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, float]]]:
        """
        Forward pass for CB0 model.
        Args:
            input_tokens: (batch, seq_len) or (batch, 1, seq_len)
            target_tokens: optional (batch, seq_len) or (batch, 1, seq_len)
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar cross-entropy loss (or None if no targets)
            metrics: dict with loss, accuracy, perplexity, and entropy
        """
        if input_tokens.ndim == 3:
            input_tokens = input_tokens.squeeze(1)
        if target_tokens is not None and target_tokens.ndim == 3:
            target_tokens = target_tokens.squeeze(1)

        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device

        # 1. Embed input tokens
        x = self.embedding(input_tokens)
        x = self.pos_encoder(x)

        # 2. Causal temporal transformer
        mask = self._generate_causal_mask(seq_len, device)
        h = self.transformer(x, mask=mask, is_causal=True)
        h = self.ln_f(h)

        # 3. Output logits
        logits = self.head(h)  # (batch, seq_len, vocab_size)

        loss = None
        metrics = None

        if target_tokens is not None:
            flat_logits = logits.reshape(-1, self.vocab_size)
            flat_targets = target_tokens.reshape(-1)

            loss = F.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=self.pad_token_id
            )

            with torch.no_grad():
                valid_mask = (target_tokens != self.pad_token_id)
                total_valid = valid_mask.sum().item()

                if total_valid > 0:
                    pred_tokens = logits.argmax(dim=-1)
                    correct = (pred_tokens == target_tokens) & valid_mask
                    accuracy = correct.sum().item() / total_valid

                    # Perplexity: exp(loss)
                    ppl = math.exp(min(loss.item(), 20.0))

                    # Entropy of predicted distribution on valid frames
                    valid_logits = flat_logits[flat_targets != self.pad_token_id]
                    probs = F.softmax(valid_logits, dim=-1)
                    entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1).mean().item()

                    metrics = {
                        "cb0_loss": loss.item(),
                        "cb0_accuracy": accuracy,
                        "cb0_perplexity": ppl,
                        "cb0_entropy": entropy
                    }
                else:
                    metrics = {
                        "cb0_loss": 0.0,
                        "cb0_accuracy": 0.0,
                        "cb0_perplexity": 1.0,
                        "cb0_entropy": 0.0
                    }

        return logits, loss, metrics

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: Optional[torch.Tensor] = None,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Autoregressively generates next CB0 audio tokens.
        Args:
            prompt_tokens: optional (1, prompt_len) or (1, 1, prompt_len)
            max_new_tokens: number of frames to generate (75 frames = 1.0s)
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus filtering
            repetition_penalty: penalty for repeated tokens in recent window
        Returns:
            generated_tokens: (1, 1, max_new_tokens)
        """
        self.eval()
        device = next(self.parameters()).device

        bos = torch.tensor([[self.bos_token_id]], dtype=torch.long, device=device)

        if prompt_tokens is None:
            curr_tokens = bos
            prompt_len = 0
        else:
            if prompt_tokens.ndim == 3:
                prompt_tokens = prompt_tokens.squeeze(1)
            prompt_tokens = prompt_tokens.to(device)
            curr_tokens = torch.cat([bos, prompt_tokens], dim=-1)
            prompt_len = prompt_tokens.shape[-1]

        recent_window = 25

        for _ in range(max_new_tokens):
            seq_len = curr_tokens.shape[-1]
            if seq_len >= self.config.max_seq_len:
                break

            x = self.embedding(curr_tokens)
            x = self.pos_encoder(x)
            mask = self._generate_causal_mask(seq_len, device)
            h = self.transformer(x, mask=mask, is_causal=True)
            h = self.ln_f(h)

            logits_t = self.head(h[:, -1, :])  # (1, vocab_size)

            # Apply repetition penalty
            if repetition_penalty > 1.0:
                recent = curr_tokens[0, -recent_window:].tolist()
                for tok in set(recent):
                    if tok < self.vocab_size:
                        if logits_t[0, tok] > 0:
                            logits_t[0, tok] /= repetition_penalty
                        else:
                            logits_t[0, tok] *= repetition_penalty

            # Sampling
            if temperature <= 1e-4:
                next_token = logits_t.argmax(dim=-1, keepdim=True)
            else:
                logits_proc = logits_t / max(temperature, 1e-5)

                if top_k > 0:
                    val, _ = torch.topk(logits_proc, min(top_k, logits_proc.size(-1)))
                    logits_proc[logits_proc < val[:, [-1]]] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits_proc, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits_proc[indices_to_remove] = float('-inf')

                probs = F.softmax(logits_proc, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            curr_tokens = torch.cat([curr_tokens, next_token], dim=-1)

        # Return (1, 1, max_new_tokens)
        gen = curr_tokens[:, (1 + prompt_len):].unsqueeze(1)
        return gen

    @classmethod
    def from_pretrained_multicodebook(cls, ckpt_path: str, config: Optional[ModelConfig] = None) -> 'TextlessTamilSpeechLM_CB0':
        """
        Warm-starts CB0 model by grafting the shared transformer trunk and CB0 head
        from a trained 4-codebook checkpoint (e.g. Step-3000 or Step-4500).
        """
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        src_sd = ckpt['model_state_dict']
        cfg = config or ckpt.get('config', ModelConfig())

        cb0_model = cls(cfg)
        tgt_sd = cb0_model.state_dict()

        # Graft trunk weights
        transferred = 0
        for k in tgt_sd.keys():
            if k in src_sd:
                tgt_sd[k].copy_(src_sd[k])
                transferred += 1

        # Graft CB0 embedding from codebook_embeddings.0
        if 'codebook_embeddings.0.weight' in src_sd:
            tgt_sd['embedding.weight'].copy_(src_sd['codebook_embeddings.0.weight'])
            transferred += 1

        # Graft CB0 head from head.heads.0
        if 'head.heads.0.weight' in src_sd:
            tgt_sd['head.weight'].copy_(src_sd['head.heads.0.weight'])
            tgt_sd['head.bias'].copy_(src_sd['head.heads.0.bias'])
            transferred += 2

        cb0_model.load_state_dict(tgt_sd)
        print(f"✓ Warm-started TextlessTamilSpeechLM_CB0 from {ckpt_path} ({transferred}/{len(tgt_sd)} parameter tensors grafted).")
        return cb0_model
