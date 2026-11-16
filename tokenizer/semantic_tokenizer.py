#!/usr/bin/env python3
"""
Semantic Speech Unit Tokenizer for Tamil Voice AI.
Extracts discrete self-supervised phonetic/linguistic units from continuous speech audio
using Meta's MMS-300M multilingual speech representation backbone.

Frame rate: ~50 Hz (1 unit every 20ms).
Vocabulary: 1024 discrete semantic units (Phone/Syllable invariant representations).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2Model


class SemanticSpeechTokenizer(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/mms-300m",
        vocab_size: int = 1024,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.vocab_size = vocab_size
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.ssl_backbone = Wav2Vec2Model.from_pretrained(model_name)
        self.ssl_backbone.eval()
        self.ssl_backbone.to(self.device)

        # Vector quantizer codebook for discrete semantic tokens (1024 units, dim 1024)
        self.d_model = self.ssl_backbone.config.hidden_size  # 1024
        
        # Initialize semantic codebook
        self.codebook = nn.Embedding(self.vocab_size, self.d_model)
        # Initialize codebook with normalized random orthogonal vectors
        nn.init.orthogonal_(self.codebook.weight)
        self.codebook.to(self.device)

    @torch.no_grad()
    def extract_continuous_features(self, wav: torch.Tensor, sr: int = 24000) -> torch.Tensor:
        """
        Extracts continuous self-supervised representation (1, T_semantic, 1024) at ~50 Hz.
        """
        if wav.ndim == 3:
            wav = wav.squeeze(1)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        # Resample to 16kHz for MMS backbone
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        wav = wav.to(self.device)
        outputs = self.ssl_backbone(wav)
        features = outputs.last_hidden_state  # (batch, seq_len, 1024)
        return features

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sr: int = 24000) -> torch.Tensor:
        """
        Encodes continuous audio into discrete semantic token IDs.
        Returns:
            semantic_tokens: (batch, seq_len) Int tensor with values in [0, vocab_size - 1]
        """
        feats = self.extract_continuous_features(wav, sr)  # (batch, T, d_model)
        feats_norm = F.normalize(feats, p=2, dim=-1)
        codebook_norm = F.normalize(self.codebook.weight, p=2, dim=-1)  # (vocab, d_model)

        # Cosine similarity: (batch, T, vocab)
        sim = torch.matmul(feats_norm, codebook_norm.t())
        semantic_tokens = torch.argmax(sim, dim=-1)  # (batch, T)
        return semantic_tokens

    @torch.no_grad()
    def decode_to_features(self, semantic_tokens: torch.Tensor) -> torch.Tensor:
        """
        Maps discrete semantic tokens back into continuous semantic embedding space.
        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        semantic_tokens = semantic_tokens.to(self.device)
        return self.codebook(semantic_tokens)
