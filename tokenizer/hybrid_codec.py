#!/usr/bin/env python3
"""
Hybrid Audio-Semantic Codec for Textless Tamil Voice AI.
Bridges:
1. Semantic Speech Units (MMS-300M SSL at ~50 Hz): models linguistic & phonetic structure.
2. Acoustic Codec Tokens (Encodec 24kHz at 75 Hz): models acoustic timbre, pitch, and high-fidelity waveform rendering.

Decouples linguistic phonetics from acoustic jitter, eliminating random hallucinations.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from config import AudioConfig
from tokenizer.codec import AudioCodecTokenizer
from tokenizer.semantic_tokenizer import SemanticSpeechTokenizer


class HybridAudioSemanticCodec(nn.Module):
    def __init__(
        self,
        audio_config: Optional[AudioConfig] = None,
        semantic_vocab_size: int = 1024,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.audio_config = audio_config or AudioConfig()
        
        # 1. Semantic Tokenizer (50 Hz SSL units)
        self.semantic_tokenizer = SemanticSpeechTokenizer(
            model_name="facebook/mms-300m",
            vocab_size=semantic_vocab_size,
            device=device
        )

        # 2. Acoustic Codec Tokenizer (75 Hz Encodec 24kHz RVQ)
        self.acoustic_tokenizer = AudioCodecTokenizer(
            config=self.audio_config,
            device=device
        )

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sr: int = 24000) -> Dict[str, torch.Tensor]:
        """
        Extracts both semantic and acoustic token representations directly from speech audio.
        Args:
            wav: Audio waveform tensor (batch, 1, samples) or (batch, samples)
            sr: sampling rate (default 24000)
        Returns:
            dict containing:
                "semantic_tokens": (batch, T_semantic) at 50 Hz
                "acoustic_tokens": (batch, 4, T_acoustic) at 75 Hz
                "sample_rate": 24000
        """
        # 1. Semantic units (phonetics / words)
        sem_tokens = self.semantic_tokenizer.encode(wav, sr=sr)

        # 2. Acoustic tokens (timbre / RVQ)
        ac_tokens = self.acoustic_tokenizer.encode(wav, sample_rate=sr)

        return {
            "semantic_tokens": sem_tokens,
            "acoustic_tokens": ac_tokens,
            "sample_rate": 24000
        }

    @torch.no_grad()
    def decode_acoustic(self, acoustic_tokens: torch.Tensor) -> torch.Tensor:
        """
        Decodes multi-codebook acoustic tokens into continuous audio waveform via Encodec.
        """
        return self.acoustic_tokenizer.decode(acoustic_tokens)
