#!/usr/bin/env python3
"""
MMS-300M Continuous Feature Extractor & Cacher.
Extracts 1024-dimensional continuous representations at 50 Hz (16 kHz audio)
with strict zero text conditioning.
"""

import os
import torch
import torchaudio
from typing import Optional, Dict
from transformers import Wav2Vec2Model


class MMSFeatureExtractor:
    """
    Extracts continuous self-supervised representations from Meta MMS-300M.
    Strictly textless: operates directly on continuous waveform samples.
    """
    def __init__(self, model_id: str = "facebook/mms-300m", device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        
        # Load frozen MMS-300M
        self.model = Wav2Vec2Model.from_pretrained(model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device)

    @torch.no_grad()
    def extract_features(self, wav: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extracts continuous hidden states.
        Args:
            wav: (batch, samples) or (batch, 1, samples)
            sample_rate: sampling rate of input audio
        Returns:
            h_mms: (batch, T_mms, 1024)
        """
        if wav.ndim == 3:
            wav = wav.squeeze(1)
        wav = wav.to(self.device)

        if sample_rate != 16000:
            wav = torchaudio.functional.resample(wav, sample_rate, 16000)

        # Normalize amplitude peak
        max_val = wav.abs().max()
        if max_val > 0:
            wav = (wav / max_val) * 0.95

        out = self.model(wav)
        return out.last_hidden_state

    @torch.no_grad()
    def extract_and_cache(self, audio_dict: Dict[str, torch.Tensor], save_path: str):
        """Extracts features for a dictionary of utterances and caches to disk."""
        cached = {}
        for key, wav in audio_dict.items():
            h = self.extract_features(wav.unsqueeze(0))
            cached[key] = h.cpu()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(cached, save_path)
        print(f"✓ Cached {len(cached)} MMS feature tensors to {save_path}")
