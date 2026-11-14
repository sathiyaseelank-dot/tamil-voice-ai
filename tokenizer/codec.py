"""
Neural Audio Codec Interface (Residual Vector Quantization).
Wraps Encodec 24kHz to convert continuous audio waveforms directly into
discrete multi-codebook token IDs, and reconstruct waveforms from token IDs.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from encodec import EncodecModel
from encodec.utils import convert_audio

from config import AudioConfig


class AudioCodecTokenizer:
    """
    Tokenizer that converts audio to discrete RVQ token sequences and vice versa.
    No text transcriptions or phoneme dictionaries are involved.
    """
    def __init__(self, config: Optional[AudioConfig] = None, device: str = "cpu"):
        self.config = config or AudioConfig()
        self.device = torch.device(device)
        
        # Load pre-trained neural audio codec
        self.model = EncodecModel.encodec_model_24khz()
        # Set target bandwidth corresponding to desired number of codebooks
        # 3.0 kbps gives 4 codebooks; 6.0 kbps gives 8 codebooks
        bandwidth = 3.0 if self.config.num_codebooks <= 4 else 6.0
        self.model.set_target_bandwidth(bandwidth)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sample_rate: int = 24000) -> torch.Tensor:
        """
        Encodes a continuous speech waveform into discrete RVQ tokens.
        Args:
            wav: Tensor of shape (batch, 1, samples) or (batch, samples)
            sample_rate: sampling rate of input audio (resampled if not 24kHz)
        Returns:
            tokens: Int tensor of shape (batch, num_codebooks, seq_len)
        """
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        wav = wav.to(self.device)

        if sample_rate != self.config.sample_rate:
            wav = convert_audio(wav, sample_rate, self.config.sample_rate, self.model.channels)

        # Encodec returns a list of frames: [(frame_codes, scale)]
        # frame_codes has shape: [batch, num_codebooks, frame_steps]
        encoded_frames = self.model.encode(wav)
        codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)

        # Slice to config.num_codebooks if codec produced more
        if codes.shape[1] > self.config.num_codebooks:
            codes = codes[:, :self.config.num_codebooks, :]

        return codes.long()

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decodes discrete RVQ tokens back into continuous audio waveform.
        Args:
            tokens: Int tensor of shape (batch, num_codebooks, seq_len)
        Returns:
            waveform: Tensor of shape (batch, 1, time_samples)
        """
        tokens = tokens.to(self.device)
        batch_size, n_q, seq_len = tokens.shape

        # Handle special tokens (clamp / map special tokens >= codebook_size to silence code)
        cleaned_tokens = tokens.clone()
        special_mask = cleaned_tokens >= self.config.codebook_size
        cleaned_tokens[special_mask] = 0

        # If model expects more codebooks than provided, zero-pad codebook dimension
        expected_codebooks = self.model.quantizer.n_q
        if n_q < expected_codebooks:
            pad_q = torch.zeros(
                (batch_size, expected_codebooks - n_q, seq_len),
                dtype=tokens.dtype,
                device=self.device
            )
            cleaned_tokens = torch.cat([cleaned_tokens, pad_q], dim=1)

        # Reconstruct frame tuple expected by encodec: [(codes, scale)]
        # scale is None for 24kHz model
        frames = [(cleaned_tokens, None)]
        waveform = self.model.decode(frames)
        
        # If entire frame was special token (e.g. silence), mute that section
        if special_mask.all(dim=1).any():
            frame_special = special_mask.all(dim=1)  # (batch, seq_len)
            upsample_ratio = self.config.sample_rate // self.config.frame_rate
            for b in range(batch_size):
                for t in range(seq_len):
                    if frame_special[b, t]:
                        start = t * upsample_ratio
                        end = min(start + upsample_ratio, waveform.shape[-1])
                        waveform[b, :, start:end] = 0.0

        return waveform
