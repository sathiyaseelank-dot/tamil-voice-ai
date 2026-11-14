"""
Audio I/O utilities using soundfile for robust reading and saving of audio files.
"""

import torch
import soundfile as sf
from typing import Tuple


def load_audio(file_path: str) -> Tuple[torch.Tensor, int]:
    """
    Loads audio file into a torch tensor of shape (1, samples) (mono) and returns sample rate.
    """
    data, sr = sf.read(file_path)
    # data is (samples,) or (samples, channels)
    tensor = torch.from_numpy(data).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)  # (1, samples)
    elif tensor.ndim == 2:
        # Convert stereo to mono by averaging channels
        tensor = tensor.mean(dim=-1, keepdim=True).transpose(0, 1)  # (1, samples)
    return tensor, sr


def save_audio(file_path: str, wav: torch.Tensor, sample_rate: int = 24000):
    """
    Saves a torch tensor waveform to disk as a .wav file.
    wav can be (samples,), (1, samples), or (1, 1, samples).
    """
    wav_np = wav.detach().cpu().squeeze().numpy()
    sf.write(file_path, wav_np, sample_rate)
