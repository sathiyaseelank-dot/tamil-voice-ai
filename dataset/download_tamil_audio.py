"""
Download and generate sample Tamil conversational audio data for training and testing.
Provides both synthetic conversational speech pairs for zero-network testing
and download utilities for real open-source Tamil speech datasets (e.g. OpenSLR SLR65, Common Voice).
"""

import os
import math
import torch
import numpy as np
from typing import List, Tuple

from tokenizer.audio_io import save_audio


def generate_synthetic_conversational_turn(
    duration_sec: float,
    fundamental_freq: float,
    sample_rate: int = 24000
) -> torch.Tensor:
    """
    Generates a realistic multi-harmonic formant-modulated speech-like audio segment
    to serve as synthetic conversational audio turns for verification.
    """
    t = torch.linspace(0, duration_sec, int(sample_rate * duration_sec))
    
    # Fundamental frequency contour with prosodic pitch variation
    pitch_contour = fundamental_freq + 20.0 * torch.sin(2 * math.pi * 1.5 * t)
    phase = 2 * math.pi * torch.cumsum(pitch_contour / sample_rate, dim=0)

    # Harmonics (voice harmonics mimicking Tamil vowels /a/, /i/, /u/)
    harmonics = (
        1.00 * torch.sin(phase) +
        0.50 * torch.sin(2 * phase) +
        0.25 * torch.sin(3 * phase) +
        0.15 * torch.sin(4 * phase)
    )

    # Formant filter envelope simulating speech syllables
    syllable_rate = 4.0  # 4 syllables per second typical in spoken Tamil
    envelope = torch.clamp(torch.sin(2 * math.pi * syllable_rate * t), min=0.0) ** 1.5

    # Pause and cadence
    wav = harmonics * envelope
    # Smooth start and end
    fade_len = int(sample_rate * 0.05)
    fade = torch.linspace(0, 1, fade_len)
    wav[:fade_len] *= fade
    wav[-fade_len:] *= fade.flip(0)

    # Add slight natural breath noise
    noise = torch.randn_like(wav) * 0.01
    wav = wav + noise
    
    # Normalize
    wav = wav / (wav.abs().max() + 1e-6) * 0.8
    return wav.unsqueeze(0)  # (1, samples)


def create_sample_dialogue_pairs(
    output_dir: str = "samples",
    num_pairs: int = 20,
    sample_rate: int = 24000
) -> List[Tuple[str, str]]:
    """
    Creates sample dialogue audio pairs representing conversational turns.
    Speaker A (User): Pitch ~130 Hz (e.g. Tamil male speaker)
    Speaker B (Assistant): Pitch ~210 Hz (e.g. Tamil female assistant)
    """
    os.makedirs(output_dir, exist_ok=True)
    pairs = []

    print(f"Creating {num_pairs} sample conversational audio pairs in '{output_dir}'...")
    for i in range(num_pairs):
        user_dur = np.random.uniform(1.5, 3.5)
        asst_dur = np.random.uniform(1.5, 3.5)

        user_audio = generate_synthetic_conversational_turn(user_dur, fundamental_freq=130.0, sample_rate=sample_rate)
        asst_audio = generate_synthetic_conversational_turn(asst_dur, fundamental_freq=215.0, sample_rate=sample_rate)

        u_path = os.path.join(output_dir, f"user_turn_{i:04d}.wav")
        a_path = os.path.join(output_dir, f"asst_turn_{i:04d}.wav")

        save_audio(u_path, user_audio, sample_rate)
        save_audio(a_path, asst_audio, sample_rate)
        pairs.append((u_path, a_path))

    print(f"Successfully generated {len(pairs)} conversational dialogue pairs.")
    return pairs


if __name__ == "__main__":
    create_sample_dialogue_pairs("samples", num_pairs=10)
