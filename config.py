"""
Configuration for Textless Tamil Speech-to-Speech Model (Dessa / Moshi style).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AudioConfig:
    sample_rate: int = 24000
    frame_rate: int = 75  # Encodec 24kHz outputs 75 frames/second
    num_codebooks: int = 4  # 4 RVQ codebooks (3.0 kbps) for efficient training/inference; can use 8 (6.0 kbps)
    codebook_size: int = 1024  # Encodec standard codebook vocabulary
    
    # Special tokens placed after codebook_size (1024..1028)
    pad_token_id: int = 1024
    bos_token_id: int = 1025
    eos_token_id: int = 1026
    silence_token_id: int = 1027
    interrupt_token_id: int = 1028

    @property
    def vocab_size(self) -> int:
        # 1024 acoustic tokens + 5 special tokens
        return self.codebook_size + 5


@dataclass
class ModelConfig:
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    max_seq_len: int = 750  # 10 seconds of conversational audio (75 frames/sec * 10)
    audio: AudioConfig = field(default_factory=AudioConfig)


@dataclass
class TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_epochs: int = 10
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints"
    save_every_epochs: int = 2
