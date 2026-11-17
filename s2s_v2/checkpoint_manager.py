#!/usr/bin/env python3
"""
Checkpoint and Reproducibility Manager for S2S V2.
Enforces strict historical checkpoint immutability,
deterministic random seeding, and isolated experimental checkpoint tracking.
"""

import os
import sys
import json
import torch
import random
import hashlib
import numpy as np
from typing import Dict, Any, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

PROTECTED_HISTORICAL_CHECKPOINTS = [
    "checkpoints/tamil_speech_lm_step_1000.pt",
    "checkpoints/tamil_speech_lm_step_2000.pt",
    "checkpoints/tamil_speech_lm_wsp_grafted.pt",
    "checkpoints/wsp_tamil_speech_generator.pt"
]


def set_seed(seed: int = 42):
    """Sets deterministic random seed across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Deterministic seed locked: {seed}")


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class SafeCheckpointManager:
    """
    Guarantees non-destructive checkpoint persistence.
    """
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.save_dir = os.path.join(REPO_ROOT, "checkpoints", f"exp_{experiment_id.lower()}")
        os.makedirs(self.save_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        metrics: Dict[str, Any],
        is_best: bool = False,
        custom_filename: Optional[str] = None
    ) -> str:
        """
        Saves checkpoint safely in isolated experiment directory.
        """
        filename = custom_filename or f"step_{step}.pt"
        target_path = os.path.join(self.save_dir, filename)

        # Immutability assertion
        rel_path = os.path.relpath(target_path, REPO_ROOT).replace("\\", "/")
        for protected in PROTECTED_HISTORICAL_CHECKPOINTS:
            if rel_path == protected or os.path.abspath(target_path) == os.path.abspath(os.path.join(REPO_ROOT, protected)):
                raise AssertionError(
                    f"CRITICAL ERROR: Attempted to overwrite protected historical checkpoint: {rel_path}"
                )

        state = {
            "experiment_id": self.experiment_id,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "saved_at": str(np.datetime64('now'))
        }
        torch.save(state, target_path)

        if is_best:
            best_path = os.path.join(self.save_dir, "best.pt")
            torch.save(state, best_path)

        return target_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, Any]:
        """
        Loads checkpoint safely without modifying source file.
        """
        assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        print(f"✓ Successfully loaded checkpoint from {checkpoint_path} (Step {ckpt.get('step', 'unknown')})")
        return ckpt
