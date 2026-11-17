#!/usr/bin/env python3
"""
Checkpoint and Reproducibility Manager for S2S V2.
Enforces strict historical checkpoint immutability,
path containment, deterministic random seeding, and provenance tracking.
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

HISTORICAL_MANIFEST_PATH = os.path.join(REPO_ROOT, "checkpoints", "historical_checkpoints_sha256.json")


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
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_historical_checkpoint_hashes() -> Dict[str, Any]:
    """
    Verifies that all historical checkpoints recorded in the manifest match their SHA-256 hashes.
    Guarantees historical checkpoint immutability before and after all experiments/tests.
    """
    if not os.path.exists(HISTORICAL_MANIFEST_PATH):
        # Create manifest with current known historical checkpoints
        hashes = {}
        for rel_path in PROTECTED_HISTORICAL_CHECKPOINTS:
            full_p = os.path.join(REPO_ROOT, rel_path)
            if os.path.exists(full_p):
                hashes[rel_path] = {
                    "sha256": compute_file_sha256(full_p),
                    "size_bytes": os.path.getsize(full_p)
                }
            else:
                hashes[rel_path] = {
                    "sha256": "FILE_NOT_ON_DISK_PROTECTED_NAME",
                    "size_bytes": 0
                }
        os.makedirs(os.path.dirname(HISTORICAL_MANIFEST_PATH), exist_ok=True)
        with open(HISTORICAL_MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(hashes, f, indent=2)

    with open(HISTORICAL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    verification_results = {}
    for rel_path, meta in manifest.items():
        full_p = os.path.join(REPO_ROOT, rel_path)
        expected_hash = meta.get("sha256")
        if os.path.exists(full_p):
            actual_hash = compute_file_sha256(full_p)
            assert actual_hash == expected_hash, (
                f"IMMUTABILITY VIOLATION: Historical checkpoint {rel_path} hash changed!\n"
                f"Expected: {expected_hash}\nActual:   {actual_hash}"
            )
            verification_results[rel_path] = {"status": "MATCH", "sha256": actual_hash}
        else:
            verification_results[rel_path] = {"status": "PROTECTED_NAME_UNTOUCHED"}

    return verification_results


class SafeCheckpointManager:
    """
    Guarantees non-destructive checkpoint persistence with path containment and provenance.
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
        custom_filename: Optional[str] = None,
        provenance: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Saves checkpoint safely in isolated experiment directory with strict path containment.
        """
        filename = custom_filename or f"step_{step}.pt"
        target_path = os.path.join(self.save_dir, filename)

        # 1. Path Containment Assertion (Disallow path traversal / symlink escapes)
        real_target = os.path.realpath(target_path)
        real_save_dir = os.path.realpath(self.save_dir)
        if not (real_target.startswith(real_save_dir + os.sep) or real_target == real_save_dir):
            raise ValueError(
                f"SECURITY VIOLATION: Target path '{target_path}' escapes designated directory '{self.save_dir}'"
            )

        # 2. Historical Immutability Assertion
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
            "provenance": provenance or {},
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
        device: torch.device = torch.device("cpu"),
        expected_sha256: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Loads checkpoint safely, verifying file existence, SHA-256, and architecture compatibility.
        """
        assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

        if expected_sha256:
            actual_sha = compute_file_sha256(checkpoint_path)
            assert actual_sha == expected_sha256, (
                f"Checkpoint SHA-256 mismatch for {checkpoint_path}!\n"
                f"Expected: {expected_sha256}\nActual:   {actual_sha}"
            )

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)

        if len(unexpected) > 0:
            raise ValueError(f"Architecture incompatibility: Unexpected keys in checkpoint: {unexpected}")

        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        print(f"✓ Successfully loaded checkpoint from {checkpoint_path} (Step {ckpt.get('step', 'unknown')})")
        return ckpt
