#!/usr/bin/env python3
"""
Experiment A: Dedicated CB0-Only Phonetic Backbone Learning.
Focuses 100% loss concentration on Codebook 0 (the phonetic envelope).
Codec remains strictly frozen.
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from model.s2s_transformer import S2STokenPredictor
from s2s_v2.encodec_target_aligner import EncodecTargetAligner
from s2s_v2.mms_feature_extractor import MMSFeatureExtractor


class ExperimentATrainer:
    """
    Experiment A Engine: Trains S2STokenPredictor solely on Codebook 0.
    """
    def __init__(
        self,
        model: S2STokenPredictor,
        aligner: EncodecTargetAligner,
        device: torch.device,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        save_dir: str = "checkpoints/exp_a"
    ):
        self.model = model.to(device)
        self.aligner = aligner
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Loss weights strictly 1.0 for CB0, 0.0 for residuals
        self.loss_weights = [1.0, 0.0, 0.0, 0.0]

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            weight_decay=weight_decay
        )

    def train_step(self, h_mms: torch.Tensor, tgt_tokens: torch.Tensor) -> Dict[str, float]:
        """
        Executes a single optimization step focusing 100% on CB0.
        Args:
            h_mms: (batch, T_mms, 1024)
            tgt_tokens: (batch, 4, T_codec)
        """
        self.model.train()
        self.optimizer.zero_grad()

        target_len = tgt_tokens.shape[-1]
        # In Exp A, no residual teacher forcing needed for CB0
        logits = self.model(h_mms, target_len=target_len, target_tokens=None)

        # Compute CB0 cross-entropy
        l_0 = logits[:, 0, :, :].reshape(-1, self.model.vocab_size)
        t_0 = tgt_tokens[:, 0, :].reshape(-1)
        loss = F.cross_entropy(l_0, t_0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            pred_0 = torch.argmax(l_0, dim=-1)
            acc_top1 = (pred_0 == t_0).float().mean().item() * 100.0
            
            top5 = torch.topk(l_0, k=5, dim=-1).indices
            acc_top5 = (top5 == t_0.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0

        return {
            "loss": loss.item(),
            "cb0_acc_top1": acc_top1,
            "cb0_acc_top5": acc_top5
        }

    @torch.no_grad()
    def evaluate_step(self, h_mms: torch.Tensor, tgt_tokens: torch.Tensor) -> Dict[str, Any]:
        """
        Evaluates CB0 predictions and synthesizes genuine CB0-only audio.
        """
        self.model.eval()
        target_len = tgt_tokens.shape[-1]
        logits = self.model(h_mms, target_len=target_len, target_tokens=None)

        pred_tokens = torch.argmax(logits, dim=-1)  # (batch, 4, target_len)
        pred_cb0 = pred_tokens[:, :1, :]           # True CB0 slicing

        # Genuine CB0-only audio decoding (n_q = 1)
        wav_cb0_only = self.aligner.decode_tokens(pred_cb0, num_codebooks=1)
        wav_gt_ref = self.aligner.decode_tokens(tgt_tokens, num_codebooks=4)

        t_0 = tgt_tokens[:, 0, :].reshape(-1)
        p_0 = pred_tokens[:, 0, :].reshape(-1)
        acc_top1 = (p_0 == t_0).float().mean().item() * 100.0

        # Unigram entropy
        counts = torch.bincount(p_0, minlength=1024).float()
        probs = counts / counts.sum().clamp(min=1.0)
        nonzero_probs = probs[probs > 0]
        entropy = -torch.sum(nonzero_probs * torch.log2(nonzero_probs)).item()

        # Repetition metric (consecutive identical tokens)
        reps = (p_0[1:] == p_0[:-1]).float().mean().item() * 100.0

        return {
            "cb0_acc_top1": acc_top1,
            "cb0_unigram_entropy": entropy,
            "repetition_rate_pct": reps,
            "wav_cb0_only": wav_cb0_only.cpu(),
            "wav_gt_ref": wav_gt_ref.cpu()
        }


def build_exp_a_system(device: Optional[torch.device] = None) -> ExperimentATrainer:
    """Builds a configured Experiment A system."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = S2STokenPredictor(
        d_model=1024,
        n_heads=16,
        n_layers=6,
        d_ff=4096,
        num_codebooks=4,
        vocab_size=1029
    )
    aligner = EncodecTargetAligner(device=dev)
    trainer = ExperimentATrainer(model=model, aligner=aligner, device=dev)
    return trainer


if __name__ == "__main__":
    print("Experiment A module loaded successfully. Full training remains prohibited pending review.")
