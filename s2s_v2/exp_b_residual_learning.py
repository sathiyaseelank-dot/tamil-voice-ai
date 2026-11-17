#!/usr/bin/env python3
"""
Experiment B: Residual Codebook Learning (DepthAudioHead Optimization).
Separates evaluations into:
- B1: Teacher-Forced Ceiling (Ground-truth CB0 conditioning + Ground-truth CB0 decoding + Predicted residuals)
- B2: True Inference Condition (Predicted CB0 conditioning + Predicted CB0 decoding + Predicted residuals)
- Diagnostic B3: Cross-condition sensitivity analysis
Explicitly loads approved Experiment A checkpoint with provenance and SHA-256 validation.
"""

import os
import sys
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from model.s2s_transformer import S2STokenPredictor
from s2s_v2.encodec_target_aligner import EncodecTargetAligner


class ExperimentBTrainer:
    """
    Experiment B Engine: Trains residual codebook heads (CB1..CB3)
    with CB0 predictor frozen.
    """
    def __init__(
        self,
        model: S2STokenPredictor,
        aligner: EncodecTargetAligner,
        device: torch.device,
        freeze_backbone_and_cb0: bool = True,
        init_type: str = "checkpoint",  # "checkpoint" or "random"
        lr: float = 1e-4,
        save_dir: str = "checkpoints/exp_b"
    ):
        self.model = model.to(device)
        self.aligner = aligner
        self.device = device
        self.init_type = init_type
        self.save_dir = save_dir
        self.parent_checkpoint_info = None
        self.is_checkpoint_loaded = (init_type == "random")
        os.makedirs(save_dir, exist_ok=True)

        if freeze_backbone_and_cb0:
            # Freeze transformer backbone, positional encoding, and CB0 head
            for p in self.model.aligner.parameters():
                p.requires_grad = False
            if hasattr(self.model, "pos_enc") and isinstance(self.model.pos_enc, nn.Module):
                for p in self.model.pos_enc.parameters():
                    p.requires_grad = False
            for p in self.model.transformer.parameters():
                p.requires_grad = False
            for p in self.model.ln_f.parameters():
                p.requires_grad = False
            self.model.head.heads[0].weight.requires_grad = False
            if self.model.head.heads[0].bias is not None:
                self.model.head.heads[0].bias.requires_grad = False

        # Only train residual heads and depth embeddings
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.98), weight_decay=0.01)

        # Residual loss weights
        self.loss_weights = [0.0, 1.0, 0.5, 0.25]

    def load_exp_a_checkpoint(self, checkpoint_path: str, verify_sha256: str) -> Dict[str, Any]:
        """
        Loads approved Experiment A checkpoint into backbone and CB0 head.
        Requires mandatory SHA-256 validation and state_dict compatibility checks.
        """
        assert os.path.exists(checkpoint_path), f"Experiment A checkpoint not found: {checkpoint_path}"
        assert verify_sha256 is not None and len(verify_sha256) == 64, (
            f"Mandatory SHA-256 hash required (expected 64-char hex string, got '{verify_sha256}')"
        )

        h = hashlib.sha256()
        with open(checkpoint_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        actual_hash = h.hexdigest()
        assert actual_hash == verify_sha256, (
            f"MANDATORY SHA-256 MISMATCH for Exp A checkpoint {checkpoint_path}!\n"
            f"Expected: {verify_sha256}\nActual:   {actual_hash}"
        )

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(ckpt["model_state_dict"], strict=False)

        # Validate that no backbone or CB0 parameters are missing
        backbone_keys = [k for k in self.model.state_dict().keys() if not any(x in k for x in ["head.heads.1", "head.heads.2", "head.heads.3", "head.depth_embeddings"])]
        missing_backbone = [k for k in missing if k in backbone_keys]
        if missing_backbone:
            raise ValueError(f"Checkpoint compatibility failure: Missing required backbone keys: {missing_backbone}")
        if len(unexpected) > 0:
            raise ValueError(f"Checkpoint compatibility failure: Unexpected keys in checkpoint: {unexpected}")

        self.parent_checkpoint_info = {
            "path": checkpoint_path,
            "sha256": actual_hash,
            "step": ckpt.get("step", 0),
            "experiment_id": ckpt.get("experiment_id", "exp_a")
        }
        self.is_checkpoint_loaded = True
        print(f"✓ Loaded approved Experiment A checkpoint from {checkpoint_path} (SHA-256 & Provenance verified)")
        return ckpt

    def train_step(self, h_mms: torch.Tensor, tgt_tokens: torch.Tensor) -> Dict[str, float]:
        """
        Trains residual heads using ground-truth preceding codebook teacher forcing.
        Refuses training when init_type='checkpoint' until approved checkpoint is loaded.
        """
        if self.init_type == "checkpoint" and not self.is_checkpoint_loaded:
            raise RuntimeError(
                "MANDATORY INITIALIZATION FAILURE: Experiment B requires an approved Experiment A checkpoint before training. "
                "Call load_exp_a_checkpoint(checkpoint_path, verify_sha256) first."
            )

        # Keep frozen backbone in eval mode (no dropout) and residual heads in train mode
        self.model.aligner.eval()
        if hasattr(self.model, "pos_enc") and isinstance(self.model.pos_enc, nn.Module):
            self.model.pos_enc.eval()
        self.model.transformer.eval()
        self.model.ln_f.eval()
        self.model.head.heads[0].eval()
        for k in range(1, self.model.num_codebooks):
            self.model.head.heads[k].train()
            self.model.head.depth_embeddings[k - 1].train()

        self.optimizer.zero_grad()

        target_len = tgt_tokens.shape[-1]
        # Teacher forcing with target tokens
        logits = self.model(h_mms, target_len=target_len, target_tokens=tgt_tokens)

        loss = 0.0
        accs = {}
        for k in range(1, 4):
            l_k = logits[:, k, :, :].reshape(-1, self.model.vocab_size)
            t_k = tgt_tokens[:, k, :].reshape(-1)
            ce_k = F.cross_entropy(l_k, t_k)
            loss = loss + self.loss_weights[k] * ce_k

            with torch.no_grad():
                pred_k = torch.argmax(l_k, dim=-1)
                accs[f"cb{k}_acc"] = (pred_k == t_k).float().mean().item() * 100.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        metrics = {"loss": loss.item()}
        metrics.update(accs)
        return metrics

    @torch.no_grad()
    def evaluate_b1_teacher_forced(self, h_mms: torch.Tensor, tgt_tokens: torch.Tensor) -> Dict[str, Any]:
        """
        Condition B1: Teacher-Forced Acoustic Ceiling Evaluation
        - Ground-truth CB0 conditioning fed into residual heads
        - Ground-truth CB0 decoding in waveform synthesis
        - Predicted residual codebooks CB1..CB3
        - Compared against Ground-Truth CB0-only decoding
        """
        self.model.eval()
        target_len = tgt_tokens.shape[-1]

        # Condition residual heads on GT tokens
        logits = self.model(h_mms, target_len=target_len, target_tokens=tgt_tokens)
        pred_res = torch.argmax(logits, dim=-1)  # (B, 4, T)

        # B1 composite tokens: [GT CB0, Pred CB1, Pred CB2, Pred CB3]
        tokens_b1 = torch.cat([tgt_tokens[:, :1, :], pred_res[:, 1:, :]], dim=1)

        # Synthesize B1 waveform (4 codebooks)
        wav_b1 = self.aligner.decode_tokens(tokens_b1, num_codebooks=4)

        # Synthesize GT CB0-only waveform (1 codebook genuine ablation)
        wav_gt_cb0_only = self.aligner.decode_tokens(tgt_tokens[:, :1, :], num_codebooks=1)

        # Residual accuracies under B1
        res_accs = {}
        for k in range(1, 4):
            acc_k = (pred_res[:, k, :] == tgt_tokens[:, k, :]).float().mean().item() * 100.0
            res_accs[f"b1_cb{k}_acc"] = acc_k

        return {
            "evaluation_mode": "B1_TEACHER_FORCED_CEILING",
            "wav_b1_4cb": wav_b1.cpu(),
            "wav_gt_cb0_only": wav_gt_cb0_only.cpu(),
            "metrics": res_accs
        }

    @torch.no_grad()
    def evaluate_b2_inference_condition(self, h_mms: torch.Tensor, tgt_tokens: torch.Tensor) -> Dict[str, Any]:
        """
        Condition B2: True Inference-Condition Evaluation
        - Predicted CB0 conditioning fed into residual heads (NO ground-truth tokens provided)
        - Predicted CB0 decoding in waveform synthesis
        - Predicted residual codebooks CB1..CB3
        - Compared against Predicted CB0-only decoding
        """
        self.model.eval()
        target_len = tgt_tokens.shape[-1]

        # In true inference, target_tokens=None.
        logits = self.model(h_mms, target_len=target_len, target_tokens=None)
        pred_tokens_b2 = torch.argmax(logits, dim=-1)  # (B, 4, T)

        # Synthesize B2 waveform (4 codebooks)
        wav_b2_4cb = self.aligner.decode_tokens(pred_tokens_b2, num_codebooks=4)

        # Synthesize Predicted CB0-only waveform (1 codebook genuine ablation)
        wav_pred_cb0_only = self.aligner.decode_tokens(pred_tokens_b2[:, :1, :], num_codebooks=1)

        # Accuracies under true inference
        accs = {}
        for k in range(4):
            acc_k = (pred_tokens_b2[:, k, :] == tgt_tokens[:, k, :]).float().mean().item() * 100.0
            accs[f"b2_cb{k}_acc"] = acc_k

        return {
            "evaluation_mode": "B2_TRUE_INFERENCE_CONDITION",
            "wav_b2_4cb": wav_b2_4cb.cpu(),
            "wav_pred_cb0_only": wav_pred_cb0_only.cpu(),
            "metrics": accs
        }


def build_exp_b_system(
    init_type: str = "checkpoint",
    pos_enc_type: str = "sinusoidal",
    device: Optional[torch.device] = None
) -> ExperimentBTrainer:
    """Builds a configured Experiment B system."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = S2STokenPredictor(
        d_model=1024,
        n_heads=16,
        n_layers=6,
        d_ff=4096,
        num_codebooks=4,
        vocab_size=1024,
        pos_enc_type=pos_enc_type
    )
    aligner = EncodecTargetAligner(device=dev)
    trainer = ExperimentBTrainer(
        model=model,
        aligner=aligner,
        device=dev,
        init_type=init_type
    )
    return trainer


if __name__ == "__main__":
    print("Experiment B module loaded successfully. B1 and B2 protocols formally decoupled.")
