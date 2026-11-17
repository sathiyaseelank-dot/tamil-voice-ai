#!/usr/bin/env python3
"""
Experiment C: Exposure Bias Mitigation and Inference Robustness.
Implements 4 formal protocols to bridge the train-eval gap:
- Protocol C1: Baseline Free-Run
- Protocol C2: Scheduled Sampling (Frame-level coin-flip)
- Protocol C3: Soft-Embedding Distribution Projection (Differentiable expectation)
- Protocol C4: End-to-End Joint Multi-Task Fine-Tuning
"""

import os
import sys
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from model.s2s_transformer import S2STokenPredictor
from s2s_v2.encodec_target_aligner import EncodecTargetAligner


class ScheduledSamplingDepthHead(nn.Module):
    """
    DepthAudioHead enhanced with scheduled sampling and soft distribution embeddings.
    """
    def __init__(self, base_head, d_model: int = 1024, num_codebooks: int = 4, vocab_size: int = 1029):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.heads = base_head.heads
        self.depth_embeddings = base_head.depth_embeddings
        self.d_model = d_model

    def forward_scheduled_sampling(
        self,
        h: torch.Tensor,
        target_tokens: torch.Tensor,
        p_teacher_forcing: float = 1.0
    ) -> torch.Tensor:
        """
        Protocol C2: Scheduled Sampling at frame-level granularity.
        For each frame t and depth k, coin flip with prob p determines whether
        ground-truth token or model's own predicted token is embedded.
        """
        B, T, D = h.shape
        logits_list = []
        curr_h = h

        # CB0 unconditionally predicted
        logits_0 = self.heads[0](curr_h)
        logits_list.append(logits_0)

        for k in range(1, self.num_codebooks):
            prev_pred = torch.argmax(logits_list[-1].detach(), dim=-1)  # (B, T)
            prev_gt = target_tokens[:, k - 1, :]                         # (B, T)

            # Frame-level coin-flip mask
            mask = (torch.rand((B, T), device=h.device) < p_teacher_forcing).long()
            sampled_tokens = mask * prev_gt + (1 - mask) * prev_pred
            sampled_tokens = torch.clamp(sampled_tokens, 0, self.vocab_size - 1)

            depth_emb = self.depth_embeddings[k - 1](sampled_tokens)
            curr_h = curr_h + depth_emb
            logits_k = self.heads[k](curr_h)
            logits_list.append(logits_k)

        return torch.stack(logits_list, dim=1)

    def forward_soft_embedding(
        self,
        h: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Protocol C3: Soft-Embedding Distribution Projection.
        Computes expected depth embedding over the softmax distribution:
            e_depth = Sum_v [ P(y = v) * E(v) ]
        Enables end-to-end gradient flow without discrete argmax bottlenecks.
        """
        B, T, D = h.shape
        logits_list = []
        curr_h = h

        logits_0 = self.heads[0](curr_h)
        logits_list.append(logits_0)

        for k in range(1, self.num_codebooks):
            # Compute softmax probabilities over vocab with temperature
            probs = F.softmax(logits_list[-1] / max(temperature, 0.05), dim=-1)  # (B, T, V)
            # Embedding weight matrix is (V, D)
            emb_weight = self.depth_embeddings[k - 1].weight  # (V, D)
            soft_depth_emb = torch.matmul(probs, emb_weight)  # (B, T, D)

            curr_h = curr_h + soft_depth_emb
            logits_k = self.heads[k](curr_h)
            logits_list.append(logits_k)

        return torch.stack(logits_list, dim=1)


class ExperimentCTrainer:
    """
    Experiment C Engine: Benchmarks and compares exposure bias mitigation protocols.
    """
    def __init__(
        self,
        model: S2STokenPredictor,
        aligner: EncodecTargetAligner,
        device: torch.device,
        protocol: str = "C2",  # C1, C2, C3, C4
        p_min: float = 0.40,
        k_decay: int = 3000,
        temp_init: float = 1.0,
        temp_min: float = 0.1,
        save_dir: str = "checkpoints/exp_c"
    ):
        self.model = model.to(device)
        self.aligner = aligner
        self.device = device
        self.protocol = protocol
        self.p_min = p_min
        self.k_decay = k_decay
        self.temp_init = temp_init
        self.temp_min = temp_min
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Wrap head with scheduled sampling capabilities
        self.scheduled_head = ScheduledSamplingDepthHead(
            self.model.head,
            d_model=self.model.d_model,
            num_codebooks=self.model.num_codebooks,
            vocab_size=self.model.vocab_size
        )
        self.model.head = self.scheduled_head

        # Configure trainable parameters per protocol
        if protocol == "C4":
            # Joint End-to-End Fine-Tuning: all parameters trainable
            for p in self.model.parameters():
                p.requires_grad = True
            lr = 5e-5
            self.loss_weights = [1.0, 0.5, 0.25, 0.125]
        else:
            # Protocols C1-C3: optimize depth head while backbone remains steady
            lr = 1e-4
            self.loss_weights = [0.0, 1.0, 0.5, 0.25]

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.98), weight_decay=0.01)

    def get_scheduled_p(self, step: int) -> float:
        """Linear decay schedule for teacher-forcing probability."""
        return max(self.p_min, 1.0 - (step / float(self.k_decay)))

    def get_temperature(self, step: int) -> float:
        """Exponential temperature annealing for soft embeddings."""
        return max(self.temp_min, self.temp_init * (0.999 ** step))

    def train_step(self, h_mms: torch.Tensor, tgt_tokens: torch.Tensor, step: int) -> Dict[str, float]:
        """
        Executes one training step according to selected protocol.
        """
        self.model.train()
        self.optimizer.zero_grad()
        target_len = tgt_tokens.shape[-1]

        # 1. Aligner + Transformer
        h_aligned = self.model.aligner(h_mms, target_len=target_len)
        h_trans = self.model.transformer(h_aligned)
        h_norm = self.model.ln_f(h_trans)

        # 2. Forward head based on protocol
        p_t = self.get_scheduled_p(step)
        temp_t = self.get_temperature(step)

        if self.protocol == "C1":
            # Baseline Free-Run
            logits = self.model.head.forward_scheduled_sampling(h_norm, tgt_tokens, p_teacher_forcing=0.0)
        elif self.protocol == "C2":
            # Scheduled Sampling
            logits = self.model.head.forward_scheduled_sampling(h_norm, tgt_tokens, p_teacher_forcing=p_t)
        elif self.protocol == "C3":
            # Soft-Embedding Distribution Projection
            logits = self.model.head.forward_soft_embedding(h_norm, temperature=temp_t)
        elif self.protocol == "C4":
            # Joint Fine-Tuning with Scheduled Sampling
            logits = self.model.head.forward_scheduled_sampling(h_norm, tgt_tokens, p_teacher_forcing=p_t)
        else:
            raise ValueError(f"Unknown protocol: {self.protocol}")

        # 3. Compute loss
        loss = 0.0
        accs = {}
        for k in range(4):
            l_k = logits[:, k, :, :].reshape(-1, self.model.vocab_size)
            t_k = tgt_tokens[:, k, :].reshape(-1)
            ce_k = F.cross_entropy(l_k, t_k)
            if self.loss_weights[k] > 0:
                loss = loss + self.loss_weights[k] * ce_k

            with torch.no_grad():
                pred_k = torch.argmax(l_k, dim=-1)
                accs[f"cb{k}_acc"] = (pred_k == t_k).float().mean().item() * 100.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        metrics = {
            "loss": loss.item(),
            "p_t": p_t,
            "temp_t": temp_t
        }
        metrics.update(accs)
        return metrics

    @torch.no_grad()
    def measure_exposure_gap(self, h_mms: torch.Tensor, tgt_tokens: torch.Tensor) -> Dict[str, float]:
        """
        Measures the exposure bias degradation gap:
        Delta = Metric(Inference Mode) - Metric(Teacher Forced Mode)
        """
        self.model.eval()
        target_len = tgt_tokens.shape[-1]

        # 1. Teacher-Forced Mode (GT conditioning)
        logits_tf = self.model(h_mms, target_len=target_len, target_tokens=tgt_tokens)
        pred_tf = torch.argmax(logits_tf, dim=-1)
        acc_res_tf = (pred_tf[:, 1:, :] == tgt_tokens[:, 1:, :]).float().mean().item() * 100.0

        # 2. Free-Run Inference Mode (Predicted conditioning)
        logits_inf = self.model(h_mms, target_len=target_len, target_tokens=None)
        pred_inf = torch.argmax(logits_inf, dim=-1)
        acc_res_inf = (pred_inf[:, 1:, :] == tgt_tokens[:, 1:, :]).float().mean().item() * 100.0

        gap = acc_res_tf - acc_res_inf
        return {
            "acc_residuals_teacher_forced": acc_res_tf,
            "acc_residuals_free_run_inference": acc_res_inf,
            "exposure_accuracy_gap_delta": gap
        }


def build_exp_c_system(protocol: str = "C2", device: Optional[torch.device] = None) -> ExperimentCTrainer:
    """Builds a configured Experiment C system."""
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
    trainer = ExperimentCTrainer(model=model, aligner=aligner, device=dev, protocol=protocol)
    return trainer


if __name__ == "__main__":
    print("Experiment C module loaded successfully. Protocols C1-C4 mathematically defined.")
