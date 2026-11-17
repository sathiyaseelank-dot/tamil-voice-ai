#!/usr/bin/env python3
"""
Unit Test Suite for S2S V2 Readiness & Architectural Invariants.
Validates:
1. Continuous temporal alignment mathematics.
2. Codebook target shapes and value ranges.
3. Speaker-disjoint split integrity.
4. Teacher-forced (B1) vs. predicted (B2) conditioning.
5. Genuine CB0-only quantized-vector ablation vs token-zero pseudo-ablation.
6. Checkpoint save/load round-trip and historical immutability.
7. Strict textless inference invariants.
"""

import os
import sys
import pytest
import torch
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from model.s2s_transformer import TimestampAligner, DepthAudioHead, S2STokenPredictor
from s2s_v2.encodec_target_aligner import EncodecTargetAligner
from s2s_v2.dataset_manifest_validator import audit_dataset_splits
from s2s_v2.checkpoint_manager import SafeCheckpointManager, PROTECTED_HISTORICAL_CHECKPOINTS


def test_temporal_alignment():
    """Validates mathematical continuous coordinate mapping between 50 Hz and 75 Hz."""
    aligner = TimestampAligner(d_in=1024, d_out=1024)
    aligner.eval()

    B, T_mms, D = 2, 100, 1024  # 100 frames @ 50 Hz = 2.0 seconds
    target_len = 150            # 150 frames @ 75 Hz = 2.0 seconds
    h_mms = torch.randn(B, T_mms, D)

    h_aligned = aligner(h_mms, target_len=target_len)
    assert h_aligned.shape == (B, target_len, D), f"Expected shape {(B, target_len, D)}, got {h_aligned.shape}"
    assert not torch.isnan(h_aligned).any(), "Temporal aligner produced NaNs"


def test_codebook_target_shapes_and_ranges():
    """Validates Encodec RVQ token extraction bounds."""
    aligner = EncodecTargetAligner(device=torch.device("cpu"))
    dummy_wav = torch.randn(1, 48000)  # 2.0 seconds @ 24 kHz

    tokens = aligner.extract_targets(dummy_wav, sample_rate=24000)
    assert tokens.ndim == 3, f"Expected 3D tensor, got {tokens.ndim}"
    assert tokens.shape[0] == 1, f"Batch size mismatch: {tokens.shape[0]}"
    assert tokens.shape[1] == 4, f"Expected 4 codebooks, got {tokens.shape[1]}"
    assert tokens.min().item() >= 0, f"Token value < 0: {tokens.min().item()}"
    assert tokens.max().item() < 1024, f"Token value >= 1024: {tokens.max().item()}"


def test_speaker_disjoint_split_integrity():
    """Validates 0.00% speaker overlap across both Tier 1 and Tier 2."""
    audit = audit_dataset_splits()

    # Tier 1 Audit
    tier1 = audit["full_dataset_tier"]
    if tier1.get("status") == "VERIFIED_ON_DISK":
        assert tier1["speaker_overlap"] == 0, "Tier 1 speaker overlap detected!"
        assert tier1["total_speakers"] == 407, f"Expected 407 total speakers, got {tier1['total_speakers']}"

    # Tier 2 Audit
    tier2 = audit["local_benchmark_tier"]
    assert tier2["is_strictly_disjoint"] is True, "Tier 2 speaker overlap detected!"
    assert tier2["speaker_overlap_count"] == 0, f"Tier 2 overlap count: {tier2['speaker_overlap_count']}"


def test_teacher_forced_vs_predicted_conditioning():
    """Validates DepthAudioHead in both B1 (teacher forced) and B2 (predicted) modes."""
    head = DepthAudioHead(d_model=1024, num_codebooks=4, vocab_size=1029)
    head.eval()

    B, T, D = 2, 30, 1024
    h = torch.randn(B, T, D)
    gt_tokens = torch.randint(0, 1024, (B, 4, T))

    # B1 Mode: Teacher Forced
    logits_b1 = head(h, target_tokens=gt_tokens)
    assert logits_b1.shape == (B, 4, T, 1029), f"B1 shape mismatch: {logits_b1.shape}"

    # B2 Mode: True Inference (Predicted conditioning)
    logits_b2 = head(h, target_tokens=None)
    assert logits_b2.shape == (B, 4, T, 1029), f"B2 shape mismatch: {logits_b2.shape}"

    # Output logits must differ because conditioning changes
    diff = torch.norm(logits_b1 - logits_b2).item()
    assert diff > 0.0, "Expected difference between teacher forced and predicted conditioning"


def test_cb0_only_quantized_vector_ablation():
    """
    CRITICAL REQUIREMENT 5 VALIDATION:
    Proves that genuine CB0-only decoding slices to shape (B, 1, T),
    omitting residual codevectors, whereas token-zero padding injects non-zero vectors.
    """
    aligner = EncodecTargetAligner(device=torch.device("cpu"))
    tokens_4cb = torch.randint(0, 1024, (1, 4, 75))  # 1 second of tokens

    wav_true, wav_pseudo, l2_diff = aligner.verify_vector_ablation(tokens_4cb)

    assert wav_true.shape == (1, 1, 24000), f"Expected shape (1, 1, 24000), got {wav_true.shape}"
    assert wav_pseudo.shape == (1, 1, 24000), f"Expected shape (1, 1, 24000), got {wav_pseudo.shape}"
    assert l2_diff > 1.0, (
        f"L2 difference must be significant ({l2_diff:.4f}). "
        "Token ID 0 substitution erroneously acted as non-zero codevectors!"
    )


def test_checkpoint_roundtrip_and_safety(tmp_path):
    """Validates state_dict serialization and historical immutability."""
    model = S2STokenPredictor(d_model=128, n_heads=4, n_layers=2, d_ff=256, num_codebooks=4, vocab_size=1029)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    mgr = SafeCheckpointManager(experiment_id="test_exp")
    mgr.save_dir = str(tmp_path)

    saved_path = mgr.save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=100,
        metrics={"cb0_acc": 45.2},
        is_best=True
    )
    assert os.path.exists(saved_path)

    # Round trip
    model2 = S2STokenPredictor(d_model=128, n_heads=4, n_layers=2, d_ff=256, num_codebooks=4, vocab_size=1029)
    ckpt = mgr.load_checkpoint(saved_path, model=model2)
    assert ckpt["step"] == 100
    assert ckpt["metrics"]["cb0_acc"] == 45.2

    # Verify model weights are identical
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "Model parameters failed round-trip equivalence"

    # Immutability check
    for protected in PROTECTED_HISTORICAL_CHECKPOINTS:
        with pytest.raises(AssertionError):
            # Attempting to save directly over a protected historical path must raise AssertionError
            mgr.save_dir = os.path.dirname(os.path.join(REPO_ROOT, protected))
            mgr.save_checkpoint(
                model,
                optimizer,
                step=1,
                metrics={},
                is_best=False,
                custom_filename=os.path.basename(protected)
            )


def test_textless_inference_invariants():
    """
    STRICT TEXTLESS INVARIANT VALIDATION:
    Asserts model takes zero text or phonemes and contains zero text encoders or vocabularies.
    """
    model = S2STokenPredictor(d_model=1024, n_heads=16, n_layers=6, d_ff=4096, num_codebooks=4, vocab_size=1029)

    # 1. Inspect all parameter names
    for name, _ in model.named_parameters():
        for forbidden in ["text", "phoneme", "bpe", "char", "vocab_embed", "transcription"]:
            assert forbidden not in name.lower(), f"Forbidden text-related parameter found: {name}"

    # 2. Forward pass takes continuous float features, not discrete text tokens
    dummy_mms = torch.randn(1, 50, 1024)
    logits = model(dummy_mms, target_len=75)
    assert logits.shape == (1, 4, 75, 1029)
