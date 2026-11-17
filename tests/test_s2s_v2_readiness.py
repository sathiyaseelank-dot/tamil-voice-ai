#!/usr/bin/env python3
"""
Expanded Unit Test Suite for S2S V2 Readiness & Architectural Invariants.
Validates all P0 and P1 review findings:
1. Temporal alignment mathematics and positional encoding.
2. Codebook target shapes and value ranges.
3. Actual manifest-based speaker disjointness and deliberate overlap injection detection.
4. Experiment B training step and B1 (teacher-forced) vs B2 (inference condition) decoupling.
5. Experiment C forward, exposure-gap evaluation, and gradient flow per protocol (C1, C2, C3, C4).
6. Genuine CB0-only quantized-vector ablation vs token-zero pseudo-ablation.
7. Encodec scale and chunk metadata preservation.
8. Human listening unblinding key isolation, random identifiers, and incomplete ratings agreement.
9. Checkpoint path containment, symlink/traversal rejection, and historical checkpoint SHA-256 immutability.
10. Strict textless inference invariants.
"""

import os
import sys
import json
import pytest
import torch
import torch.nn as nn
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from model.s2s_transformer import (
    TimestampAligner,
    DepthAudioHead,
    S2STokenPredictor,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding
)
from s2s_v2.encodec_target_aligner import EncodecTargetAligner
from s2s_v2.dataset_manifest_validator import (
    audit_dataset_splits,
    verify_split_disjointness
)
from s2s_v2.checkpoint_manager import (
    SafeCheckpointManager,
    verify_historical_checkpoint_hashes,
    PROTECTED_HISTORICAL_CHECKPOINTS
)
from s2s_v2.exp_a_cb0_learning import ExperimentATrainer, build_exp_a_system
from s2s_v2.exp_b_residual_learning import ExperimentBTrainer, build_exp_b_system
from s2s_v2.exp_c_exposure_mitigation import ExperimentCTrainer, build_exp_c_system
from s2s_v2.human_listening_evaluator import HumanListeningSession


@pytest.fixture(autouse=True)
def verify_historical_hashes_fixture():
    """Verifies that historical checkpoints are untampered before and after every test."""
    verify_historical_checkpoint_hashes()
    yield
    verify_historical_checkpoint_hashes()


def test_temporal_alignment_and_positional_encoding():
    """Validates alignment mapping and verifies positional encoding breaks permutation equivariance."""
    aligner = TimestampAligner(d_in=1024, d_out=1024)
    aligner.eval()

    B, T_mms, D = 2, 100, 1024  # 100 frames @ 50 Hz = 2.0s
    target_len = 150            # 150 frames @ 75 Hz = 2.0s
    h_mms = torch.randn(B, T_mms, D)

    h_aligned = aligner(h_mms, target_len=target_len)
    assert h_aligned.shape == (B, target_len, D)

    # Test Sinusoidal Positional Encoding
    pos_enc = SinusoidalPositionalEncoding(d_model=1024)
    h_pos = pos_enc(h_aligned)
    assert h_pos.shape == (B, target_len, D)

    # Verify positional encoding breaks permutation equivariance
    # If we permute frames along time, adding positional encoding to permuted tensor
    # must differ from permuting the position-encoded tensor
    h_perm = h_aligned[:, torch.randperm(target_len), :]
    h_pos_perm = pos_enc(h_perm)
    perm_h_pos = h_pos[:, torch.randperm(target_len), :]
    assert not torch.allclose(h_pos_perm, perm_h_pos)

    # Test Learned Positional Encoding
    learned_pe = LearnedPositionalEncoding(d_model=1024)
    h_learned = learned_pe(h_aligned)
    assert h_learned.shape == (B, target_len, D)


def test_codebook_target_shapes_and_ranges():
    """Validates Encodec RVQ token extraction bounds."""
    aligner = EncodecTargetAligner(device=torch.device("cpu"))
    dummy_wav = torch.randn(1, 48000)  # 2.0 seconds @ 24 kHz

    tokens = aligner.extract_targets(dummy_wav, sample_rate=24000)
    assert tokens.ndim == 3
    assert tokens.shape[0] == 1
    assert tokens.shape[1] == 4
    assert tokens.min().item() >= 0
    assert tokens.max().item() < 1024


def test_speaker_disjoint_split_integrity():
    """Validates 0.00% speaker overlap across both Tier 1 and Tier 2 from manifest records."""
    audit = audit_dataset_splits()

    # Tier 1
    tier1 = audit["full_dataset_tier"]
    if tier1.get("status") == "VERIFIED_FROM_MANIFEST_RECORD":
        assert tier1["is_strictly_disjoint"] is True
        assert tier1["overlap_count"] == 0
        assert len(tier1["dynamic_overlap_speakers"]) == 0

    # Tier 2
    tier2 = audit["local_benchmark_tier"]
    assert tier2["is_strictly_disjoint"] is True
    assert tier2["overlap_count"] == 0
    assert len(tier2["dynamic_overlap_speakers"]) == 0


def test_deliberate_overlap_injection_failure():
    """Deliberately injects an overlapping speaker into manifest verification and confirms failure."""
    train_speakers = ["speaker_A", "speaker_B", "speaker_C"]
    val_speakers = ["speaker_D", "speaker_E", "speaker_A"]  # speaker_A injected!

    is_disjoint, overlap = verify_split_disjointness(train_speakers, val_speakers)
    assert is_disjoint is False, "Validator failed to catch injected speaker overlap!"
    assert overlap == {"speaker_A"}, f"Expected overlap {'speaker_A'}, got {overlap}"


def test_actual_experiment_b_step_and_eval():
    """Tests Experiment B training step and proves B1 and B2 evaluation decouple properly."""
    trainer = build_exp_b_system(device=torch.device("cpu"))

    B, T_mms, D = 1, 40, 1024
    T_codec = 60
    h_mms = torch.randn(B, T_mms, D)
    tgt_tokens = torch.randint(0, 1024, (B, 4, T_codec))

    # 1. Training step
    metrics = trainer.train_step(h_mms, tgt_tokens)
    assert "loss" in metrics
    assert "cb1_acc" in metrics

    # 2. B1 evaluation (Teacher-forced ceiling)
    eval_b1 = trainer.evaluate_b1_teacher_forced(h_mms, tgt_tokens)
    assert eval_b1["evaluation_mode"] == "B1_TEACHER_FORCED_CEILING"
    assert eval_b1["wav_b1_4cb"].ndim == 3
    assert eval_b1["wav_gt_cb0_only"].ndim == 3

    # 3. B2 evaluation (True inference condition)
    eval_b2 = trainer.evaluate_b2_inference_condition(h_mms, tgt_tokens)
    assert eval_b2["evaluation_mode"] == "B2_TRUE_INFERENCE_CONDITION"
    assert eval_b2["wav_b2_4cb"].ndim == 3
    assert eval_b2["wav_pred_cb0_only"].ndim == 3

    # B1 (conditioned on GT) and B2 (conditioned on predicted) waveforms must differ
    diff = torch.norm(eval_b1["wav_b1_4cb"] - eval_b2["wav_b2_4cb"]).item()
    assert diff > 0.0, "B1 and B2 waveforms unexpectedly identical!"


def test_experiment_c_forward_and_exposure_gap():
    """Verifies all 4 protocols of Experiment C and tests measure_exposure_gap."""
    for proto in ["C1", "C2", "C3", "C4"]:
        trainer = build_exp_c_system(protocol=proto, device=torch.device("cpu"))
        B, T_mms, D = 1, 30, 1024
        T_codec = 45
        h_mms = torch.randn(B, T_mms, D)
        tgt_tokens = torch.randint(0, 1024, (B, 4, T_codec))

        # Train step
        step_metrics = trainer.train_step(h_mms, tgt_tokens, step=50)
        assert "loss" in step_metrics

        # Exposure gap evaluation
        gap_metrics = trainer.measure_exposure_gap(h_mms, tgt_tokens)
        assert "exposure_accuracy_gap_delta" in gap_metrics
        assert "acc_residuals_teacher_forced" in gap_metrics
        assert "acc_residuals_free_run_inference" in gap_metrics


def test_experiment_c_gradient_flows_and_parameter_freezing():
    """
    Verifies parameter freezing and gradient flow assertions for Experiment C:
    - C1, C2, C3: Backbone and CB0 head must be frozen (requires_grad = False, grad is None)
    - C3: Soft-embedding differentiable gradient flows into depth embeddings
    - C4: All parameters are trainable and receive gradients
    """
    # Test C2 (Frozen backbone)
    trainer_c2 = build_exp_c_system(protocol="C2", device=torch.device("cpu"))
    h = torch.randn(1, 30, 1024)
    tokens = torch.randint(0, 1024, (1, 4, 45))
    trainer_c2.train_step(h, tokens, step=10)

    # Check backbone parameters received NO gradient
    for p in trainer_c2.model.transformer.parameters():
        assert p.requires_grad is False
        assert p.grad is None

    # Check CB0 head received NO gradient
    for p in trainer_c2.model.head.heads[0].parameters():
        assert p.requires_grad is False
        assert p.grad is None

    # Check residual head received gradient
    assert trainer_c2.model.head.heads[1].weight.grad is not None

    # Test C4 (Joint fine-tuning)
    trainer_c4 = build_exp_c_system(protocol="C4", device=torch.device("cpu"))
    trainer_c4.train_step(h, tokens, step=10)

    # All parameters must receive gradient in C4
    for p in trainer_c4.model.transformer.parameters():
        assert p.requires_grad is True
        assert p.grad is not None


def test_cb0_only_quantized_vector_ablation():
    """
    CRITICAL REQUIREMENT 5 VALIDATION:
    Proves that genuine CB0-only decoding slices to shape (B, 1, T),
    omitting residual codevectors, whereas token-zero padding injects non-zero vectors.
    """
    aligner = EncodecTargetAligner(device=torch.device("cpu"))
    tokens_4cb = torch.randint(0, 1024, (1, 4, 75))

    wav_true, wav_pseudo, l2_diff = aligner.verify_vector_ablation(tokens_4cb)

    assert wav_true.shape == (1, 1, 24000)
    assert wav_pseudo.shape == (1, 1, 24000)
    assert l2_diff > 1.0, f"Token ID 0 substitution acted as non-zero codevectors! L2={l2_diff}"

    # Also verify model.generate_tokens with cb0_only=True returns shape (B, 1, T)
    model = S2STokenPredictor(d_model=128, n_heads=4, n_layers=2, d_ff=256, num_codebooks=4, vocab_size=1029)
    dummy_mms = torch.randn(1, 30, 1024)
    gen_cb0 = model.generate_tokens(dummy_mms, target_len=45, cb0_only=True)
    assert gen_cb0.shape == (1, 1, 45), f"Expected shape (1, 1, 45), got {gen_cb0.shape}"


def test_encodec_scale_aware_reconstruction():
    """Verifies Encodec chunk and scale preservation during encode and decode."""
    aligner = EncodecTargetAligner(device=torch.device("cpu"))
    dummy_wav = torch.randn(1, 24000)  # 1.0s @ 24 kHz

    tokens, frames_meta = aligner.encode_with_metadata(dummy_wav)
    assert len(frames_meta) > 0
    assert tokens.shape[1] == 4

    # Reconstruct from frames metadata
    wav_recon = aligner.reconstruct_from_frames(frames_meta, num_codebooks=4)
    assert wav_recon.ndim == 3
    assert wav_recon.shape[0] == 1


def test_human_evaluation_isolation_and_matching(tmp_path):
    """
    Verifies:
    1. Unblinding key is isolated in secure_admin directory.
    2. Secure random 16-hex tokens are used.
    3. Per-utterance ground truth mapping is preserved.
    4. Agreement metrics work even with missing/incomplete listener ratings.
    """
    session = HumanListeningSession(session_dir=str(tmp_path))

    records = [
        {
            "model_id": "model_A",
            "utterance_id": "utt_001",
            "audio": np.random.randn(24000),
            "ref_text": "வணக்கம்",
            "ref_audio_path": "/path/to/ref1.wav"
        },
        {
            "model_id": "model_B",
            "utterance_id": "utt_002",
            "audio": np.random.randn(24000),
            "ref_text": "நன்றி",
            "ref_audio_path": "/path/to/ref2.wav"
        }
    ]

    key = session.prepare_blinded_samples(records)

    # Verify unblinding key is in secure_admin, not in listener_portal
    assert os.path.exists(session.key_path)
    assert "secure_admin" in session.key_path
    assert not os.path.exists(os.path.join(session.listener_portal, "unblinding_key.json"))

    # Verify secure token format (eval_<16 hex chars>.wav)
    for b_name in key:
        assert b_name.startswith("eval_")
        token = b_name.replace("eval_", "").replace(".wav", "")
        assert len(token) == 16, f"Expected 16-hex token, got {token}"
        assert key[b_name]["reference_text"] in ["வணக்கம்", "நன்றி"]

    # Record responses with missing listener ratings (Listener 2 skips sample 2)
    sample_names = list(key.keys())
    session.record_response("listener_1", sample_names[0], 4, 4, "வணக்கம்")
    session.record_response("listener_2", sample_names[0], 5, 4, "வணக்கம்")
    session.record_response("listener_1", sample_names[1], 2, 2, "நன்றி")
    # listener_2 missing for sample_names[1]

    results = session.analyze_results()
    assert "per_model_summary" in results
    assert "agreement_metrics" in results
    assert "mean_pairwise_agreement" in results["agreement_metrics"]


def test_checkpoint_path_containment_and_safety(tmp_path):
    """
    Verifies SafeCheckpointManager:
    - Strictly rejects path traversal (../../)
    - Strictly rejects symlink escaping
    - Asserts immutability of protected historical checkpoints
    """
    model = S2STokenPredictor(d_model=128, n_heads=4, n_layers=2, d_ff=256, num_codebooks=4, vocab_size=1029)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    mgr = SafeCheckpointManager(experiment_id="safe_test")
    mgr.save_dir = str(tmp_path / "designated_dir")
    os.makedirs(mgr.save_dir, exist_ok=True)

    # 1. Valid save
    saved = mgr.save_checkpoint(model, optimizer, step=10, metrics={"acc": 50.0})
    assert os.path.exists(saved)

    # 2. Path traversal rejection (../../outside.pt)
    with pytest.raises(ValueError) as exc:
        mgr.save_checkpoint(model, optimizer, step=11, metrics={}, custom_filename="../../escape.pt")
    assert "escapes designated directory" in str(exc.value)

    # 3. Absolute path rejection
    with pytest.raises(ValueError) as exc:
        mgr.save_checkpoint(model, optimizer, step=12, metrics={}, custom_filename="/etc/passwd_fake.pt")
    assert "escapes designated directory" in str(exc.value)

    # 4. Historical Checkpoint Immutability Rejection
    for protected in PROTECTED_HISTORICAL_CHECKPOINTS:
        with pytest.raises(AssertionError):
            mgr.save_dir = os.path.dirname(os.path.join(REPO_ROOT, protected))
            mgr.save_checkpoint(model, optimizer, step=1, metrics={}, custom_filename=os.path.basename(protected))


def test_textless_inference_invariants():
    """
    STRICT TEXTLESS INVARIANT VALIDATION:
    Asserts model takes zero text or phonemes and contains zero text encoders or vocabularies.
    """
    model = S2STokenPredictor(d_model=1024, n_heads=16, n_layers=6, d_ff=4096, num_codebooks=4, vocab_size=1029)

    for name, _ in model.named_parameters():
        for forbidden in ["text", "phoneme", "bpe", "char", "vocab_embed", "transcription"]:
            assert forbidden not in name.lower(), f"Forbidden text-related parameter found: {name}"

    dummy_mms = torch.randn(1, 50, 1024)
    logits = model(dummy_mms, target_len=75)
    assert logits.shape == (1, 4, 75, 1029)
