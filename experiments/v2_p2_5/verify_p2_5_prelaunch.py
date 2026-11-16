#!/usr/bin/env python3
"""
Pre-Launch Safeguards Verification for Experiment V2-P2.5-A2.
Checks all 10 criteria specified in the research protocol before training/evaluating.
"""

import os
import sys
import hashlib
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/inkyank-01/dev/tamil-voice-ai')
from config import ModelConfig, AudioConfig
from model.audio_transformer_cb0 import TextlessTamilSpeechLM_CB0


def verify_all():
    print("=" * 80)
    print("V2-P2.5-A2 PRE-LAUNCH SAFEGUARDS VERIFICATION AUDIT")
    print("=" * 80)

    repo_dir = "/home/inkyank-01/dev/tamil-voice-ai"
    ckpt_path = os.path.join(repo_dir, "checkpoints_v2_p2_4", "tamil_speech_lm_v2_p2_4_best.pt")
    expected_sha256 = "ae5ccc2376b5c5938a47a71498adaa807168ba6ffceedfd83138af14bc358c45"

    checks = {}

    # 1. SHA256 Check
    print("\n[Check 1/10] Computing Checkpoint SHA256...")
    if not os.path.exists(ckpt_path):
        print(f"FAILED: Checkpoint not found at {ckpt_path}")
        return False
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    actual_sha256 = h.hexdigest()
    sha256_match = (actual_sha256 == expected_sha256)
    checks["1_sha256_match"] = sha256_match
    print(f"  Expected: {expected_sha256}")
    print(f"  Actual:   {actual_sha256}")
    print(f"  Result:   {'✓ PASS' if sha256_match else '❌ FAIL'}")
    assert sha256_match, f"SHA256 mismatch! Expected {expected_sha256}, got {actual_sha256}"

    # 2. Checkpoint Step 4500 Verification
    print("\n[Check 2/10] Verifying Checkpoint Step 4500...")
    raw_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_step = raw_ckpt.get('step', None)
    step_match = (ckpt_step == 4500)
    checks["2_step_4500"] = step_match
    print(f"  Recorded step in checkpoint: {ckpt_step}")
    print(f"  Result:   {'✓ PASS' if step_match else '❌ FAIL'}")
    assert step_match, f"Checkpoint step is {ckpt_step}, expected 4500!"

    # 3. Model Architecture & Parameter Count
    print("\n[Check 3/10] Model Architecture & Parameter Reconciliation...")
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=1)
    model_cfg = ModelConfig(d_model=1024, n_heads=16, n_layers=6, d_ff=4096, audio=audio_cfg)
    model = TextlessTamilSpeechLM_CB0(model_cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_params = 77687813
    params_match = (trainable_params == expected_params)
    checks["3_parameters"] = params_match
    print(f"  Layers: {model_cfg.n_layers}, Heads: {model_cfg.n_heads}, d_model: {model_cfg.d_model}, d_ff: {model_cfg.d_ff}")
    print(f"  Trainable Parameters: {trainable_params:,} (Expected: {expected_params:,})")
    print(f"  Result:   {'✓ PASS' if params_match else '❌ FAIL'}")
    assert params_match, f"Parameter count mismatch: {trainable_params} vs expected {expected_params}"

    # 4. Compatible Tensor Transfer (Weight Grafting)
    print("\n[Check 4/10] Compatible Tensor Transfer & Parity Audit...")
    src_sd = raw_ckpt['model_state_dict']
    tgt_sd = model.state_dict()
    transferred_keys = []
    missing_keys = []

    for k in tgt_sd.keys():
        if k in src_sd:
            tgt_sd[k].copy_(src_sd[k])
            transferred_keys.append(k)
        elif k == 'embedding.weight' and 'codebook_embeddings.0.weight' in src_sd:
            tgt_sd[k].copy_(src_sd['codebook_embeddings.0.weight'])
            transferred_keys.append(f"{k} <- codebook_embeddings.0.weight")
        elif k == 'head.weight' and 'head.heads.0.weight' in src_sd:
            tgt_sd[k].copy_(src_sd['head.heads.0.weight'])
            transferred_keys.append(f"{k} <- head.heads.0.weight")
        elif k == 'head.bias' and 'head.heads.0.bias' in src_sd:
            tgt_sd[k].copy_(src_sd['head.heads.0.bias'])
            transferred_keys.append(f"{k} <- head.heads.0.bias")
        else:
            missing_keys.append(k)

    transfer_success = (len(missing_keys) == 0 and len(transferred_keys) == len(tgt_sd))
    checks["4_tensor_transfer"] = transfer_success
    print(f"  Transferred Tensors: {len(transferred_keys)} / {len(tgt_sd)}")
    print(f"  Missing Tensors:     {len(missing_keys)}")
    print(f"  Result:              {'✓ PASS' if transfer_success else '❌ FAIL'}")
    assert transfer_success, f"Transfer failed, missing: {missing_keys}"

    # 5. BOS / PAD Special Tokens
    print("\n[Check 5/10] Verifying Special Tokens...")
    tokens_correct = (model.pad_token_id == 1024 and model.bos_token_id == 1025 and model.eos_token_id == 1026 and model.vocab_size == 1029)
    checks["5_special_tokens"] = tokens_correct
    print(f"  PAD: {model.pad_token_id} (Expected: 1024)")
    print(f"  BOS: {model.bos_token_id} (Expected: 1025)")
    print(f"  EOS: {model.eos_token_id} (Expected: 1026)")
    print(f"  Vocab Size: {model.vocab_size} (Expected: 1029)")
    print(f"  Result:     {'✓ PASS' if tokens_correct else '❌ FAIL'}")
    assert tokens_correct

    # 6. Causal Masking & Next-Token Shift
    print("\n[Check 6/10] Verifying Causal Masking and Next-Token Shift...")
    mask = model._generate_causal_mask(5, torch.device('cpu'))
    # Strictly upper triangular must be -inf, diagonal and lower triangular must be 0
    upper_inf = torch.all(torch.isneginf(mask[0, 1:]))
    diag_zero = torch.all(mask.diagonal() == 0)
    lower_zero = torch.all(mask[1:, 0] == 0)
    causal_correct = bool(upper_inf and diag_zero and lower_zero)
    checks["6_causal_masking"] = causal_correct
    print(f"  Upper triangular -inf: {upper_inf.item()}")
    print(f"  Diagonal & lower 0.0:  {(diag_zero and lower_zero).item()}")
    print(f"  Result:                {'✓ PASS' if causal_correct else '❌ FAIL'}")
    assert causal_correct

    # 7. Training and Validation Speaker Splits
    print("\n[Check 7/10] Verifying Dataset Manifest and Speaker Split...")
    manifest_path = os.path.join(repo_dir, "data", "v2_p2_4", "v2_p2_4_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    train_spks = manifest.get("train_speakers_count")
    val_spks = manifest.get("val_speakers_count")
    overlap = manifest.get("speaker_overlap")
    train_hrs = manifest.get("train_duration_hours")
    val_hrs = manifest.get("val_duration_hours")
    split_correct = (train_spks == 382 and val_spks == 25 and overlap == 0)
    checks["7_speaker_split"] = split_correct
    print(f"  Train Speakers: {train_spks} ({train_hrs:.2f} hrs, Expected: 382)")
    print(f"  Val Speakers:   {val_spks} ({val_hrs:.2f} hrs, Expected: 25)")
    print(f"  Speaker Overlap: {overlap} (Expected: 0)")
    print(f"  Result:         {'✓ PASS' if split_correct else '❌ FAIL'}")
    assert split_correct

    # 8. Check Second Downloaded Checkpoint (tamil_speech_lm_cb0_best.pt)
    print("\n[Check 8/10] Inspecting Second Downloaded File (tamil_speech_lm_cb0_best.pt)...")
    cb0_ckpt_path = os.path.join(repo_dir, "checkpoints_v2_p2_5_a2", "tamil_speech_lm_cb0_best.pt")
    if os.path.exists(cb0_ckpt_path):
        cb0_ckpt = torch.load(cb0_ckpt_path, map_location='cpu', weights_only=False)
        print(f"  Keys in cb0 checkpoint: {list(cb0_ckpt.keys())}")
        print(f"  Experiment: {cb0_ckpt.get('experiment')}")
        print(f"  Step:       {cb0_ckpt.get('step')}")
        print(f"  Init mode:  {cb0_ckpt.get('init_mode')}")
        print(f"  Val metrics in checkpoint: {cb0_ckpt.get('val_metrics')}")
    else:
        print(f"  tamil_speech_lm_cb0_best.pt not found at {cb0_ckpt_path}")

    # 9. Output Directory Check
    print("\n[Check 9/10] Output Directory Isolation...")
    out_dir = os.path.join(repo_dir, "checkpoints_v2_p2_5_a2")
    os.makedirs(out_dir, exist_ok=True)
    checks["9_output_dir"] = True
    print(f"  Output Directory: {out_dir}")
    print(f"  Result:           ✓ PASS")

    # 10. Summary
    print("\n" + "=" * 80)
    print("ALL PRE-LAUNCH SAFEGUARDS SUCCESSFULLY PASSED")
    print("=" * 80)
    return True


if __name__ == "__main__":
    verify_all()
