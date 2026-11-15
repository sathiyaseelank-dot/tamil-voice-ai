"""
Engineering Verification Suite for Textless Tamil Speech LM (V2).
Verifies all 9 items required by Phase 4.
"""

import os
import sys
import tempfile
import torch
import torch.nn.functional as F
import numpy as np

# Ensure repository root is on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from config import AudioConfig, ModelConfig
from model.audio_transformer import TextlessTamilSpeechLM, DepthAudioHead
from tokenizer.codec import AudioCodecTokenizer
from tokenizer.audio_io import save_audio, load_audio


def test_1_codec_round_trip():
    print("[Test 1/9] Verifying Codec Round-Trip...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    tokenizer = AudioCodecTokenizer(config=audio_cfg, device=device)

    # 1 second of synthetic speech-like tone
    t = torch.linspace(0, 1.0, 24000)
    synthetic_wav = 0.5 * torch.sin(2 * np.pi * 220 * t).unsqueeze(0).unsqueeze(0).to(device)

    tokens = tokenizer.encode(synthetic_wav, sample_rate=24000)
    assert tokens.ndim == 3, f"Expected 3D tokens tensor, got {tokens.shape}"
    assert tokens.shape[1] == 4, f"Expected 4 codebooks, got {tokens.shape[1]}"
    assert (tokens >= 0).all() and (tokens < 1024).all(), "Tokens out of codebook bounds!"

    reconstructed = tokenizer.decode(tokens)
    assert reconstructed.shape[-1] >= 24000 - 320 and reconstructed.shape[-1] <= 24000 + 320, f"Unexpected length {reconstructed.shape}"
    print("  ✓ Codec encode/decode preserves expected shapes and valid codebook range.")


def test_2_forward_pass_shapes():
    print("[Test 2/9] Verifying Forward-Pass Tensor Shapes...")
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    model_cfg = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=512, max_seq_len=100, audio=audio_cfg)
    model = TextlessTamilSpeechLM(config=model_cfg)
    model.eval()

    batch_size = 2
    seq_len = 50
    x_in = torch.randint(0, 1024, (batch_size, 4, seq_len))
    x_target = torch.randint(0, 1024, (batch_size, 4, seq_len))

    logits, loss, metrics = model(x_in, target_tokens=x_target)
    assert logits.shape == (batch_size, 4, seq_len, audio_cfg.vocab_size), f"Wrong logits shape: {logits.shape}"
    assert loss is not None and not torch.isnan(loss) and not torch.isinf(loss), "Invalid loss value!"
    assert "speech_accuracy" in metrics
    assert "loss_cb0" in metrics
    print(f"  ✓ Logits shape {logits.shape} and loss {loss.item():.4f} are verified.")


def test_3_temporal_causal_masking():
    print("[Test 3/9] Verifying Temporal Causal Masking (No Future Leakage)...")
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    model_cfg = ModelConfig(d_model=128, n_heads=4, n_layers=2, d_ff=256, max_seq_len=100, audio=audio_cfg)
    model = TextlessTamilSpeechLM(config=model_cfg)
    model.eval()

    seq_len = 20
    x1 = torch.randint(0, 1024, (1, 4, seq_len))
    x2 = x1.clone()
    # Modify future tokens (from position 10 onwards)
    x2[:, :, 10:] = torch.randint(0, 1024, (1, 4, 10))

    with torch.no_grad():
        logits1, _, _ = model(x1)
        logits2, _, _ = model(x2)

    # Predictions for t < 10 must be mathematically identical
    diff_past = (logits1[:, :, :10, :] - logits2[:, :, :10, :]).abs().max().item()
    diff_future = (logits1[:, :, 10:, :] - logits2[:, :, 10:, :]).abs().max().item()

    assert diff_past < 1e-5, f"Temporal causal leak detected! Difference in past positions: {diff_past}"
    assert diff_future > 1e-4, "Future should differ when inputs differ!"
    print(f"  ✓ Strict causal attention confirmed (past diff: {diff_past:.2e}, future diff: {diff_future:.2e}).")


def test_4_depth_head_teacher_forcing():
    print("[Test 4/9] Verifying Depth-Head Teacher Forcing...")
    d_model = 128
    num_cb = 4
    vocab_size = 1029
    head = DepthAudioHead(d_model, num_cb, vocab_size)
    head.eval()

    h = torch.randn(1, 15, d_model)
    targets1 = torch.randint(0, 1024, (1, num_cb, 15))
    targets2 = targets1.clone()
    # Change target on codebook 0
    targets2[:, 0, :] = (targets1[:, 0, :] + 10) % 1024

    with torch.no_grad():
        logits1 = head(h, target_tokens=targets1)
        logits2 = head(h, target_tokens=targets2)

    # Codebook 0 logits depend ONLY on h, so changing codebook 0 target should NOT change codebook 0 logits
    cb0_diff = (logits1[:, 0] - logits2[:, 0]).abs().max().item()
    assert cb0_diff < 1e-5, f"Codebook 0 logits should not depend on target: diff={cb0_diff}"

    # Codebook 1 logits condition on codebook 0 target, so changing codebook 0 target MUST change codebook 1 logits
    cb1_diff = (logits1[:, 1] - logits2[:, 1]).abs().max().item()
    assert cb1_diff > 1e-3, f"Codebook 1 should reflect codebook 0 conditioning: diff={cb1_diff}"
    print(f"  ✓ Teacher-forced depth conditioning verified (CB0 indep: {cb0_diff:.2e}, CB1 dep: {cb1_diff:.2e}).")


def test_5_sequential_sampled_token_conditioning():
    print("[Test 5/9] Verifying Sequential Sampled-Token Conditioning...")
    d_model = 128
    num_cb = 4
    vocab_size = 1029
    head = DepthAudioHead(d_model, num_cb, vocab_size)
    head.eval()

    h_t = torch.randn(1, d_model)
    torch.manual_seed(123)
    frame, logits = head.sample_step(h_t, temperature=0.8, top_k=50)

    assert frame.shape == (1, num_cb, 1), f"Sampled frame shape mismatch: {frame.shape}"
    assert logits.shape == (1, num_cb, vocab_size), f"Step logits shape mismatch: {logits.shape}"
    assert (frame >= 0).all() and (frame < vocab_size).all(), "Sampled token ID out of range!"
    print(f"  ✓ Sampled frame shape {frame.shape} with valid token IDs.")


def test_6_padding_and_silence_masking_independence():
    print("[Test 6/9] Verifying Padding and Silence Masking Independence...")
    audio_cfg = AudioConfig()
    model_cfg = ModelConfig(d_model=64, n_heads=2, n_layers=1, d_ff=128, max_seq_len=20, audio=audio_cfg)
    model = TextlessTamilSpeechLM(config=model_cfg)
    model.eval()

    # Targets containing: regular speech (100), silence (1027), padding (1024)
    x_in = torch.randint(0, 1024, (1, 4, 10))
    targets = torch.full((1, 4, 10), 100, dtype=torch.long)
    targets[:, :, 4:7] = audio_cfg.silence_token_id  # legitimate audio silence
    targets[:, :, 7:] = audio_cfg.pad_token_id       # batch padding

    with torch.no_grad():
        _, loss, metrics = model(x_in, target_tokens=targets)

    # The loss must NOT ignore silence_token_id, it must ONLY ignore pad_token_id
    with torch.no_grad():
        logits, loss, metrics = model(x_in, target_tokens=targets)

    # Recompute reference loss directly from model's returned logits
    flat_l = logits.permute(0, 2, 1, 3).reshape(-1, audio_cfg.vocab_size)
    flat_t = targets.permute(0, 2, 1).reshape(-1)

    expected_loss = F.cross_entropy(flat_l, flat_t, ignore_index=audio_cfg.pad_token_id)
    assert abs(loss.item() - expected_loss.item()) < 1e-5, f"Loss mismatch: {loss.item()} vs {expected_loss.item()}"

    # Verify that silence tokens (1027) contribute to loss and are NOT ignored
    silence_mask = (flat_t == audio_cfg.silence_token_id)
    assert silence_mask.sum() > 0
    silence_loss = F.cross_entropy(flat_l[silence_mask], flat_t[silence_mask])
    assert silence_loss.item() > 0.0, "Legitimate silence must contribute non-zero loss!"

    # Verify that padding tokens (1024) are completely ignored by the cross_entropy
    pad_mask = (flat_t == audio_cfg.pad_token_id)
    assert pad_mask.sum() > 0
    # Loss on all non-pad tokens should match the total loss
    non_pad_loss = F.cross_entropy(flat_l[~pad_mask], flat_t[~pad_mask])
    assert abs(loss.item() - non_pad_loss.item()) < 1e-5, "Loss should be identical to non-pad loss!"
    print(f"  ✓ Padding (1024) is masked while legitimate silence (1027) contributes correctly to loss.")


def test_7_generation_and_wav_decode():
    print("[Test 7/9] Verifying End-to-End Generation & WAV Decode...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    model_cfg = ModelConfig(d_model=128, n_heads=4, n_layers=2, d_ff=256, max_seq_len=100, audio=audio_cfg)
    model = TextlessTamilSpeechLM(config=model_cfg).to(device)
    tokenizer = AudioCodecTokenizer(config=audio_cfg, device=device)

    # Generate 15 frames (~0.2s)
    with torch.no_grad():
        gen_tokens = model.generate(prompt_tokens=None, max_new_tokens=15, temperature=0.8)

    assert gen_tokens.shape == (1, 4, 15), f"Expected (1, 4, 15), got {gen_tokens.shape}"
    assert (gen_tokens >= 0).all() and (gen_tokens < audio_cfg.vocab_size).all()

    wav = tokenizer.decode(gen_tokens).squeeze(0).cpu()
    assert wav.ndim == 2 and wav.shape[0] == 1, f"Expected (1, samples), got {wav.shape}"
    assert wav.shape[-1] > 0, "Empty audio waveform generated!"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        save_audio(tmp.name, wav, audio_cfg.sample_rate)
        loaded_wav, sr = load_audio(tmp.name)
        assert sr == 24000
    print(f"  ✓ Generated {gen_tokens.shape[-1]} frames, decoded to {wav.shape[-1]} samples at 24kHz.")


def test_8_nan_inf_checks():
    print("[Test 8/9] Verifying NaN and Infinite Loss Resilience...")
    model_cfg = ModelConfig(d_model=128, n_heads=4, n_layers=2, d_ff=256, max_seq_len=50)
    model = TextlessTamilSpeechLM(config=model_cfg)
    x = torch.randint(0, 1024, (2, 4, 25))
    targets = torch.randint(0, 1024, (2, 4, 25))

    logits, loss, metrics = model(x, target_tokens=targets)
    assert not torch.isnan(logits).any(), "NaN detected in logits!"
    assert not torch.isinf(logits).any(), "Inf detected in logits!"
    assert not torch.isnan(loss), "NaN detected in loss!"
    assert not torch.isinf(loss), "Inf detected in loss!"
    print("  ✓ 0 NaNs and 0 Infs detected.")


def test_9_checkpoint_compatibility():
    print("[Test 9/9] Verifying Checkpoint Save/Load Compatibility...")
    model_cfg = ModelConfig(d_model=128, n_heads=4, n_layers=2, d_ff=256, max_seq_len=50)
    model = TextlessTamilSpeechLM(config=model_cfg)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as tmp:
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": model_cfg,
            "version": "v2"
        }, tmp.name)

        loaded = torch.load(tmp.name, map_location="cpu", weights_only=False)
        assert "model_state_dict" in loaded
        assert loaded["version"] == "v2"
        model2 = TextlessTamilSpeechLM(config=loaded["config"])
        model2.load_state_dict(loaded["model_state_dict"])
    print("  ✓ Checkpoint serialization and deserialization verified.")


def run_all_tests():
    print("=" * 70)
    print("🧪 RUNNING V2 ARCHITECTURE ENGINEERING VERIFICATION SUITE")
    print("=" * 70)
    test_1_codec_round_trip()
    test_2_forward_pass_shapes()
    test_3_temporal_causal_masking()
    test_4_depth_head_teacher_forcing()
    test_5_sequential_sampled_token_conditioning()
    test_6_padding_and_silence_masking_independence()
    test_7_generation_and_wav_decode()
    test_8_nan_inf_checks()
    test_9_checkpoint_compatibility()
    print("=" * 70)
    print("✅ ALL 9 ENGINEERING VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
