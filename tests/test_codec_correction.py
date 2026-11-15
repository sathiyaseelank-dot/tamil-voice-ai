"""
Regression Tests for Encodec 24 kHz Codec Correction.
Verifies tensor shapes, codebook count, token range, sample rate, and reconstruction fidelity.
"""

import os
import sys
import pytest
import torch
import numpy as np
import soundfile as sf

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from config import AudioConfig
from tokenizer.codec import AudioCodecTokenizer


def test_codec_configuration():
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    tok = AudioCodecTokenizer(config=audio_cfg, device="cpu")
    assert tok.config.sample_rate == 24000
    assert tok.config.frame_rate == 75
    assert tok.config.num_codebooks == 4
    assert tok.config.codebook_size == 1024
    assert tok.model.bandwidth == 3.0


def test_tensor_shapes_and_codebook_count():
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    tok = AudioCodecTokenizer(config=audio_cfg, device="cpu")

    # 1.0 second of dummy audio (24,000 samples)
    dummy_wav = torch.randn(1, 1, 24000)
    codes = tok.encode(dummy_wav)
    assert codes.shape == (1, 4, 75), f"Expected (1, 4, 75), got {codes.shape}"

    # Decode
    recon = tok.decode(codes)
    assert recon.shape == (1, 1, 24000), f"Expected (1, 1, 24000), got {recon.shape}"


def test_token_range_and_special_token_handling():
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    tok = AudioCodecTokenizer(config=audio_cfg, device="cpu")

    # Tokens with valid codes and special tokens (pad=1024, bos=1025, silence=1027)
    tokens = torch.randint(0, 1024, (1, 4, 150))
    # Insert special tokens
    tokens[0, :, 50] = 1024
    tokens[0, :, 51] = 1027

    recon = tok.decode(tokens)
    assert not torch.isnan(recon).any()
    assert not torch.isinf(recon).any()
    # Frame 50 and 51 should be muted to 0.0
    hop = 24000 // 75
    assert (recon[0, 0, 50 * hop:(50 + 1) * hop] == 0.0).all()
    assert (recon[0, 0, 51 * hop:(51 + 1) * hop] == 0.0).all()


def test_reconstruction_fidelity_on_real_tamil_speech():
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    tok = AudioCodecTokenizer(config=audio_cfg, device="cpu")

    real_wav_path = os.path.join(repo_root, "experiments", "v2_p1", "wavs", "sample_03_original.wav")
    assert os.path.exists(real_wav_path)
    raw_aud, sr = sf.read(real_wav_path)
    assert sr == 24000

    w = torch.from_numpy(raw_aud).float().unsqueeze(0).unsqueeze(0)
    codes = tok.encode(w)
    recon = tok.decode(codes).squeeze().numpy()

    min_l = min(len(raw_aud), len(recon))
    sig_power = np.sum(raw_aud[:min_l] ** 2)
    noise_power = np.sum((raw_aud[:min_l] - recon[:min_l]) ** 2)
    snr = 10 * np.log10(sig_power / (noise_power + 1e-9))

    print(f"Corrected Codec Reconstruction SNR: {snr:.2f} dB")
    # Must achieve clean reconstruction (> 7.0 dB)
    assert snr > 7.0, f"Reconstruction SNR too low: {snr:.2f} dB"


def test_no_32_codebook_zero_padding_distortion():
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    tok = AudioCodecTokenizer(config=audio_cfg, device="cpu")

    # Compare decode with raw model.decode([(codes, None)])
    dummy_codes = torch.randint(0, 1024, (1, 4, 75))
    recon_tok = tok.decode(dummy_codes)
    with torch.no_grad():
        recon_raw = tok.model.decode([(dummy_codes, None)])

    diff = (recon_tok - recon_raw).abs().max().item()
    assert diff == 0.0, f"Discrepancy between tok.decode and raw model.decode: {diff}"


if __name__ == "__main__":
    test_codec_configuration()
    test_tensor_shapes_and_codebook_count()
    test_token_range_and_special_token_handling()
    test_reconstruction_fidelity_on_real_tamil_speech()
    test_no_32_codebook_zero_padding_distortion()
    print("✅ All Codec Correction Regression Tests Passed!")
