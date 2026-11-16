"""
Unit tests for TextlessTamilSpeechLM_CB0 (P2.5-A Architecture).
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, '/content/tamil-voice-ai')
from config import ModelConfig, AudioConfig
from model.audio_transformer_cb0 import TextlessTamilSpeechLM_CB0
from tokenizer.codec import AudioCodecTokenizer


def test_forward_pass_and_loss():
    cfg = ModelConfig()
    model = TextlessTamilSpeechLM_CB0(cfg)
    model.eval()

    batch_size = 2
    seq_len = 50
    x_in = torch.randint(0, 1024, (batch_size, seq_len))
    x_tgt = torch.randint(0, 1024, (batch_size, seq_len))

    logits, loss, metrics = model(x_in, target_tokens=x_tgt)

    assert logits.shape == (batch_size, seq_len, 1029), f"Unexpected logits shape: {logits.shape}"
    assert loss is not None and loss.item() > 0, "Loss should be a positive scalar."
    assert "cb0_accuracy" in metrics, "Missing cb0_accuracy in metrics."
    assert "cb0_perplexity" in metrics, "Missing cb0_perplexity in metrics."
    print("✓ Forward pass and loss verified.")


def test_backward_gradient_flow():
    cfg = ModelConfig()
    model = TextlessTamilSpeechLM_CB0(cfg)
    model.train()

    x_in = torch.randint(0, 1024, (1, 30))
    x_tgt = torch.randint(0, 1024, (1, 30))

    logits, loss, _ = model(x_in, target_tokens=x_tgt)
    loss.backward()

    assert model.embedding.weight.grad is not None, "Embedding grad is None."
    assert model.transformer.layers[0].linear1.weight.grad is not None, "Trunk grad is None."
    assert model.head.weight.grad is not None, "Head grad is None."
    print("✓ Gradient flow verified.")


def test_grafting_from_v2_p2_4():
    ckpt_path = '/content/tamil-voice-ai/checkpoints_v2_p2_4/tamil_speech_lm_v2_p2_4_best.pt'
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    model = TextlessTamilSpeechLM_CB0.from_pretrained_multicodebook(ckpt_path)
    assert model is not None
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Grafting succeeded. CB0 Model parameter count: {total_params/1e6:.2f}M")


def test_autoregressive_generation_and_codec_decode():
    cfg = ModelConfig()
    model = TextlessTamilSpeechLM_CB0(cfg)
    model.eval()

    prompt = torch.randint(0, 1024, (1, 1, 20))
    gen_tokens = model.generate(prompt_tokens=prompt, max_new_tokens=15, temperature=0.8)

    assert gen_tokens.shape == (1, 1, 15), f"Unexpected gen_tokens shape: {gen_tokens.shape}"

    # Verify native 1-codebook decode with Encodec
    tokenizer = AudioCodecTokenizer(device='cpu')
    with torch.no_grad():
        emb = tokenizer.model.quantizer.decode(gen_tokens)
        wav = tokenizer.model.decoder(emb)

    assert wav.shape[1] == 1, f"Unexpected wav shape: {wav.shape}"
    assert wav.shape[-1] == 15 * 320, f"Expected {15*320} samples, got {wav.shape[-1]}"
    print("✓ Autoregressive generation and 1-codebook Encodec decoding verified.")


if __name__ == '__main__':
    test_forward_pass_and_loss()
    test_backward_gradient_flow()
    test_grafting_from_v2_p2_4()
    test_autoregressive_generation_and_codec_decode()
    print("\nALL CB0 ARCHITECTURE TESTS PASSED.")
