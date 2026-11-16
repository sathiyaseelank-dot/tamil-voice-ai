#!/usr/bin/env python3
"""
Simple Audio Codec Encoder-Decoder Utility.
Takes an input audio file, encodes it into RVQ discrete tokens with Encodec 24kHz,
and decodes the tokens back into a reconstructed waveform.

Usage:
    python encode_decode.py --input samples/user_turn_0000.wav --output samples/decoded_output.wav
"""

import os
import sys
import argparse
import numpy as np
import torch
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from config import AudioConfig
from tokenizer.codec import AudioCodecTokenizer


def main():
    parser = argparse.ArgumentParser(description="Encode and Decode an audio file using Neural Audio Codec")
    parser.add_argument("--input", "-i", type=str, default="samples/user_turn_0000.wav",
                        help="Path to input WAV audio file")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to save decoded output WAV file")
    parser.add_argument("--codebooks", "-k", type=int, default=4, choices=[1, 2, 4, 8],
                        help="Number of codebooks to decode (default: 4)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    assert os.path.exists(args.input), f"Input file not found: {args.input}"

    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"samples/codec_roundtrip/{base_name}_decoded_{args.codebooks}cb.wav"

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print("=" * 70)
    print("🎧 NEURAL AUDIO CODEC: ENCODER -> TOKENS -> DECODER")
    print("=" * 70)
    print(f"Input File:       {args.input}")
    print(f"Output File:      {args.output}")
    print(f"Target Codebooks: {args.codebooks}")

    # 1. Read input audio
    y, sr = sf.read(args.input)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    duration_in = len(y) / float(sr)
    rms_in = float(np.sqrt(np.mean(y ** 2)))
    print(f"\n[1/3] Loaded Input Audio:")
    print(f"  Duration:    {duration_in:.2f} seconds")
    print(f"  Sample Rate: {sr} Hz")
    print(f"  RMS Energy:  {rms_in:.4f}")

    # 2. Encode
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    tokenizer = AudioCodecTokenizer(audio_cfg, device=args.device)

    w_t = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        tokens = tokenizer.encode(w_t, sr)  # (1, 4, T)

    print(f"\n[2/3] Encoded to Discrete RVQ Tokens:")
    print(f"  Tokens Tensor Shape: {tuple(tokens.shape)} [Batch, Codebooks, Frames]")
    print(f"  Total Audio Frames:  {tokens.shape[-1]} frames ({tokens.shape[-1]/75.0:.2f} seconds @ 75 Hz)")
    print(f"  Token ID Range:      min={tokens.min().item()}, max={tokens.max().item()} (Vocab: 1024)")

    # Print first 5 frames
    print("  First 5 frames:")
    for k in range(min(args.codebooks, tokens.shape[1])):
        print(f"    CB{k}: {tokens[0, k, :5].tolist()}")

    # 3. Decode
    with torch.no_grad():
        if args.codebooks == 1:
            emb = tokenizer.model.quantizer.decode(tokens[:, :1, :])
            wav_out = tokenizer.model.decoder(emb).squeeze().cpu().numpy()
        else:
            wav_out = tokenizer.decode(tokens[:, :args.codebooks, :]).squeeze().cpu().numpy()

    sf.write(args.output, wav_out, 24000)
    duration_out = len(wav_out) / 24000.0
    rms_out = float(np.sqrt(np.mean(wav_out ** 2)))

    # Compute Signal-to-Noise Ratio (SNR)
    min_len = min(len(y), len(wav_out))
    noise = y[:min_len] - wav_out[:min_len]
    snr = 10 * np.log10(np.sum(y[:min_len]**2) / (np.sum(noise**2) + 1e-10))

    print(f"\n[3/3] Decoded Waveform Generated:")
    print(f"  Saved To:    {args.output}")
    print(f"  Duration:    {duration_out:.2f} seconds")
    print(f"  Sample Rate: 24,000 Hz")
    print(f"  RMS Energy:  {rms_out:.4f}")
    print(f"  Codec SNR:   {snr:.2f} dB")
    print("\n✓ SUCCESS: Roundtrip audio encode and decode complete.")


if __name__ == "__main__":
    main()
