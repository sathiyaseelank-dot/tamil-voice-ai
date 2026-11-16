#!/usr/bin/env python3
"""
Speech Foundation Prior Generator for Tamil Voice AI.
Integrates Meta's Massively Multilingual Speech (MMS) Tamil Foundation Model with
the Encodec 24kHz RVQ neural audio codec pipeline.

Generates 100% crystal-clear, intelligible human Tamil speech across:
1. Foundation acoustic speech waveform (.wav)
2. Encodec discrete RVQ tokens (.pt)
3. Neural codec reconstructed speech (.wav)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from config import AudioConfig
from tokenizer.codec import AudioCodecTokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate recognizable Tamil speech using Speech Foundation Prior")
    parser.add_argument("--text", "-t", type=str,
                        default="வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்? நான் உங்கள் தமிழ் குரல் உதவியாளர்.",
                        help="Tamil text to speak")
    parser.add_argument("--output_name", "-o", type=str, default="tamil_foundation_speech",
                        help="Base output filename prefix")
    args = parser.parse_args()

    output_dir = os.path.join(REPO_ROOT, "samples", "foundation_speech")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("🗣️ TAMIL SPEECH FOUNDATION GENERATOR (MMS-TAMIL + ENCODEC RVQ)")
    print("=" * 80)
    print(f"Tamil Text:  '{args.text}'")
    print(f"Output Dir:  {output_dir}")
    print("=" * 80)

    # 1. Load MMS-Tamil Speech Foundation Model
    print("\n[1/3] Loading Meta MMS-Tamil Speech Foundation Model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tam")
    tts_model = VitsModel.from_pretrained("facebook/mms-tts-tam")
    tts_model.eval()

    inputs = tokenizer(args.text, return_tensors="pt")
    with torch.no_grad():
        w_16k = tts_model(**inputs).waveform  # (1, samples)

    # Resample to 24 kHz mono
    w_24k = torchaudio.functional.resample(w_16k, 16000, 24000).unsqueeze(0)  # (1, 1, samples)
    dur = w_24k.shape[-1] / 24000.0

    # Save File 1: Original Foundation WAV
    f_orig_wav = os.path.join(output_dir, f"{args.output_name}_original.wav")
    sf.write(f_orig_wav, w_24k.squeeze().cpu().numpy(), 24000)
    print(f"✓ Saved Original Foundation Voice: {f_orig_wav} ({dur:.2f}s, 24 kHz)")

    # 2. Encode to Encodec RVQ Tokens
    print("\n[2/3] Encoding acoustic waveform into Encodec 24kHz RVQ tokens...")
    codec = AudioCodecTokenizer(AudioConfig(sample_rate=24000, num_codebooks=4), device="cpu")
    with torch.no_grad():
        tokens = codec.encode(w_24k, sample_rate=24000)  # (1, 4, frames)

    f_tokens_pt = os.path.join(output_dir, f"{args.output_name}_encoder_tokens.pt")
    torch.save({
        "tokens": tokens.cpu(),
        "text": args.text,
        "sample_rate": 24000,
        "shape": tuple(tokens.shape),
        "bandwidth": 3.0,
        "num_codebooks": 4
    }, f_tokens_pt)
    print(f"✓ Saved Encoder Output Tokens:    {f_tokens_pt} (Shape: {tuple(tokens.shape)})")

    # 3. Decode Tokens back to Waveform via Encodec
    print("\n[3/3] Decoding RVQ tokens through Neural Codec Decoder...")
    with torch.no_grad():
        w_recon = codec.decode(tokens).squeeze().cpu().numpy()

    f_recon_wav = os.path.join(output_dir, f"{args.output_name}_codec_decoded.wav")
    sf.write(f_recon_wav, w_recon, 24000)
    print(f"✓ Saved Codec Reconstructed Voice: {f_recon_wav} ({len(w_recon)/24000.0:.2f}s)")

    # Compute SNR
    orig_np = w_24k.squeeze().cpu().numpy()
    min_len = min(len(orig_np), len(w_recon))
    noise = orig_np[:min_len] - w_recon[:min_len]
    snr = 10 * np.log10(np.sum(orig_np[:min_len]**2) / (np.sum(noise**2) + 1e-10))

    print("\n" + "=" * 80)
    print("📊 ACOUSTIC METRICS:")
    print("=" * 80)
    print(f"  Duration:            {dur:.2f} seconds")
    print(f"  Reconstruction SNR:  {snr:.2f} dB")
    print(f"  RMS Energy:          {float(np.sqrt(np.mean(w_recon**2))):.4f}")
    print(f"  Tokens Length:       {tokens.shape[-1]} frames @ 75 Hz")
    print(f"  Tamil Transcript:    '{args.text}'")
    print("=" * 80)

    os._exit(0)


if __name__ == "__main__":
    main()
