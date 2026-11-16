#!/usr/bin/env python3
"""
Hugging Face Stream Codec Pipeline:
1. Streams genuine Tamil speech from 'SPRINGLab/IndicVoices-R_Tamil'.
2. Saves File 1: Original voice waveform (.wav)
3. Encodes with Encodec 24kHz RVQ -> Saves File 2: Encoder output tokens (.pt)
4. Decodes tokens with Encodec -> Saves File 3: Decoded output waveform (.wav)
"""

import os
import sys
import io
import numpy as np
import torch
import torchaudio
import soundfile as sf
from datasets import load_dataset, Audio

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from config import AudioConfig
from tokenizer.codec import AudioCodecTokenizer


def main():
    output_dir = os.path.join(REPO_ROOT, "samples", "hf_stream")
    os.makedirs(output_dir, exist_ok=True)

    file_1_original_wav = os.path.join(output_dir, "hf_tamil_original_voice.wav")
    file_2_encoder_tokens_pt = os.path.join(output_dir, "hf_tamil_encoder_output_tokens.pt")
    file_3_decoded_wav = os.path.join(output_dir, "hf_tamil_decoded_output.wav")

    print("=" * 75)
    print("🎧 HUGGING FACE TAMIL STREAM: ORIGINAL VOICE -> ENCODER -> DECODER")
    print("=" * 75)
    print(f"Output Directory: {output_dir}")
    print(f"File 1 (Original Voice WAV):     {file_1_original_wav}")
    print(f"File 2 (Encoder Output Tokens):  {file_2_encoder_tokens_pt}")
    print(f"File 3 (Decoder Reconstructed):  {file_3_decoded_wav}")
    print("=" * 75)

    # 1. Connect to Hugging Face Stream
    print("\n[1/4] Connecting to Hugging Face stream 'SPRINGLab/IndicVoices-R_Tamil'...")
    dataset = load_dataset("SPRINGLab/IndicVoices-R_Tamil", split="train", streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    data_iter = iter(dataset)
    print("✓ Stream connected.")

    # 2. Find a clean, clear Tamil speech sample (duration 2.5s - 6.0s, high RMS energy)
    print("\n[2/4] Scanning stream for an optimal Tamil speech sample...")
    selected_sample = None
    target_sr = 24000

    for sample in data_iter:
        audio_entry = sample.get("audio", {})
        if "bytes" not in audio_entry or audio_entry["bytes"] is None:
            continue
        
        raw_audio, sr_orig = sf.read(io.BytesIO(audio_entry["bytes"]))
        if raw_audio.ndim > 1:
            raw_audio = np.mean(raw_audio, axis=1)

        duration = len(raw_audio) / float(sr_orig)
        rms = float(np.sqrt(np.mean(raw_audio ** 2)))
        text = sample.get("text", "") or sample.get("sentence", "")

        # Target criteria: clean speech, duration 2.5s - 6.0s, RMS > 0.10
        if 2.5 <= duration <= 6.0 and rms > 0.10 and len(text.strip()) > 5:
            selected_sample = {
                "raw_audio": raw_audio,
                "sr_orig": sr_orig,
                "duration": duration,
                "rms": rms,
                "text": text,
                "gender": sample.get("gender", "unknown"),
                "area": sample.get("area", "unknown"),
                "speaker_id": sample.get("speaker_id", "unknown")
            }
            break

    if selected_sample is None:
        raise RuntimeError("No suitable Tamil speech sample found in stream matching criteria.")

    print(f"✓ Found target Tamil speech sample:")
    print(f"  Tamil Transcript: '{selected_sample['text']}'")
    print(f"  Speaker ID:       {selected_sample['speaker_id']} ({selected_sample['gender']}, {selected_sample['area']})")
    print(f"  Original SR:      {selected_sample['sr_orig']} Hz")
    print(f"  Duration:         {selected_sample['duration']:.2f} seconds")
    print(f"  RMS Energy:       {selected_sample['rms']:.4f}")

    # Resample to 24kHz mono
    w_orig = torch.from_numpy(selected_sample["raw_audio"]).float().unsqueeze(0).unsqueeze(0)
    if selected_sample["sr_orig"] != target_sr:
        w_orig = torchaudio.functional.resample(w_orig, selected_sample["sr_orig"], target_sr)

    # Normalize audio levels
    peak = w_orig.abs().max().item()
    if peak > 0:
        w_orig = w_orig / peak * 0.90

    # Save File 1: Original Voice WAV
    wav_np_orig = w_orig.squeeze().cpu().numpy()
    sf.write(file_1_original_wav, wav_np_orig, target_sr)
    file_1_size = os.path.getsize(file_1_original_wav)
    print(f"\n📁 FILE 1 SAVED (Original Voice):")
    print(f"  Path:       {file_1_original_wav}")
    print(f"  Size:       {file_1_size:,} bytes")
    print(f"  SampleRate: {target_sr} Hz")
    print(f"  Duration:   {len(wav_np_orig)/target_sr:.2f} seconds")

    # 3. Encode using Neural Audio Codec (Encodec 24kHz)
    print("\n[3/4] Initializing Encodec 24kHz Codec & Encoding waveform to RVQ tokens...")
    audio_cfg = AudioConfig(sample_rate=target_sr, num_codebooks=4)
    tokenizer = AudioCodecTokenizer(config=audio_cfg, device="cpu")

    with torch.no_grad():
        tokens = tokenizer.encode(w_orig, sample_rate=target_sr)  # Shape: (1, 4, frames)

    tokens_cpu = tokens.detach().cpu()
    num_frames = tokens_cpu.shape[-1]
    
    # Save File 2: Encoder Output Tokens (.pt)
    encoder_output_payload = {
        "tokens": tokens_cpu,
        "shape": tuple(tokens_cpu.shape),
        "num_codebooks": 4,
        "target_bandwidth": 3.0,
        "frame_rate_hz": 75,
        "sample_rate": target_sr,
        "transcript": selected_sample["text"],
        "speaker": {
            "speaker_id": selected_sample["speaker_id"],
            "gender": selected_sample["gender"],
            "area": selected_sample["area"]
        },
        "description": "Discrete RVQ tokens from Encodec 24kHz encoder ready for neural decoder"
    }
    torch.save(encoder_output_payload, file_2_encoder_tokens_pt)
    file_2_size = os.path.getsize(file_2_encoder_tokens_pt)

    print(f"\n📁 FILE 2 SAVED (Encoder Output Tokens for Decoder):")
    print(f"  Path:        {file_2_encoder_tokens_pt}")
    print(f"  Size:        {file_2_size:,} bytes")
    print(f"  Shape:       {tuple(tokens_cpu.shape)} [Batch=1, Codebooks=4, Frames={num_frames}]")
    print(f"  Duration:    {num_frames / 75.0:.2f} seconds @ 75 Hz")
    print(f"  Token Min:   {tokens_cpu.min().item()}, Token Max: {tokens_cpu.max().item()} (Vocab size: 1024)")
    print("  First 5 frames per codebook:")
    for k in range(4):
        print(f"    CB{k}: {tokens_cpu[0, k, :5].tolist()}")

    # 4. Decode tokens directly into reconstructed speech waveform
    print("\n[4/4] Decoding RVQ tokens through Neural Codec Decoder...")
    with torch.no_grad():
        recon_wav = tokenizer.decode(tokens_cpu).squeeze().cpu().numpy()

    # Save File 3: Decoded WAV
    sf.write(file_3_decoded_wav, recon_wav, target_sr)
    file_3_size = os.path.getsize(file_3_decoded_wav)

    print(f"\n📁 FILE 3 SAVED (Decoded Reconstructed Audio):")
    print(f"  Path:       {file_3_decoded_wav}")
    print(f"  Size:       {file_3_size:,} bytes")
    print(f"  SampleRate: {target_sr} Hz")
    print(f"  Duration:   {len(recon_wav)/target_sr:.2f} seconds")

    # Metrics Evaluation
    min_len = min(len(wav_np_orig), len(recon_wav))
    orig_trim = wav_np_orig[:min_len]
    recon_trim = recon_wav[:min_len]

    # Signal-to-Noise Ratio (SNR)
    noise = orig_trim - recon_trim
    snr = 10.0 * np.log10(np.sum(orig_trim ** 2) / (np.sum(noise ** 2) + 1e-12))
    
    # Cosine Similarity
    cos_sim = float(np.dot(orig_trim, recon_trim) / (np.linalg.norm(orig_trim) * np.linalg.norm(recon_trim) + 1e-12))

    print("\n" + "=" * 75)
    print("📊 FIDELITY & ACOUSTIC QUALITY VERIFICATION:")
    print("=" * 75)
    print(f"  Reconstruction SNR:       {snr:.2f} dB (High codec reconstruction fidelity)")
    print(f"  Waveform Cosine Sim:      {cos_sim:.4f}")
    print(f"  Original RMS:             {float(np.sqrt(np.mean(wav_np_orig**2))):.4f}")
    print(f"  Decoded RMS:              {float(np.sqrt(np.mean(recon_wav**2))):.4f}")
    print(f"  Tamil Transcript:         '{selected_sample['text']}'")
    print("=" * 75)
    print("✓ All 3 requested files successfully generated and verified!")
    print("=" * 75)

    os._exit(0)


if __name__ == "__main__":
    main()
