#!/usr/bin/env python3
"""
Prompt-Conditioned and WSP-Calibrated Speech Generation using Real Tamil HF Tokens.
Uses genuine Tamil speech tokens from 'samples/hf_stream/hf_tamil_encoder_output_tokens.pt'
to prompt the Step-4500 model and its WSP-corrected counterparts, producing listenable human speech.
"""

import os
import sys
import numpy as np
import torch
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from config import AudioConfig, ModelConfig
from model.audio_transformer import TextlessTamilSpeechLM
from tokenizer.codec import AudioCodecTokenizer


def load_model(checkpoint_path: str, device: torch.device) -> TextlessTamilSpeechLM:
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    model_cfg = ModelConfig(
        d_model=1024,
        n_heads=16,
        n_layers=6,
        d_ff=4096,
        max_seq_len=750,
        audio=audio_cfg
    )
    model = TextlessTamilSpeechLM(config=model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    device = torch.device("cpu")
    output_dir = os.path.join(REPO_ROOT, "samples", "human_voice_generation")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load real Tamil prompt tokens
    tokens_file = os.path.join(REPO_ROOT, "samples", "hf_stream", "hf_tamil_encoder_output_tokens.pt")
    assert os.path.exists(tokens_file), f"Tokens file not found: {tokens_file}"
    payload = torch.load(tokens_file, map_location="cpu", weights_only=False)
    real_tokens = payload["tokens"]  # (1, 4, 250)
    transcript = payload.get("transcript", "")
    print("=" * 80)
    print("🗣️ TAMIL SPEECH GENERATION: PROMPT CONDITIONING & WSP CALIBRATION")
    print("=" * 80)
    print(f"Loaded Real Human Tamil Tokens from: {tokens_file}")
    print(f"Transcript: '{transcript}'")
    print(f"Total Sequence Length: {real_tokens.shape[-1]} frames ({real_tokens.shape[-1]/75.0:.2f}s)")
    print(f"Output Directory:      {output_dir}")
    print("=" * 80)

    # Codec Tokenizer for decoding
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    tokenizer = AudioCodecTokenizer(config=audio_cfg, device="cpu")

    # Prompt length: first 1.2 seconds = 90 frames of real human speech
    prompt_len = 90
    prompt_tokens = real_tokens[:, :, :prompt_len].to(device)
    max_new_tokens = 90  # generate another 1.2 seconds continuation

    # Decode prompt audio alone for reference
    with torch.no_grad():
        prompt_wav = tokenizer.decode(prompt_tokens).squeeze().cpu().numpy()
    sf.write(os.path.join(output_dir, "00_real_tamil_prompt.wav"), prompt_wav, 24000)

    # Checkpoints to test
    checkpoints = {
        "raw_4500": os.path.join(REPO_ROOT, "checkpoints_v2_p2_4", "tamil_speech_lm_v2_p2_4_step_4500.pt"),
        "wsp_corrected": os.path.join(REPO_ROOT, "checkpoints", "tamil_speech_lm_v2_p2_4_wsp_corrected.pt"),
        "wsp_human_form": os.path.join(REPO_ROOT, "checkpoints", "tamil_speech_lm_v2_p2_4_human_form.pt")
    }

    results = {}

    for name, path in checkpoints.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: {path} not found.")
            continue

        print(f"\n[{name.upper()}] Loading model from {path}...")
        model = load_model(path, device)

        # 1. Evaluate cross-entropy loss & accuracy on real Tamil speech
        with torch.no_grad():
            bos = torch.full((1, 4, 1), model.bos_token_id, dtype=torch.long, device=device)
            full_tok_bos = torch.cat([bos, real_tokens.to(device)], dim=-1)
            x_in = full_tok_bos[:, :, :-1]
            x_target = full_tok_bos[:, :, 1:]
            _, loss, metrics = model(x_in, target_tokens=x_target)
            print(f"  Evaluation on Real HF Tamil Stream:")
            print(f"    Cross-Entropy Loss: {loss.item():.4f}")
            print(f"    Speech Accuracy:    {metrics.get('speech_accuracy', 0.0)*100:.2f}%")
            for k in range(4):
                print(f"      CB{k} Loss: {metrics.get(f'loss_cb{k}', 0.0):.4f} | Acc: {metrics.get(f'acc_cb{k}', 0.0)*100:.2f}%")

        # 2. Conditioned Generation from Real Human Tamil Prompt
        print(f"  Generating prompt-conditioned continuation ({max_new_tokens} frames / {max_new_tokens/75.0:.2f}s)...")
        with torch.no_grad():
            # Use gentle temperature for speech fidelity
            continuation_tokens = model.generate(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_k=40,
                top_p=0.88,
                repetition_penalty=1.20
            )

        # Full concatenated utterance: Prompt + Generated Continuation
        full_tokens = torch.cat([prompt_tokens, continuation_tokens], dim=-1)

        # Decode to audio
        with torch.no_grad():
            full_wav = tokenizer.decode(full_tokens).squeeze().cpu().numpy()
            cont_wav = tokenizer.decode(continuation_tokens).squeeze().cpu().numpy()

        full_wav_path = os.path.join(output_dir, f"{name}_prompted_full.wav")
        cont_wav_path = os.path.join(output_dir, f"{name}_continuation_only.wav")
        sf.write(full_wav_path, full_wav, 24000)
        sf.write(cont_wav_path, cont_wav, 24000)

        results[name] = {
            "loss": loss.item(),
            "acc": metrics.get("speech_accuracy", 0.0),
            "full_wav": full_wav_path,
            "cont_wav": cont_wav_path,
            "duration": len(full_wav) / 24000.0,
            "rms": float(np.sqrt(np.mean(full_wav**2)))
        }

        print(f"  ✓ Saved Full Prompted Utterance: {full_wav_path} ({len(full_wav)/24000.0:.2f}s, RMS={float(np.sqrt(np.mean(full_wav**2))):.4f})")
        print(f"  ✓ Saved Continuation Only:      {cont_wav_path} ({len(cont_wav)/24000.0:.2f}s, RMS={float(np.sqrt(np.mean(cont_wav**2))):.4f})")

    print("\n" + "=" * 80)
    print("📊 COMPARISON SUMMARY:")
    print("=" * 80)
    for name, res in results.items():
        print(f"  {name:18s} | Loss: {res['loss']:.4f} | Acc: {res['acc']*100:.2f}% | RMS: {res['rms']:.4f} | Full Audio: {os.path.basename(res['full_wav'])}")
    print("=" * 80)
    print(f"✓ Real Tamil prompt audio: {os.path.join(output_dir, '00_real_tamil_prompt.wav')}")
    print("=" * 80)

    os._exit(0)


if __name__ == "__main__":
    main()
