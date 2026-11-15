"""
Inference & Speech Generation for Textless Tamil Autoregressive LM (V2).
Supports unconditional speech generation from BOS and continuation from a spoken Tamil prompt.
"""

import os
import argparse
from typing import Optional, Dict
import torch
import numpy as np

from config import AudioConfig, ModelConfig
from tokenizer.codec import AudioCodecTokenizer
from tokenizer.audio_io import load_audio, save_audio
from model.audio_transformer import TextlessTamilSpeechLM


class TamilVoiceAIInference:
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
        self.model_cfg = ModelConfig(
            d_model=1024,
            n_heads=16,
            n_layers=6,
            d_ff=4096,
            max_seq_len=750,
            audio=self.audio_cfg
        )

        # Initialize Tokenizer / Codec
        print("Initializing Neural Audio Codec (Encodec 24kHz)...")
        self.tokenizer = AudioCodecTokenizer(config=self.audio_cfg, device=device)

        # Initialize Model
        print("Initializing Textless Tamil Speech Autoregressive LM...")
        self.model = TextlessTamilSpeechLM(config=self.model_cfg).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print("No checkpoint provided or file not found; running with initialized weights.")

        self.model.eval()

    def generate_speech(
        self,
        prompt_audio_path: Optional[str] = None,
        output_audio_path: str = "generated_tamil_speech.wav",
        duration_sec: float = 2.0,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15
    ) -> Dict[str, any]:
        """
        Generates spoken audio either unconditionally (from BOS) or continuing a spoken prompt.
        """
        num_frames = int(duration_sec * self.audio_cfg.frame_rate)
        prompt_tokens = None

        if prompt_audio_path is not None and os.path.exists(prompt_audio_path):
            print(f"\n[1/4] Reading prompt audio: {prompt_audio_path}")
            wav, sr = load_audio(prompt_audio_path)
            prompt_tokens = self.tokenizer.encode(wav, sample_rate=sr)  # (1, 4, seq_len)
            print(f"      Encoded prompt tokens: {list(prompt_tokens.shape)} ({prompt_tokens.shape[-1]} frames)")
        else:
            print(f"\n[1/4] Unconditional generation from BOS start token.")

        print(f"[2/4] Autoregressively generating {num_frames} frames ({duration_sec:.1f}s) with sequential depth sampling...")
        with torch.no_grad():
            gen_tokens = self.model.generate(
                prompt_tokens=prompt_tokens,
                max_new_tokens=num_frames,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )

        print(f"[3/4] Analyzing token quality and diversity:")
        diagnostics = self._compute_token_diagnostics(gen_tokens)
        for k in range(self.audio_cfg.num_codebooks):
            print(f"      Codebook {k}: {diagnostics[f'cb{k}_unique_tokens']}/{num_frames} unique tokens, "
                  f"repetition rate: {diagnostics[f'cb{k}_consecutive_repetition_rate']*100:.1f}%, "
                  f"valid range: {diagnostics[f'cb{k}_all_valid']}")

        print(f"[4/4] Decoding discrete audio tokens into continuous waveform via Encodec...")
        wav_out = self.tokenizer.decode(gen_tokens).squeeze(0).cpu()

        os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
        save_audio(output_audio_path, wav_out, self.audio_cfg.sample_rate)
        actual_duration = wav_out.shape[-1] / self.audio_cfg.sample_rate
        print(f"✓ Saved generated audio to: {output_audio_path} ({actual_duration:.2f}s, {self.audio_cfg.sample_rate} Hz)")

        diagnostics["output_path"] = output_audio_path
        diagnostics["duration"] = actual_duration
        diagnostics["generated_tokens"] = gen_tokens
        return diagnostics

    def _compute_token_diagnostics(self, tokens: torch.Tensor) -> Dict[str, any]:
        """
        Inspects token diversity, repetition, and valid codebook boundaries.
        """
        diag = {}
        tok_np = tokens.squeeze(0).cpu().numpy()  # (num_codebooks, seq_len)
        num_codebooks, seq_len = tok_np.shape

        for k in range(num_codebooks):
            cb_toks = tok_np[k]
            unique_count = len(np.unique(cb_toks))
            repetition_rate = (cb_toks[1:] == cb_toks[:-1]).mean() if seq_len > 1 else 0.0
            all_valid = bool((cb_toks >= 0).all() and (cb_toks < self.audio_cfg.codebook_size).all())

            diag[f"cb{k}_unique_tokens"] = unique_count
            diag[f"cb{k}_consecutive_repetition_rate"] = float(repetition_rate)
            diag[f"cb{k}_all_valid"] = all_valid

        return diag

    # Alias for backward compatibility
    def respond_to_speech(self, input_audio_path: str, output_audio_path: str = "response_tamil.wav", **kwargs):
        return self.generate_speech(prompt_audio_path=input_audio_path, output_audio_path=output_audio_path, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Textless Tamil Speech LM (V2) Inference")
    parser.add_argument("--prompt", type=str, default=None, help="Path to prompt Tamil .wav (optional)")
    parser.add_argument("--output", type=str, default="generated_tamil_speech.wav", help="Path to save output .wav")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration in seconds")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    engine = TamilVoiceAIInference(checkpoint_path=args.checkpoint, device=args.device)
    engine.generate_speech(
        prompt_audio_path=args.prompt,
        output_audio_path=args.output,
        duration_sec=args.duration,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty
    )
