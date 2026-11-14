"""
Inference & Generation for Textless Tamil Speech-to-Speech Model.
Takes Tamil user speech audio, generates spoken Tamil response audio directly
without text transcription.
"""

import os
import argparse
from typing import Optional
import torch

from config import AudioConfig, ModelConfig
from tokenizer.codec import AudioCodecTokenizer
from tokenizer.audio_io import load_audio, save_audio
from model.audio_transformer import TextlessTamilSpeechLM


class TamilVoiceAIInference:
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.audio_cfg = AudioConfig()
        self.model_cfg = ModelConfig(audio=self.audio_cfg)
        
        # Initialize Tokenizer / Codec
        print("Initializing Neural Audio Codec (Encodec 24kHz)...")
        self.tokenizer = AudioCodecTokenizer(config=self.audio_cfg, device=device)

        # Initialize Model
        print("Initializing Speech-to-Speech Language Model...")
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

    def respond_to_speech(
        self,
        input_audio_path: str,
        output_audio_path: str = "response_tamil.wav",
        max_duration_sec: float = 3.0,
        temperature: float = 0.35,
        top_k: int = 40,
        top_p: float = 0.85,
        repetition_penalty: float = 1.25
    ) -> str:
        """
        Takes input Tamil audio file, processes it end-to-end, and writes the response audio file.
        """
        print(f"\n[1] Reading user input speech from: {input_audio_path}")
        wav, sr = load_audio(input_audio_path)
        input_duration = wav.shape[-1] / sr
        print(f"    Input Duration: {input_duration:.2f}s | Sample Rate: {sr} Hz")

        # Encode speech waveform directly into discrete RVQ tokens
        print("[2] Tokenizing speech waveform into RVQ discrete tokens (Zero text)...")
        user_tokens = self.tokenizer.encode(wav, sample_rate=sr)  # (1, num_codebooks, seq_len)
        print(f"    Encoded Tokens Shape: {list(user_tokens.shape)} ({user_tokens.shape[-1]} frames)")

        # Autoregressively generate assistant audio tokens
        num_response_frames = int(max_duration_sec * self.audio_cfg.frame_rate)
        print(f"[3] Autoregressively predicting next speech tokens ({num_response_frames} frames = {max_duration_sec:.1f}s)...")
        with torch.no_grad():
            response_tokens = self.model.generate(
                user_tokens=user_tokens,
                max_new_tokens=num_response_frames,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )

        print(f"    Generated Response Tokens Shape: {list(response_tokens.shape)}")

        # Decode response tokens back into continuous audio waveform
        print("[4] Synthesizing continuous audio waveform from generated speech tokens...")
        response_wav = self.tokenizer.decode(response_tokens).squeeze(0).cpu()

        # Save output audio
        save_audio(output_audio_path, response_wav, self.audio_cfg.sample_rate)
        output_duration = response_wav.shape[-1] / self.audio_cfg.sample_rate
        print(f"[5] Saved generated Tamil response speech to: {output_audio_path} ({output_duration:.2f}s, {self.audio_cfg.sample_rate} Hz)")

        return output_audio_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference on Textless Tamil Speech Model")
    parser.add_argument("--input", type=str, required=True, help="Path to input Tamil .wav file")
    parser.add_argument("--output", type=str, default="response_tamil.wav", help="Path to save output .wav")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--duration", type=float, default=2.5, help="Response duration in seconds")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    engine = TamilVoiceAIInference(checkpoint_path=args.checkpoint, device=args.device)
    engine.respond_to_speech(args.input, output_audio_path=args.output, max_duration_sec=args.duration)
