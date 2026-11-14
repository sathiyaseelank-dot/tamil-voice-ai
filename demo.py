"""
End-to-End Demo and Verification for Textless Tamil Speech-to-Speech System.
Validates the complete pipeline:
  1. Synthetic conversational Tamil audio generation
  2. RVQ Audio tokenization (zero text)
  3. Pre-tokenization data caching
  4. Speech Language Model training (forward + backward pass)
  5. Autoregressive speech generation
  6. Audio decoding back to Tamil speech waveform (.wav)
"""

import os
import sys
import torch

from dataset.download_tamil_audio import create_sample_dialogue_pairs
from dataset.tamil_dataset import prepare_tokens_from_audio_pairs, TamilConversationalAudioDataset
from tokenizer.codec import AudioCodecTokenizer
from train import train
from inference import TamilVoiceAIInference


def run_demo():
    print("=" * 70)
    print("🚀 TEXTLESS TAMIL SPEECH-TO-SPEECH MODEL: END-TO-END DEMO")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(base_dir, "samples")
    token_dir = os.path.join(base_dir, "tokenized_data")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    demo_output = os.path.join(base_dir, "demo_response_tamil.wav")

    # Step 1: Generate sample conversational turns
    print("\n[Step 1/6] Generating sample conversational turns...")
    pairs = create_sample_dialogue_pairs(output_dir=samples_dir, num_pairs=6)

    # Step 2: Initialize Audio Codec Tokenizer
    print("\n[Step 2/6] Loading Neural Audio Codec (RVQ Tokenizer)...")
    tokenizer = AudioCodecTokenizer(device="cpu")

    # Step 3: Tokenize audio to discrete codes
    print("\n[Step 3/6] Pre-tokenizing audio pairs into discrete token tensors...")
    prepare_tokens_from_audio_pairs(pairs, output_dir=token_dir, tokenizer=tokenizer)

    # Step 4: Verify dataset loading
    print("\n[Step 4/6] Verifying dataset and collation...")
    dataset = TamilConversationalAudioDataset(token_dir=token_dir, max_seq_len=300)
    print(f"Dataset successfully created with {len(dataset)} items.")
    u_sample, a_in_sample, a_tgt_sample = dataset[0]
    print(f"Sample shapes -> User tokens: {list(u_sample.shape)}, Target: {list(a_tgt_sample.shape)}")

    # Step 5: Run mini-training to verify forward & backward passes
    print("\n[Step 5/6] Training Speech-to-Speech LM on conversational tokens (2 epochs)...")
    train(
        data_dir=token_dir,
        checkpoint_dir=ckpt_dir,
        epochs=2,
        batch_size=2,
        lr=5e-4,
        device="cpu"
    )

    # Step 6: Test Inference Pipeline
    print("\n[Step 6/6] Running End-to-End Inference from Audio...")
    latest_ckpt = os.path.join(ckpt_dir, "tamil_speech_lm_latest.pt")
    engine = TamilVoiceAIInference(checkpoint_path=latest_ckpt, device="cpu")
    
    test_input = pairs[0][0]
    engine.respond_to_speech(
        input_audio_path=test_input,
        output_audio_path=demo_output,
        max_duration_sec=2.0
    )

    print("\n" + "=" * 70)
    print("✅ DEMO SUCCEEDED! All pipeline components verified:")
    print(f"  • Input Audio:  {test_input}")
    print(f"  • Output Audio: {demo_output}")
    print(f"  • Checkpoint:   {latest_ckpt}")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
