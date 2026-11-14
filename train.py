"""
Training script for Textless Tamil Speech-to-Speech Model.
Trains the causal Audio-to-Audio LM directly on multi-codebook RVQ tokens.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import AudioConfig, ModelConfig, TrainingConfig
from model.audio_transformer import TextlessTamilSpeechLM
from dataset.tamil_dataset import TamilConversationalAudioDataset


def train(
    data_dir: str = "tokenized_data",
    checkpoint_dir: str = "checkpoints",
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 3e-4,
    device: str = "cpu"
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Starting training on device: {device}")

    # Configurations
    audio_cfg = AudioConfig()
    model_cfg = ModelConfig(audio=audio_cfg)
    
    # Dataset and DataLoader
    dataset = TamilConversationalAudioDataset(data_dir, max_seq_len=model_cfg.max_seq_len, config=audio_cfg)
    if len(dataset) == 0:
        print(f"Error: No token files found in '{data_dir}'. Please prepare tokens first.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Loaded {len(dataset)} dialogue samples across {len(dataloader)} batches (batch_size={batch_size}).")

    # Model
    model = TextlessTamilSpeechLM(config=model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized: {total_params / 1e6:.2f}M trainable parameters.")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-5)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for step, (user_stream, asst_in, asst_target) in enumerate(pbar):
            user_stream = user_stream.to(device)
            asst_in = asst_in.to(device)
            asst_target = asst_target.to(device)

            optimizer.zero_grad()
            logits, loss = model(user_stream, asst_in, target_assistant_tokens=asst_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            avg_loss = running_loss / (step + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        # Save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"tamil_speech_lm_epoch_{epoch}.pt")
        latest_path = os.path.join(checkpoint_dir, "tamil_speech_lm_latest.pt")
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": model_cfg
        }, ckpt_path)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": model_cfg
        }, latest_path)

        print(f"Checkpoint saved: {ckpt_path}")

    print("Training completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Textless Tamil Speech-to-Speech Model")
    parser.add_argument("--data_dir", type=str, default="tokenized_data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
