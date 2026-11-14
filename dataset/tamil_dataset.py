"""
Dataset loader and audio pre-tokenization for Tamil Conversational Speech.
Encodes raw audio waveforms into discrete RVQ tokens with caching for fast training.
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from tqdm import tqdm

from config import AudioConfig
from tokenizer.codec import AudioCodecTokenizer
from tokenizer.audio_io import load_audio


class TamilConversationalAudioDataset(Dataset):
    """
    Dataset that pairs user speech tokens with assistant response speech tokens.
    Operates 100% on discrete audio tokens.
    """
    def __init__(
        self,
        token_dir: str,
        max_seq_len: int = 750,
        config: Optional[AudioConfig] = None
    ):
        self.token_dir = token_dir
        self.max_seq_len = max_seq_len
        self.config = config or AudioConfig()
        
        self.file_list = sorted(glob.glob(os.path.join(token_dir, "*.pt")))
        if not self.file_list:
            print(f"[Warning] No token files found in {token_dir}. Run prepare_tokens() first.")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Loads a pre-tokenized dialogue turn.
        Returns:
            user_tokens: (num_codebooks, seq_len)
            assistant_in: (num_codebooks, seq_len)
            assistant_target: (num_codebooks, seq_len)
        """
        data = torch.load(self.file_list[idx], map_location="cpu", weights_only=True)
        user_tok = data["user_tokens"]  # (num_codebooks, user_len)
        asst_tok = data["assistant_tokens"]  # (num_codebooks, asst_len)

        num_codebooks = self.config.num_codebooks
        u_len = user_tok.shape[-1]
        a_len = asst_tok.shape[-1]

        # Construct full-duplex conversational sequence:
        # User talks first, assistant is silent.
        # Then assistant responds, user is silent.
        total_len = u_len + a_len

        # Truncate if exceeds max_seq_len
        if total_len > self.max_seq_len:
            # Scale down lengths proportionally
            ratio = self.max_seq_len / total_len
            u_len = int(u_len * ratio)
            a_len = self.max_seq_len - u_len
            user_tok = user_tok[:, :u_len]
            asst_tok = asst_tok[:, :a_len]
            total_len = self.max_seq_len

        # Construct full User stream
        user_stream = torch.full((num_codebooks, total_len), self.config.silence_token_id, dtype=torch.long)
        user_stream[:, :u_len] = user_tok

        # Construct full Assistant stream
        asst_full = torch.full((num_codebooks, total_len), self.config.silence_token_id, dtype=torch.long)
        asst_full[:, u_len:u_len + a_len] = asst_tok

        # Prepare Autoregressive Shift for Assistant:
        # assistant_in starts with BOS token at t=0
        # assistant_target is shifted by 1 frame
        bos = torch.full((num_codebooks, 1), self.config.bos_token_id, dtype=torch.long)
        asst_in = torch.cat([bos, asst_full[:, :-1]], dim=-1)
        asst_target = asst_full

        # Pad to max_seq_len if shorter
        if total_len < self.max_seq_len:
            pad_len = self.max_seq_len - total_len
            pad_u = torch.full((num_codebooks, pad_len), self.config.pad_token_id, dtype=torch.long)
            pad_a = torch.full((num_codebooks, pad_len), self.config.pad_token_id, dtype=torch.long)
            
            user_stream = torch.cat([user_stream, pad_u], dim=-1)
            asst_in = torch.cat([asst_in, pad_a], dim=-1)
            asst_target = torch.cat([asst_target, pad_a], dim=-1)

        return user_stream, asst_in, asst_target


def prepare_tokens_from_audio_pairs(
    audio_pairs: List[Tuple[str, str]],
    output_dir: str,
    tokenizer: AudioCodecTokenizer,
    sample_rate: int = 24000
):
    """
    Converts raw audio file pairs (user_wav, assistant_wav) into token tensors on disk.
    Args:
        audio_pairs: list of tuples (path_to_user_wav, path_to_asst_wav)
        output_dir: directory to store .pt files
        tokenizer: AudioCodecTokenizer instance
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Tokenizing {len(audio_pairs)} Tamil conversational audio pairs into {output_dir}...")

    for i, (u_path, a_path) in enumerate(tqdm(audio_pairs)):
        try:
            u_wav, sr_u = load_audio(u_path)
            a_wav, sr_a = load_audio(a_path)

            u_tokens = tokenizer.encode(u_wav, sample_rate=sr_u).squeeze(0).cpu()
            a_tokens = tokenizer.encode(a_wav, sample_rate=sr_a).squeeze(0).cpu()

            out_path = os.path.join(output_dir, f"dialogue_{i:06d}.pt")
            torch.save({
                "user_tokens": u_tokens,
                "assistant_tokens": a_tokens
            }, out_path)
        except Exception as e:
            print(f"Error processing pair ({u_path}, {a_path}): {e}")
