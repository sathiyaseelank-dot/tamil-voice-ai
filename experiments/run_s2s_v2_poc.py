#!/usr/bin/env python3
"""
S2S V2 Proof-of-Concept Experiment Runner (Phases A–E)
Mandatory Requirements:
1. Timestamp-based MMS-to-Encodec alignment (accounting for receptive fields and frame centres).
2. Separate 16 kHz (MMS) and 24 kHz (Encodec) audio processing.
3. Compare 3 Initializations: Random, MMS, and Validated WSP.
4. Frozen Encodec and MMS representations.
5. 50 utterances for pipeline verification; evaluate on held-out speaker-independent validation set.
6. Decode predicted integer tokens through original Encodec decoder.
7. Zero text, transcripts, phonemes, or VITS components.
"""

import os
import sys
import math
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import torchaudio
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from config import AudioConfig, ModelConfig
from tokenizer.codec import AudioCodecTokenizer
from transformers import Wav2Vec2Model


class TimestampAligner(nn.Module):
    """
    Continuous timestamp-based alignment between MMS (16 kHz, 50 Hz)
    and Encodec (24 kHz, 75 Hz).
    
    Exact Temporal Physics:
    - MMS-300M Conv Feature Extractor:
        Cumulative Stride: 320 samples at 16 kHz = 20.0 ms per frame (50 Hz).
        Receptive Field: 400 samples at 16 kHz = 25.0 ms.
        Frame m centre: t_mms(m) = (320 * m + 199.5) / 16000.0 s.
    - Encodec 24kHz:
        Cumulative Stride: 320 samples at 24 kHz = 13.333 ms per frame (75 Hz).
        Frame n centre: t_codec(n) = (320 * n + 159.5) / 24000.0 s.
    - Mapping coordinate for Encodec frame n along MMS timeline:
        mu(n) = (t_codec(n) - t_mms(0)) / 0.020
              = (2/3) * n - 0.291146
    """
    def __init__(self, d_in: int = 1024, d_out: int = 1024):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.smooth_conv = nn.Conv1d(d_out, d_out, kernel_size=3, padding=1)

    def forward(self, h_mms: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Args:
            h_mms: (batch, T_mms, d_in) from MMS-300M
            target_len: target number of 75 Hz Encodec frames
        Returns:
            h_aligned: (batch, target_len, d_out) aligned to Encodec frame centres
        """
        B, T_mms, D = h_mms.shape
        device = h_mms.device

        # Compute continuous target coordinates
        n_idx = torch.arange(target_len, dtype=torch.float32, device=device)
        # mu(n) = (2/3) * n - 0.29114583
        mu = (2.0 / 3.0) * n_idx - 0.29114583
        mu = torch.clamp(mu, 0.0, float(T_mms - 1))

        mu_floor = torch.floor(mu).long()
        mu_ceil = torch.clamp(mu_floor + 1, max=T_mms - 1)
        w = (mu - mu_floor.float()).unsqueeze(0).unsqueeze(-1)  # (1, target_len, 1)

        # Gather continuous representations via linear interpolation
        # h_floor, h_ceil: (B, target_len, D)
        h_floor = h_mms[:, mu_floor, :]
        h_ceil = h_mms[:, mu_ceil, :]
        h_interp = (1.0 - w) * h_floor + w * h_ceil

        # Apply projection, norm, and temporal smoothing conv
        h_proj = self.norm(self.proj(h_interp))
        # Smooth along temporal dimension: (B, D, target_len)
        h_conv = self.smooth_conv(h_proj.transpose(1, 2)).transpose(1, 2)
        return h_proj + h_conv


class DepthAudioHead(nn.Module):
    """
    Hierarchical Multi-Codebook Prediction Head for 4 RVQ codebooks.
    """
    def __init__(self, d_model: int = 1024, num_codebooks: int = 4, vocab_size: int = 1029):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(num_codebooks)
        ])
        self.depth_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for _ in range(num_codebooks - 1)
        ])

    def forward(self, h: torch.Tensor, target_tokens: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            h: (batch, seq_len, d_model)
            target_tokens: optional (batch, num_codebooks, seq_len) for teacher-forcing
        Returns:
            logits: (batch, num_codebooks, seq_len, vocab_size)
        """
        B, T, D = h.shape
        logits_list = []

        curr_h = h
        logits_0 = self.heads[0](curr_h)
        logits_list.append(logits_0)

        for k in range(1, self.num_codebooks):
            if target_tokens is not None:
                prev_tokens = target_tokens[:, k - 1, :]
            else:
                prev_tokens = torch.argmax(logits_list[-1], dim=-1)

            # Clamp valid range
            prev_tokens = torch.clamp(prev_tokens, 0, self.vocab_size - 1)
            depth_emb = self.depth_embeddings[k - 1](prev_tokens)
            curr_h = curr_h + depth_emb
            logits_k = self.heads[k](curr_h)
            logits_list.append(logits_k)

        return torch.stack(logits_list, dim=1)


class S2STokenPredictor(nn.Module):
    """
    Textless Speech-to-Speech Acoustic Token Predictor.
    Maps MMS continuous representations to discrete Encodec multi-codebook sequences.
    """
    def __init__(self, d_model: int = 1024, n_heads: int = 16, n_layers: int = 6, d_ff: int = 4096, vocab_size: int = 1029):
        super().__init__()
        self.aligner = TimestampAligner(d_in=1024, d_out=d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = DepthAudioHead(d_model=d_model, num_codebooks=4, vocab_size=vocab_size)

    def forward(self, h_mms: torch.Tensor, target_len: int, target_tokens: torch.Tensor = None) -> torch.Tensor:
        # Align 50 Hz MMS features to 75 Hz Encodec time points
        h_aligned = self.aligner(h_mms, target_len=target_len)
        # Backbone Transformer
        h_trans = self.transformer(h_aligned)
        h_norm = self.ln_f(h_trans)
        # Hierarchical Depth Head
        logits = self.head(h_norm, target_tokens=target_tokens)
        return logits


def load_audio_pair(file_path: str, start_sec: float = 0.0, dur_sec: float = 2.0):
    """
    Loads raw audio and produces:
    - audio_16k for MMS-300M (strictly 16 kHz)
    - audio_24k for Encodec (strictly 24 kHz)
    """
    wav, sr = sf.read(file_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    
    start_sample = int(start_sec * sr)
    dur_sample = int(dur_sec * sr)
    segment = wav[start_sample:start_sample + dur_sample]
    if len(segment) < dur_sample:
        segment = np.pad(segment, (0, dur_sample - len(segment)))

    # Convert to torch
    t_seg = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)  # (1, samples)

    # Resample separately
    audio_16k = torchaudio.functional.resample(t_seg, sr, 16000)
    audio_24k = torchaudio.functional.resample(t_seg, sr, 24000)

    # Peak normalize
    if audio_16k.abs().max() > 0:
        audio_16k = audio_16k / audio_16k.abs().max() * 0.95
    if audio_24k.abs().max() > 0:
        audio_24k = audio_24k / audio_24k.abs().max() * 0.95

    return audio_16k, audio_24k


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor):
    """
    Computes per-codebook loss, perplexity, top-1 and top-5 accuracy.
    logits: (B, 4, T, vocab_size)
    targets: (B, 4, T)
    """
    B, K, T, V = logits.shape
    metrics = {}
    total_loss = 0.0
    weights = [1.0, 0.5, 0.25, 0.125]

    for k in range(K):
        l_k = logits[:, k, :, :].reshape(-1, V)
        t_k = targets[:, k, :].reshape(-1)
        ce_loss = F.cross_entropy(l_k, t_k).item()
        ppl = math.exp(min(ce_loss, 20.0))
        
        preds = torch.argmax(l_k, dim=-1)
        top1 = (preds == t_k).float().mean().item() * 100.0

        _, top5_idx = torch.topk(l_k, k=5, dim=-1)
        top5 = (top5_idx == t_k.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0

        metrics[f"loss_cb{k}"] = ce_loss
        metrics[f"ppl_cb{k}"] = ppl
        metrics[f"acc_cb{k}_top1"] = top1
        metrics[f"acc_cb{k}_top5"] = top5
        total_loss += weights[k] * ce_loss

    metrics["total_loss"] = total_loss
    return metrics


def main():
    print("=" * 80)
    print("🚀 S2S V2 PROOF-OF-CONCEPT: PHASES A–E FORENSIC EXECUTION")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    out_dir = os.path.join(REPO_ROOT, "samples", "s2s_v2_poc")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Pretrained Models (MMS-300M and Encodec 24kHz)
    print("\n[Phase A] Loading Pretrained MMS-300M and Encodec 24kHz...")
    mms = Wav2Vec2Model.from_pretrained("facebook/mms-300m").to(device)
    mms.eval()
    for p in mms.parameters():
        p.requires_grad = False

    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    tokenizer = AudioCodecTokenizer(audio_cfg, device=str(device))
    print("✓ MMS-300M and Encodec 24kHz loaded and frozen.")

    # 2. Curate 50 Pipeline Verification Utterances and Speaker-Independent Validation Set
    print("\n[Phase A] Curating 50 Pipeline Verification Utterances and Validation Utterances...")
    
    train_sources = [
        ("samples/hf_stream/hf_tamil_original_voice.wav", 0.0, 1.5),
        ("samples/hf_stream/hf_tamil_original_voice.wav", 1.0, 1.5),
        ("samples/foundation_speech/tamil_sentence_continuation_original.wav", 0.0, 1.5),
        ("samples/foundation_speech/tamil_sentence_continuation_original.wav", 1.0, 1.5),
        ("samples/foundation_speech/tamil_sentence_continuation_original.wav", 2.0, 1.5),
        ("samples/foundation_speech/tamil_sentence_continuation_original.wav", 3.0, 1.5),
        ("samples/tamil_speech_01_foundation.wav", 0.0, 1.5),
        ("samples/tamil_speech_01_foundation.wav", 1.0, 1.5),
        ("samples/tamil_speech_01_foundation.wav", 2.0, 1.5),
        ("samples/tamil_speech_01_foundation.wav", 3.0, 1.5),
        ("samples/tamil_speech_02_foundation.wav", 0.0, 1.5),
        ("samples/tamil_speech_02_foundation.wav", 1.0, 1.5),
        ("samples/tamil_speech_02_foundation.wav", 2.0, 1.5),
        ("samples/tamil_speech_02_foundation.wav", 3.0, 1.5),
        ("samples/tamil_speech_02_foundation.wav", 4.0, 1.5),
        ("samples/tamil_speech_02_foundation.wav", 5.0, 1.5),
        ("demo_response_tamil.wav", 0.0, 1.5),
        ("wsp_tamil_response.wav", 0.0, 1.5),
    ]
    for i in range(6):
        train_sources.append((f"samples/asst_turn_{i:04d}.wav", 0.0, 1.5))
        train_sources.append((f"samples/asst_turn_{i:04d}.wav", 1.0, 1.5))

    # Replicate/expand to exactly 50 distinct segments
    pipeline_items = []
    idx = 0
    while len(pipeline_items) < 50:
        src, s, dur = train_sources[idx % len(train_sources)]
        shift = (idx // len(train_sources)) * 0.2
        pipeline_items.append((src, s + shift, dur))
        idx += 1

    # Held-out speaker-independent validation set (Speaker MMS, Speaker Indic4, Speaker User)
    val_sources = [
        ("samples/mms_tamil_original.wav", 0.0, 2.0, "speaker_mms"),
        ("samples/mms_tamil_original.wav", 2.0, 2.0, "speaker_mms"),
        ("samples/tamil_speech_03_foundation.wav", 0.0, 2.0, "speaker_indic4"),
        ("samples/tamil_speech_03_foundation.wav", 2.0, 2.0, "speaker_indic4"),
        ("samples/user_turn_0000.wav", 0.0, 1.7, "speaker_user"),
        ("samples/user_turn_0001.wav", 0.0, 2.0, "speaker_user"),
        ("samples/user_turn_0002.wav", 0.0, 2.0, "speaker_user"),
        ("samples/user_turn_0003.wav", 0.0, 2.0, "speaker_user"),
    ]

    print(f"✓ Pipeline verification dataset: {len(pipeline_items)} utterances.")
    print(f"✓ Speaker-independent validation set: {len(val_sources)} utterances across 3 held-out speakers.")

    # 3. Extract Features and Tokens
    print("\n[Phase A & B] Extracting MMS Features & Encodec Tokens with Timestamp Alignment...")
    train_data = []
    for relpath, start_s, dur_s in pipeline_items:
        fpath = os.path.join(REPO_ROOT, relpath)
        a16, a24 = load_audio_pair(fpath, start_s, dur_s)
        a16, a24 = a16.to(device), a24.to(device)

        with torch.no_grad():
            # MMS extraction (Layer 18 representation)
            out_mms = mms(a16, output_hidden_states=True)
            # Hidden state 18 (index 18)
            h_mms = out_mms.hidden_states[18]  # (1, T_50, 1024)
            # Encodec tokens
            tokens_codec = tokenizer.encode(a24, sample_rate=24000)  # (1, 4, T_75)

        train_data.append({
            "h_mms": h_mms.cpu(),
            "tokens": tokens_codec.cpu(),
            "T_codec": tokens_codec.shape[-1]
        })

    val_data = []
    for relpath, start_s, dur_s, spk in val_sources:
        fpath = os.path.join(REPO_ROOT, relpath)
        a16, a24 = load_audio_pair(fpath, start_s, dur_s)
        a16, a24 = a16.to(device), a24.to(device)

        with torch.no_grad():
            out_mms = mms(a16, output_hidden_states=True)
            h_mms = out_mms.hidden_states[18]
            tokens_codec = tokenizer.encode(a24, sample_rate=24000)

        val_data.append({
            "relpath": relpath,
            "speaker": spk,
            "a16": a16.cpu(),
            "a24": a24.cpu(),
            "h_mms": h_mms.cpu(),
            "tokens": tokens_codec.cpu(),
            "T_codec": tokens_codec.shape[-1]
        })

    print(f"✓ Feature extraction complete: 50 train items, {len(val_data)} validation items.")

    # Audit Phase B Alignment
    sample_mms = train_data[0]["h_mms"]
    sample_tok = train_data[0]["tokens"]
    t_mms_len = sample_mms.shape[1]
    t_tok_len = sample_tok.shape[-1]
    print(f"  • Sample Utterance: MMS frames (50Hz) = {t_mms_len}, Encodec frames (75Hz) = {t_tok_len}")
    print(f"  • Ratio: {t_tok_len / t_mms_len:.4f} (Ideal: 1.5000)")

    # 4. Phase C: Initialize 3 Experimental Controls
    print("\n[Phase C] Instantiating 3 Initialization Controls...")
    models = {}

    # Control 1: Random Init
    torch.manual_seed(42)
    m_rand = S2STokenPredictor(d_model=1024, n_heads=16, n_layers=6, d_ff=4096, vocab_size=1029).to(device)
    models["random_init"] = m_rand

    # Control 2: MMS Init (Transfer layers 0-5 from MMS-300M Transformer)
    torch.manual_seed(42)
    m_mms = S2STokenPredictor(d_model=1024, n_heads=16, n_layers=6, d_ff=4096, vocab_size=1029).to(device)
    with torch.no_grad():
        for i in range(6):
            mms_l = mms.encoder.layers[i]
            tgt_l = m_mms.transformer.layers[i]
            # Map attention weights
            tgt_l.self_attn.in_proj_weight.copy_(
                torch.cat([mms_l.attention.q_proj.weight, mms_l.attention.k_proj.weight, mms_l.attention.v_proj.weight], dim=0)
            )
            tgt_l.self_attn.in_proj_bias.copy_(
                torch.cat([mms_l.attention.q_proj.bias, mms_l.attention.k_proj.bias, mms_l.attention.v_proj.bias], dim=0)
            )
            tgt_l.self_attn.out_proj.weight.copy_(mms_l.attention.out_proj.weight)
            tgt_l.self_attn.out_proj.bias.copy_(mms_l.attention.out_proj.bias)
            # Map feedforward
            tgt_l.linear1.weight.copy_(mms_l.feed_forward.intermediate_dense.weight)
            tgt_l.linear1.bias.copy_(mms_l.feed_forward.intermediate_dense.bias)
            tgt_l.linear2.weight.copy_(mms_l.feed_forward.output_dense.weight)
            tgt_l.linear2.bias.copy_(mms_l.feed_forward.output_dense.bias)
            tgt_l.norm1.weight.copy_(mms_l.layer_norm.weight)
            tgt_l.norm1.bias.copy_(mms_l.layer_norm.bias)
            tgt_l.norm2.weight.copy_(mms_l.final_layer_norm.weight)
            tgt_l.norm2.bias.copy_(mms_l.final_layer_norm.bias)
    models["mms_init"] = m_mms

    # Control 3: Validated WSP Init
    torch.manual_seed(42)
    m_wsp = S2STokenPredictor(d_model=1024, n_heads=16, n_layers=6, d_ff=4096, vocab_size=1029).to(device)
    wsp_path = os.path.join(REPO_ROOT, "checkpoints", "tamil_speech_lm_v2_p2_4_wsp_corrected.pt")
    wsp_ckpt = torch.load(wsp_path, map_location="cpu", weights_only=False)
    wsp_sd = wsp_ckpt["model_state_dict"] if "model_state_dict" in wsp_ckpt else wsp_ckpt
    with torch.no_grad():
        # Load transformer layers directly from validated WSP
        for i in range(6):
            tgt_l = m_wsp.transformer.layers[i]
            prefix = f"transformer.layers.{i}."
            tgt_l.self_attn.in_proj_weight.copy_(wsp_sd[prefix + "self_attn.in_proj_weight"])
            tgt_l.self_attn.in_proj_bias.copy_(wsp_sd[prefix + "self_attn.in_proj_bias"])
            tgt_l.self_attn.out_proj.weight.copy_(wsp_sd[prefix + "self_attn.out_proj.weight"])
            tgt_l.self_attn.out_proj.bias.copy_(wsp_sd[prefix + "self_attn.out_proj.bias"])
            tgt_l.linear1.weight.copy_(wsp_sd[prefix + "linear1.weight"])
            tgt_l.linear1.bias.copy_(wsp_sd[prefix + "linear1.bias"])
            tgt_l.linear2.weight.copy_(wsp_sd[prefix + "linear2.weight"])
            tgt_l.linear2.bias.copy_(wsp_sd[prefix + "linear2.bias"])
            tgt_l.norm1.weight.copy_(wsp_sd[prefix + "norm1.weight"])
            tgt_l.norm1.bias.copy_(wsp_sd[prefix + "norm1.bias"])
            tgt_l.norm2.weight.copy_(wsp_sd[prefix + "norm2.weight"])
            tgt_l.norm2.bias.copy_(wsp_sd[prefix + "norm2.bias"])
        # Load final norm
        m_wsp.ln_f.weight.copy_(wsp_sd["ln_f.weight"])
        m_wsp.ln_f.bias.copy_(wsp_sd["ln_f.bias"])
        # Load depth head heads and embeddings
        for k in range(4):
            m_wsp.head.heads[k].weight.copy_(wsp_sd[f"head.heads.{k}.weight"])
            m_wsp.head.heads[k].bias.copy_(wsp_sd[f"head.heads.{k}.bias"])
        for k in range(3):
            m_wsp.head.depth_embeddings[k].weight.copy_(wsp_sd[f"head.depth_embeddings.{k}.weight"])
    models["wsp_init"] = m_wsp

    print("✓ All 3 controls instantiated successfully.")

    # 5. Training Loop for each control
    num_epochs = 15
    training_histories = {}

    for name, model in models.items():
        print(f"\n--- Training Control: {name} ({num_epochs} epochs) ---")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        history = []

        start_time = time.time()
        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_loss = 0.0
            cb0_acc_list = []
            
            for item in train_data:
                optimizer.zero_grad()
                h_in = item["h_mms"].to(device)
                targets = item["tokens"].to(device)
                T_tgt = item["T_codec"]

                logits = model(h_in, target_len=T_tgt, target_tokens=targets)
                m = compute_metrics(logits, targets)
                loss = logits.new_tensor(m["total_loss"], requires_grad=True)
                
                # Recompute autograd graph
                V = logits.shape[-1]
                loss_autograd = (
                    F.cross_entropy(logits[:, 0, :, :].reshape(-1, V), targets[:, 0, :].reshape(-1)) * 1.0 +
                    F.cross_entropy(logits[:, 1, :, :].reshape(-1, V), targets[:, 1, :].reshape(-1)) * 0.5 +
                    F.cross_entropy(logits[:, 2, :, :].reshape(-1, V), targets[:, 2, :].reshape(-1)) * 0.25 +
                    F.cross_entropy(logits[:, 3, :, :].reshape(-1, V), targets[:, 3, :].reshape(-1)) * 0.125
                )
                loss_autograd.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss_autograd.item()
                cb0_acc_list.append(m["acc_cb0_top1"])

            avg_loss = epoch_loss / len(train_data)
            avg_cb0_acc = np.mean(cb0_acc_list)
            history.append({"epoch": epoch, "loss": avg_loss, "cb0_acc": avg_cb0_acc})

            if epoch in [1, 5, 10, 15]:
                print(f"  Epoch {epoch:02d}/{num_epochs}: Total Loss = {avg_loss:.4f} | CB0 Top-1 Acc = {avg_cb0_acc:.2f}%")

        training_histories[name] = history
        print(f"✓ {name} training finished in {time.time() - start_time:.1f}s.")

    # 6. Phase D: Evaluation on Complete Speaker-Independent Validation Set
    print("\n" + "=" * 80)
    print("📊 [Phase D] EVALUATING ON SPEAKER-INDEPENDENT VALIDATION SET")
    print("=" * 80)

    val_eval_results = {}
    for name, model in models.items():
        model.eval()
        spk_metrics = []
        with torch.no_grad():
            for item in val_data:
                h_in = item["h_mms"].to(device)
                targets = item["tokens"].to(device)
                T_tgt = item["T_codec"]

                logits = model(h_in, target_len=T_tgt, target_tokens=targets)
                m = compute_metrics(logits, targets)
                m["speaker"] = item["speaker"]
                m["relpath"] = item["relpath"]
                spk_metrics.append(m)

        # Aggregate across validation set
        agg = {
            "total_loss": np.mean([x["total_loss"] for x in spk_metrics]),
            "loss_cb0": np.mean([x["loss_cb0"] for x in spk_metrics]),
            "loss_cb1": np.mean([x["loss_cb1"] for x in spk_metrics]),
            "loss_cb2": np.mean([x["loss_cb2"] for x in spk_metrics]),
            "loss_cb3": np.mean([x["loss_cb3"] for x in spk_metrics]),
            "ppl_cb0": np.mean([x["ppl_cb0"] for x in spk_metrics]),
            "ppl_cb1": np.mean([x["ppl_cb1"] for x in spk_metrics]),
            "ppl_cb2": np.mean([x["ppl_cb2"] for x in spk_metrics]),
            "ppl_cb3": np.mean([x["ppl_cb3"] for x in spk_metrics]),
            "acc_cb0_top1": np.mean([x["acc_cb0_top1"] for x in spk_metrics]),
            "acc_cb0_top5": np.mean([x["acc_cb0_top5"] for x in spk_metrics]),
            "acc_cb1_top1": np.mean([x["acc_cb1_top1"] for x in spk_metrics]),
            "acc_cb2_top1": np.mean([x["acc_cb2_top1"] for x in spk_metrics]),
            "acc_cb3_top1": np.mean([x["acc_cb3_top1"] for x in spk_metrics]),
        }
        val_eval_results[name] = {"aggregate": agg, "per_utterance": spk_metrics}
        print(f"\nResults for [{name}] on Held-Out Validation Speakers:")
        print(f"  • Total Loss:    {agg['total_loss']:.4f}")
        print(f"  • CB0 Top-1 Acc: {agg['acc_cb0_top1']:.2f}% | Top-5 Acc: {agg['acc_cb0_top5']:.2f}% | PPL: {agg['ppl_cb0']:.2f}")
        print(f"  • CB1 Top-1 Acc: {agg['acc_cb1_top1']:.2f}% | PPL: {agg['ppl_cb1']:.2f}")
        print(f"  • CB2 Top-1 Acc: {agg['acc_cb2_top1']:.2f}% | PPL: {agg['ppl_cb2']:.2f}")
        print(f"  • CB3 Top-1 Acc: {agg['acc_cb3_top1']:.2f}% | PPL: {agg['ppl_cb3']:.2f}")

    # 7. Phase E: Acoustic Reconstruction through Original Encodec Decoder
    print("\n" + "=" * 80)
    print("🔊 [Phase E] ACOUSTIC RECONSTRUCTION THROUGH ORIGINAL ENCODEC DECODER")
    print("=" * 80)

    # Pick representative held-out speaker utterance (e.g. Speaker MMS)
    test_item = val_data[0]
    h_test = test_item["h_mms"].to(device)
    tgt_tokens = test_item["tokens"].to(device)
    T_tgt = test_item["T_codec"]

    # Decode ground truth reference
    with torch.no_grad():
        wav_gt_codec = tokenizer.decode(tgt_tokens).squeeze().cpu().numpy()
    sf.write(os.path.join(out_dir, "val_spk_gt_codec.wav"), wav_gt_codec, 24000)
    print(f"✓ Saved ground-truth Encodec reconstruction: {os.path.join(out_dir, 'val_spk_gt_codec.wav')}")

    snr_results = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            # Greedy prediction of discrete tokens
            logits = model(h_test, target_len=T_tgt)
            pred_tokens = torch.argmax(logits, dim=-1)  # (1, 4, T_tgt)
            # Strictly integer tokens into pure Encodec decoder
            wav_recon = tokenizer.decode(pred_tokens).squeeze().cpu().numpy()

        # Compute SNR against GT codec reconstruction
        min_len = min(len(wav_gt_codec), len(wav_recon))
        sig_power = np.mean(wav_gt_codec[:min_len] ** 2)
        noise_power = np.mean((wav_gt_codec[:min_len] - wav_recon[:min_len]) ** 2)
        snr_db = 10.0 * np.log10(sig_power / (noise_power + 1e-9))
        snr_results[name] = float(snr_db)

        wav_path = os.path.join(out_dir, f"val_spk_{name}_decoded.wav")
        sf.write(wav_path, wav_recon, 24000)
        print(f"✓ [{name}] Decoded Audio: SNR = {snr_db:.2f} dB | Saved: {wav_path}")

    # Save summary report
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_histories": training_histories,
        "validation_evaluation": val_eval_results,
        "snr_results": snr_results
    }
    with open(os.path.join(out_dir, "poc_summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved complete summary metrics: {os.path.join(out_dir, 'poc_summary_metrics.json')}")
    print("=" * 80)
    print("🏁 S2S V2 PROOF-OF-CONCEPT COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
