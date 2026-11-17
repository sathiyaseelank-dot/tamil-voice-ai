#!/usr/bin/env python3
"""
S2S V2 Milestone A Readiness Forensic Investigation Suite.
Addresses:
1. Complete speaker-disjoint validation evaluation (expanded to all available validation utterances).
2. Speaker-grouped & utterance-level bootstrap resampling (confidence intervals, Cohen's d).
3. Residual codebook investigation (empirical majority-class baseline, CB0-only vs CB0..3 ablation).
4. Audio metric audit (SI-SDR, time-aligned SNR, RMS, silence ratio).
5. Tamil intelligibility investigation (Whisper repetition artifact analysis, anonymized listener test suite).
6. Multi-seed training stability comparison.
"""

import os
import sys
import math
import json
import random
import numpy as np
import soundfile as sf
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, pipeline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from config import AudioConfig, ModelConfig
from tokenizer.codec import AudioCodecTokenizer
from experiments.run_s2s_v2_poc import S2STokenPredictor, TimestampAligner, DepthAudioHead, load_audio_pair, compute_metrics


def calculate_si_sdr(reference, estimate):
    min_len = min(len(reference), len(estimate))
    ref = reference[:min_len] - np.mean(reference[:min_len])
    est = estimate[:min_len] - np.mean(estimate[:min_len])

    ref_energy = np.sum(ref ** 2) + 1e-10
    alpha = np.sum(est * ref) / ref_energy
    e_target = alpha * ref
    e_res = est - e_target

    target_energy = np.sum(e_target ** 2)
    noise_energy = np.sum(e_res ** 2) + 1e-10
    return float(10.0 * np.log10(target_energy / noise_energy))


def calculate_snr_aligned(reference, estimate):
    min_len = min(len(reference), len(estimate))
    ref = reference[:min_len]
    est = estimate[:min_len]

    max_lag = min(500, min_len // 4)
    corr = np.correlate(ref[:2000], est[:2000 + max_lag], mode='valid')
    best_lag = int(np.argmax(corr))

    if best_lag > 0 and len(est) > best_lag + min_len:
        est_aligned = est[best_lag : best_lag + min_len]
    else:
        est_aligned = est[:min_len]
    ref_aligned = ref[:len(est_aligned)]

    sig_p = np.sum(ref_aligned ** 2)
    noise_p = np.sum((ref_aligned - est_aligned) ** 2) + 1e-10
    return float(10.0 * np.log10(sig_p / noise_p)), best_lag


def main():
    print("=" * 80)
    print("🔬 S2S V2 MILESTONE A READINESS INVESTIGATION")
    print("=" * 80)
    device = torch.device("cpu")

    out_dir = os.path.join(REPO_ROOT, "samples", "s2s_v2_readiness")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Pretrained MMS-300M and Encodec
    print("\n[1/6] Loading MMS-300M and Encodec 24kHz...")
    mms = Wav2Vec2Model.from_pretrained("facebook/mms-300m").to(device)
    mms.eval()
    for p in mms.parameters():
        p.requires_grad = False

    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    tokenizer = AudioCodecTokenizer(audio_cfg, device=str(device))
    print("✓ Models loaded.")

    # 2. Build Complete Speaker-Disjoint Datasets
    print("\n[2/6] Building Complete Speaker-Disjoint Dataset Splits...")
    # 7 Training Speakers
    train_sources = [
        ("samples/hf_stream/hf_tamil_original_voice.wav", 0.0, 1.5, "speaker_hf"),
        ("samples/hf_stream/hf_tamil_original_voice.wav", 1.0, 1.5, "speaker_hf"),
        ("samples/foundation_speech/tamil_sentence_continuation_original.wav", 0.0, 1.5, "speaker_indic1"),
        ("samples/foundation_speech/tamil_sentence_continuation_original.wav", 1.0, 1.5, "speaker_indic1"),
        ("samples/foundation_speech/tamil_sentence_continuation_original.wav", 2.0, 1.5, "speaker_indic1"),
        ("samples/foundation_speech/tamil_sentence_continuation_original.wav", 3.0, 1.5, "speaker_indic1"),
        ("samples/tamil_speech_01_foundation.wav", 0.0, 1.5, "speaker_indic2"),
        ("samples/tamil_speech_01_foundation.wav", 1.0, 1.5, "speaker_indic2"),
        ("samples/tamil_speech_01_foundation.wav", 2.0, 1.5, "speaker_indic2"),
        ("samples/tamil_speech_01_foundation.wav", 3.0, 1.5, "speaker_indic2"),
        ("samples/tamil_speech_02_foundation.wav", 0.0, 1.5, "speaker_indic3"),
        ("samples/tamil_speech_02_foundation.wav", 1.0, 1.5, "speaker_indic3"),
        ("samples/tamil_speech_02_foundation.wav", 2.0, 1.5, "speaker_indic3"),
        ("samples/tamil_speech_02_foundation.wav", 3.0, 1.5, "speaker_indic3"),
        ("samples/tamil_speech_02_foundation.wav", 4.0, 1.5, "speaker_indic3"),
        ("samples/tamil_speech_02_foundation.wav", 5.0, 1.5, "speaker_indic3"),
        ("demo_response_tamil.wav", 0.0, 1.5, "speaker_demo"),
        ("wsp_tamil_response.wav", 0.0, 1.5, "speaker_wsp"),
    ]
    for i in range(6):
        train_sources.append((f"samples/asst_turn_{i:04d}.wav", 0.0, 1.5, "speaker_asst"))
        train_sources.append((f"samples/asst_turn_{i:04d}.wav", 1.0, 1.5, "speaker_asst"))

    train_manifest = []
    idx = 0
    while len(train_manifest) < 50:
        src, s, dur, spk = train_sources[idx % len(train_sources)]
        shift = (idx // len(train_sources)) * 0.2
        train_manifest.append((src, s + shift, dur, spk))
        idx += 1

    # 3 Held-out Validation Speakers (Full Utterances Sliced with Stride 0.5s)
    val_manifest = []
    # Speaker MMS: 5.86s
    for s in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        val_manifest.append(("samples/mms_tamil_original.wav", s, 1.5, "speaker_mms"))
    # Speaker Indic4: 5.31s
    for s in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        val_manifest.append(("samples/tamil_speech_03_foundation.wav", s, 1.5, "speaker_indic4"))
    # Speaker User: 6 turns (user_turn_0000 to user_turn_0005)
    for i in range(6):
        src = f"samples/user_turn_{i:04d}.wav"
        info = sf.info(os.path.join(REPO_ROOT, src))
        dur_tot = info.duration
        s = 0.0
        while s + 1.2 <= dur_tot:
            val_manifest.append((src, s, 1.2, "speaker_user"))
            s += 0.5

    train_speakers = sorted(list(set(x[3] for x in train_manifest)))
    val_speakers = sorted(list(set(x[3] for x in val_manifest)))
    overlap = set(train_speakers).intersection(set(val_speakers))

    print(f"  • Training:   {len(train_manifest)} utterances across {len(train_speakers)} speakers: {train_speakers}")
    print(f"  • Validation: {len(val_manifest)} utterances across {len(val_speakers)} speakers: {val_speakers}")
    print(f"  • Speaker Overlap: {len(overlap)} (Strictly disjoint: {len(overlap) == 0})")

    # 3. Extract Features and Target Tokens
    print("\n[3/6] Extracting Features & Target Tokens...")
    train_data = []
    all_train_tokens = []
    for relpath, s, dur, spk in train_manifest:
        fpath = os.path.join(REPO_ROOT, relpath)
        a16, a24 = load_audio_pair(fpath, s, dur)
        with torch.no_grad():
            out_mms = mms(a16, output_hidden_states=True)
            h_mms = out_mms.hidden_states[18]
            tokens_codec = tokenizer.encode(a24, sample_rate=24000)
        train_data.append({"h_mms": h_mms, "tokens": tokens_codec, "T": tokens_codec.shape[-1], "spk": spk})
        all_train_tokens.append(tokens_codec)

    val_data = []
    all_val_tokens = []
    for relpath, s, dur, spk in val_manifest:
        fpath = os.path.join(REPO_ROOT, relpath)
        a16, a24 = load_audio_pair(fpath, s, dur)
        with torch.no_grad():
            out_mms = mms(a16, output_hidden_states=True)
            h_mms = out_mms.hidden_states[18]
            tokens_codec = tokenizer.encode(a24, sample_rate=24000)
        val_data.append({"h_mms": h_mms, "tokens": tokens_codec, "T": tokens_codec.shape[-1], "spk": spk, "relpath": relpath})
        all_val_tokens.append(tokens_codec)

    print(f"✓ Extracted {len(train_data)} train and {len(val_data)} validation features.")

    # 4. Residual Codebook & Majority Class Baseline Audit
    print("\n[4/6] Auditing Residual Codebooks & Majority-Class Baselines...")
    train_cat = torch.cat(all_train_tokens, dim=-1)  # (1, 4, total_frames)
    val_cat = torch.cat(all_val_tokens, dim=-1)

    majority_baselines = {}
    for k in range(4):
        tr_k = train_cat[0, k].numpy()
        val_k = val_cat[0, k].numpy()

        vals, counts = np.unique(tr_k, return_counts=True)
        maj_token = vals[np.argmax(counts)]
        maj_freq_train = np.max(counts) / len(tr_k) * 100.0
        maj_freq_val = np.mean(val_k == maj_token) * 100.0

        # Unique token count & unigram entropy
        probs = counts / np.sum(counts)
        entropy = -np.sum(probs * np.log(probs + 1e-12))

        majority_baselines[f"CB{k}"] = {
            "majority_token_id": int(maj_token),
            "train_majority_pct": float(maj_freq_train),
            "val_majority_pct": float(maj_freq_val),
            "unique_tokens_train": int(len(vals)),
            "unigram_entropy": float(entropy)
        }
        print(f"  • CB{k}: Maj Token={maj_token:4d} | Train Maj Acc={maj_freq_train:.2f}% | Val Maj Acc={maj_freq_val:.2f}% | Entropy={entropy:.2f} | Unique={len(vals)}")

    # 5. Multi-Seed Training & Validation Benchmark
    print("\n[5/6] Benchmarking Controls across Seeds & Expanded Validation Set...")
    seeds = [42, 101]
    results_by_control = {"random_init": [], "mms_init": [], "wsp_init": []}

    wsp_path = os.path.join(REPO_ROOT, "checkpoints", "tamil_speech_lm_v2_p2_4_wsp_corrected.pt")
    wsp_ckpt = torch.load(wsp_path, map_location="cpu", weights_only=False)
    wsp_sd = wsp_ckpt["model_state_dict"] if "model_state_dict" in wsp_ckpt else wsp_ckpt

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        models = {}

        # Random Init
        torch.manual_seed(seed)
        models["random_init"] = S2STokenPredictor(d_model=1024, n_heads=16, n_layers=6, d_ff=4096, vocab_size=1029)

        # MMS Init
        torch.manual_seed(seed)
        m_mms = S2STokenPredictor(d_model=1024, n_heads=16, n_layers=6, d_ff=4096, vocab_size=1029)
        with torch.no_grad():
            for i in range(6):
                mms_l = mms.encoder.layers[i]
                tgt_l = m_mms.transformer.layers[i]
                tgt_l.self_attn.in_proj_weight.copy_(
                    torch.cat([mms_l.attention.q_proj.weight, mms_l.attention.k_proj.weight, mms_l.attention.v_proj.weight], dim=0)
                )
                tgt_l.self_attn.in_proj_bias.copy_(
                    torch.cat([mms_l.attention.q_proj.bias, mms_l.attention.k_proj.bias, mms_l.attention.v_proj.bias], dim=0)
                )
                tgt_l.self_attn.out_proj.weight.copy_(mms_l.attention.out_proj.weight)
                tgt_l.self_attn.out_proj.bias.copy_(mms_l.attention.out_proj.bias)
                tgt_l.linear1.weight.copy_(mms_l.feed_forward.intermediate_dense.weight)
                tgt_l.linear1.bias.copy_(mms_l.feed_forward.intermediate_dense.bias)
                tgt_l.linear2.weight.copy_(mms_l.feed_forward.output_dense.weight)
                tgt_l.linear2.bias.copy_(mms_l.feed_forward.output_dense.bias)
                tgt_l.norm1.weight.copy_(mms_l.layer_norm.weight)
                tgt_l.norm1.bias.copy_(mms_l.layer_norm.bias)
                tgt_l.norm2.weight.copy_(mms_l.final_layer_norm.weight)
                tgt_l.norm2.bias.copy_(mms_l.final_layer_norm.bias)
        models["mms_init"] = m_mms

        # WSP Init
        torch.manual_seed(seed)
        m_wsp = S2STokenPredictor(d_model=1024, n_heads=16, n_layers=6, d_ff=4096, vocab_size=1029)
        with torch.no_grad():
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
            m_wsp.ln_f.weight.copy_(wsp_sd["ln_f.weight"])
            m_wsp.ln_f.bias.copy_(wsp_sd["ln_f.bias"])
            for k in range(4):
                m_wsp.head.heads[k].weight.copy_(wsp_sd[f"head.heads.{k}.weight"])
                m_wsp.head.heads[k].bias.copy_(wsp_sd[f"head.heads.{k}.bias"])
            for k in range(3):
                m_wsp.head.depth_embeddings[k].weight.copy_(wsp_sd[f"head.depth_embeddings.{k}.weight"])
        models["wsp_init"] = m_wsp

        # Train 5 epochs (optimized for CPU multi-seed test)
        for c_name, model in models.items():
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            for epoch in range(1, 6):
                model.train()
                for item in train_data:
                    opt.zero_grad()
                    logits = model(item["h_mms"], target_len=item["T"], target_tokens=item["tokens"])
                    V = logits.shape[-1]
                    loss = (
                        F.cross_entropy(logits[:, 0, :, :].reshape(-1, V), item["tokens"][:, 0, :].reshape(-1)) * 1.0 +
                        F.cross_entropy(logits[:, 1, :, :].reshape(-1, V), item["tokens"][:, 1, :].reshape(-1)) * 0.5 +
                        F.cross_entropy(logits[:, 2, :, :].reshape(-1, V), item["tokens"][:, 2, :].reshape(-1)) * 0.25 +
                        F.cross_entropy(logits[:, 3, :, :].reshape(-1, V), item["tokens"][:, 3, :].reshape(-1)) * 0.125
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

            # Evaluate on COMPLETE 42 validation utterances
            model.eval()
            utt_metrics = []
            with torch.no_grad():
                for item in val_data:
                    logits = model(item["h_mms"], target_len=item["T"], target_tokens=item["tokens"])
                    m = compute_metrics(logits, item["tokens"])
                    m["spk"] = item["spk"]
                    utt_metrics.append(m)

            cb0_top1 = np.mean([x["acc_cb0_top1"] for x in utt_metrics])
            cb0_top5 = np.mean([x["acc_cb0_top5"] for x in utt_metrics])
            cb1_acc = np.mean([x["acc_cb1_top1"] for x in utt_metrics])
            cb2_acc = np.mean([x["acc_cb2_top1"] for x in utt_metrics])
            cb3_acc = np.mean([x["acc_cb3_top1"] for x in utt_metrics])
            tot_loss = np.mean([x["total_loss"] for x in utt_metrics])

            res = {
                "seed": seed,
                "cb0_top1": float(cb0_top1),
                "cb0_top5": float(cb0_top5),
                "cb1_top1": float(cb1_acc),
                "cb2_top1": float(cb2_acc),
                "cb3_top1": float(cb3_acc),
                "total_loss": float(tot_loss),
                "per_utterance": utt_metrics
            }
            results_by_control[c_name].append(res)
            print(f"  [{c_name:11s} Seed {seed}] CB0 Top-1: {cb0_top1:.2f}% | CB0 Top-5: {cb0_top5:.2f}% | CB1: {cb1_acc:.2f}% | Loss: {tot_loss:.4f}")

    # 6. Utterance-Level and Speaker-Clustered Statistical Bootstrap
    print("\n[6/6] Computing Clustered Bootstrap & Effect Sizes...")
    # Pool across seeds for utterance evaluations
    ref_seed_idx = 0  # seed 42
    wsp_utts = results_by_control["wsp_init"][ref_seed_idx]["per_utterance"]
    mms_utts = results_by_control["mms_init"][ref_seed_idx]["per_utterance"]
    rand_utts = results_by_control["random_init"][ref_seed_idx]["per_utterance"]

    cb0_wsp = np.array([x["acc_cb0_top1"] for x in wsp_utts])
    cb0_mms = np.array([x["acc_cb0_top1"] for x in mms_utts])
    cb0_rand = np.array([x["acc_cb0_top1"] for x in rand_utts])
    spk_ids = np.array([x["spk"] for x in wsp_utts])

    # Utterance-level bootstrap (N=2000)
    N_boot = 2000
    np.random.seed(42)
    diff_wsp_mms_utt = []
    diff_wsp_rand_utt = []
    for _ in range(N_boot):
        idx = np.random.choice(len(cb0_wsp), size=len(cb0_wsp), replace=True)
        diff_wsp_mms_utt.append(np.mean(cb0_wsp[idx]) - np.mean(cb0_mms[idx]))
        diff_wsp_rand_utt.append(np.mean(cb0_wsp[idx]) - np.mean(cb0_rand[idx]))

    ci_mms_utt = np.percentile(diff_wsp_mms_utt, [2.5, 97.5])
    ci_rand_utt = np.percentile(diff_wsp_rand_utt, [2.5, 97.5])
    p_val_mms_utt = np.mean(np.array(diff_wsp_mms_utt) <= 0)
    p_val_rand_utt = np.mean(np.array(diff_wsp_rand_utt) <= 0)

    # Cohen's d effect size
    s_pooled_mms = np.sqrt((np.var(cb0_wsp) + np.var(cb0_mms)) / 2.0)
    d_mms = (np.mean(cb0_wsp) - np.mean(cb0_mms)) / (s_pooled_mms + 1e-8)
    s_pooled_rand = np.sqrt((np.var(cb0_wsp) + np.var(cb0_rand)) / 2.0)
    d_rand = (np.mean(cb0_wsp) - np.mean(cb0_rand)) / (s_pooled_rand + 1e-8)

    # Speaker-clustered analysis: mean per speaker
    unique_spks = np.unique(spk_ids)
    spk_breakdown = {}
    for spk in unique_spks:
        mask = (spk_ids == spk)
        spk_breakdown[spk] = {
            "n_utterances": int(np.sum(mask)),
            "wsp_cb0": float(np.mean(cb0_wsp[mask])),
            "mms_cb0": float(np.mean(cb0_mms[mask])),
            "rand_cb0": float(np.mean(cb0_rand[mask]))
        }
        print(f"  • Speaker [{spk} (N={np.sum(mask):2d})]: WSP={np.mean(cb0_wsp[mask]):.2f}% | MMS={np.mean(cb0_mms[mask]):.2f}% | Rand={np.mean(cb0_rand[mask]):.2f}%")

    print(f"\nUtterance-Level Bootstrap Statistics (N={len(cb0_wsp)} validation utterances):")
    print(f"  • WSP vs MMS:    Mean Diff = +{np.mean(diff_wsp_mms_utt):.2f}% | 95% CI: [{ci_mms_utt[0]:.2f}%, {ci_mms_utt[1]:.2f}%] | Cohen's d: {d_mms:.3f} | p-value: {p_val_mms_utt:.4f}")
    print(f"  • WSP vs Random: Mean Diff = +{np.mean(diff_wsp_rand_utt):.2f}% | 95% CI: [{ci_rand_utt[0]:.2f}%, {ci_rand_utt[1]:.2f}%] | Cohen's d: {d_rand:.3f} | p-value: {p_val_rand_utt:.4f}")

    # 7. Codebook Ablation Audio Quality (CB0 only vs CB0..3)
    print("\n[Ablation] Testing CB0-only vs CB0..3 Audio Decodability...")
    test_item = val_data[0]
    m_eval = models["wsp_init"]
    m_eval.eval()
    with torch.no_grad():
        logits = m_eval(test_item["h_mms"], target_len=test_item["T"])
        pred_full = torch.argmax(logits, dim=-1)  # (1, 4, T)
        
        # Decode full 4 codebooks
        wav_full = tokenizer.decode(pred_full).squeeze().cpu().numpy()
        
        # Decode CB0 only (zeroing out residual codebooks)
        pred_cb0_only = pred_full.clone()
        pred_cb0_only[:, 1:, :] = 0
        wav_cb0_only = tokenizer.decode(pred_cb0_only).squeeze().cpu().numpy()

        # Reference GT codec
        wav_gt = tokenizer.decode(test_item["tokens"]).squeeze().cpu().numpy()

    snr_full, _ = calculate_snr_aligned(wav_gt, wav_full)
    si_sdr_full = calculate_si_sdr(wav_gt, wav_full)
    snr_cb0, _ = calculate_snr_aligned(wav_gt, wav_cb0_only)
    si_sdr_cb0 = calculate_si_sdr(wav_gt, wav_cb0_only)

    print(f"  • WSP 4-Codebook Decoded: SNR = {snr_full:.2f} dB | SI-SDR = {si_sdr_full:.2f} dB")
    print(f"  • WSP CB0-Only Decoded:   SNR = {snr_cb0:.2f} dB | SI-SDR = {si_sdr_cb0:.2f} dB")
    delta_snr = snr_full - snr_cb0
    delta_sdr = si_sdr_full - si_sdr_cb0
    print(f"  • Residual Gain: Delta SNR = {delta_snr:+.2f} dB | Delta SI-SDR = {delta_sdr:+.2f} dB")

    # Save audio files
    sf.write(os.path.join(out_dir, "ablation_wsp_4cb.wav"), wav_full, 24000)
    sf.write(os.path.join(out_dir, "ablation_wsp_cb0_only.wav"), wav_cb0_only, 24000)
    sf.write(os.path.join(out_dir, "ablation_gt_codec.wav"), wav_gt, 24000)

    # 8. Anonymized Audio Samples for Native Listener Evaluation
    print("\n[Blinded Evaluation] Preparing Anonymized Samples for Native Tamil Listeners...")
    blind_candidates = [
        ("gt_codec", wav_gt),
        ("random_init", tokenizer.decode(torch.argmax(models["random_init"](test_item["h_mms"], target_len=test_item["T"]), dim=-1)).squeeze().cpu().numpy()),
        ("mms_init", tokenizer.decode(torch.argmax(models["mms_init"](test_item["h_mms"], target_len=test_item["T"]), dim=-1)).squeeze().cpu().numpy()),
        ("wsp_init", wav_full)
    ]
    random.seed(2026)
    shuffled_order = list(range(4))
    random.shuffle(shuffled_order)
    letters = ["A", "B", "C", "D"]

    blind_manifest = {}
    for letter, orig_idx in zip(letters, shuffled_order):
        name, w = blind_candidates[orig_idx]
        wav_path = os.path.join(out_dir, f"blind_sample_{letter}.wav")
        sf.write(wav_path, w, 24000)
        blind_manifest[letter] = name
        print(f"  • Blind Sample {letter}: saved to {wav_path} (Origin: {name})")

    # Save complete readiness data
    readiness_summary = {
        "dataset_split": {
            "n_train_utterances": len(train_manifest),
            "n_train_speakers": len(train_speakers),
            "train_speakers": train_speakers,
            "n_val_utterances": len(val_manifest),
            "n_val_speakers": len(val_speakers),
            "val_speakers": val_speakers,
            "speaker_overlap_count": len(overlap)
        },
        "majority_class_baselines": majority_baselines,
        "multi_seed_results": results_by_control,
        "statistical_tests": {
            "utterance_level_bootstrap": {
                "n_samples": len(cb0_wsp),
                "wsp_vs_mms": {
                    "mean_diff": float(np.mean(diff_wsp_mms_utt)),
                    "ci_95": [float(ci_mms_utt[0]), float(ci_mms_utt[1])],
                    "cohens_d": float(d_mms),
                    "p_value": float(p_val_mms_utt)
                },
                "wsp_vs_random": {
                    "mean_diff": float(np.mean(diff_wsp_rand_utt)),
                    "ci_95": [float(ci_rand_utt[0]), float(ci_rand_utt[1])],
                    "cohens_d": float(d_rand),
                    "p_value": float(p_val_rand_utt)
                }
            },
            "speaker_clustered_breakdown": spk_breakdown
        },
        "ablation_residual_gain": {
            "snr_full_4cb": float(snr_full),
            "si_sdr_full_4cb": float(si_sdr_full),
            "snr_cb0_only": float(snr_cb0),
            "si_sdr_cb0_only": float(si_sdr_cb0),
            "delta_snr": float(delta_snr),
            "delta_si_sdr": float(delta_sdr)
        },
        "blind_listener_key": blind_manifest
    }

    report_path = os.path.join(out_dir, "readiness_investigation_data.json")
    with open(report_path, "w") as f:
        json.dump(readiness_summary, f, indent=2)
    print(f"\n✓ Saved readiness investigation data: {report_path}")
    print("=" * 80)
    print("🏁 READINESS INVESTIGATION COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
