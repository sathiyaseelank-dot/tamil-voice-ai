#!/usr/bin/env python3
"""
Audio Evaluation, Synthesis, and Metrics Engine.
Computes acoustic quality metrics, duration fidelity, silence ratio,
and interfaces with offline ASR for Tamil Character Error Rate (CER).
"""

import os
import sys
import math
import numpy as np
import soundfile as sf
import torch
from typing import Dict, Any, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def compute_waveform_metrics(wav_pred: np.ndarray, wav_ref: np.ndarray, sr: int = 24000) -> Dict[str, float]:
    """
    Computes time-aligned physical acoustic metrics.
    """
    if wav_pred.ndim > 1:
        wav_pred = wav_pred.squeeze()
    if wav_ref.ndim > 1:
        wav_ref = wav_ref.squeeze()

    min_len = min(len(wav_pred), len(wav_ref))
    dur_err = abs(len(wav_pred) - len(wav_ref)) / float(len(wav_ref)) * 100.0

    p = wav_pred[:min_len]
    r = wav_ref[:min_len]

    # Energy & Peak
    rms_p = float(np.sqrt(np.mean(p ** 2)))
    rms_r = float(np.sqrt(np.mean(r ** 2)))
    peak_p = float(np.max(np.abs(p))) if len(p) > 0 else 0.0

    # Silence ratio (< 1e-4)
    silence_ratio = float(np.mean(np.abs(p) < 1e-4)) * 100.0

    # Cross-correlation time-lag alignment
    best_lag = 0
    if len(p) > 1000 and len(r) > 1000:
        corr = np.correlate(p[:4800], r[:4800], mode="full")
        best_lag = int(np.argmax(corr) - (len(p[:4800]) - 1))

    if best_lag > 0 and len(p) > best_lag:
        p_aligned = p[best_lag:]
        r_aligned = r[:len(p_aligned)]
    elif best_lag < 0 and len(r) > abs(best_lag):
        r_aligned = r[abs(best_lag):]
        p_aligned = p[:len(r_aligned)]
    else:
        p_aligned = p
        r_aligned = r

    # Time-aligned SNR
    noise = r_aligned - p_aligned
    signal_pow = np.sum(r_aligned ** 2) + 1e-10
    noise_pow = np.sum(noise ** 2) + 1e-10
    snr = float(10.0 * np.log10(signal_pow / noise_pow))

    # Scale-Invariant SDR (SI-SDR)
    # alpha = <p, r> / ||r||^2
    dot = np.dot(p_aligned, r_aligned)
    r_norm_sq = np.dot(r_aligned, r_aligned) + 1e-10
    s_target = (dot / r_norm_sq) * r_aligned
    e_noise = p_aligned - s_target
    target_pow = np.sum(s_target ** 2) + 1e-10
    e_noise_pow = np.sum(e_noise ** 2) + 1e-10
    si_sdr = float(10.0 * np.log10(target_pow / e_noise_pow))

    return {
        "duration_error_pct": dur_err,
        "rms_pred": rms_p,
        "rms_ref": rms_r,
        "peak_pred": peak_p,
        "silence_ratio_pct": silence_ratio,
        "time_lag_samples": best_lag,
        "snr_db": snr,
        "si_sdr_db": si_sdr
    }


def compute_levenshtein_cer(pred_text: str, ref_text: str) -> float:
    """
    Computes Character Error Rate (CER) via Levenshtein distance.
    Used purely offline for evaluation.
    """
    p = pred_text.strip()
    r = ref_text.strip()
    if not r:
        return 0.0 if not p else 100.0

    dp = np.zeros((len(r) + 1, len(p) + 1), dtype=int)
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(p) + 1):
        dp[0][j] = j

    for i in range(1, len(r) + 1):
        for j in range(1, len(p) + 1):
            if r[i - 1] == p[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    dist = dp[len(r)][len(p)]
    return float(dist / float(len(r))) * 100.0
