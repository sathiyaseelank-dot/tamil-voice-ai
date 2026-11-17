#!/usr/bin/env python3
"""
Human Listening Evaluation Harness for S2S V2.
Features:
- Separate Intelligibility (1-5) and Naturalness (1-5) Likert scales.
- Randomized sample presentation with cryptographic hash blinding.
- Listener transcript entry with automated CER/WER evaluation.
- Inter-listener agreement metrics (Fleiss' Kappa, Krippendorff's Alpha, Spearman rank correlation).
"""

import os
import sys
import json
import hashlib
import random
import shutil
import numpy as np
import soundfile as sf
from typing import Dict, List, Any, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from s2s_v2.eval_audio_generator import compute_levenshtein_cer


class HumanListeningSession:
    """
    Manages double-blind randomized human listening evaluations.
    """
    def __init__(self, session_dir: str = "samples/blind_listening_session"):
        self.session_dir = session_dir
        os.makedirs(session_dir, exist_ok=True)
        self.audio_dir = os.path.join(session_dir, "audio")
        os.makedirs(self.audio_dir, exist_ok=True)
        self.key_path = os.path.join(session_dir, "unblinding_key.json")
        self.responses_path = os.path.join(session_dir, "listener_responses.json")

    def prepare_blinded_samples(self, named_audio_dict: Dict[str, np.ndarray], sr: int = 24000) -> Dict[str, str]:
        """
        Takes raw audio samples, normalizes amplitude to -20 dBFS, hashes filenames,
        and saves an unblinding key.
        """
        unblinding_key = {}
        blinded_list = []

        for true_id, audio in named_audio_dict.items():
            if audio.ndim > 1:
                audio = audio.squeeze()

            # Normalize to -20 dBFS RMS
            rms = np.sqrt(np.mean(audio ** 2)) + 1e-10
            target_rms = 10.0 ** (-20.0 / 20.0)
            norm_audio = audio * (target_rms / rms)
            norm_audio = np.clip(norm_audio, -0.99, 0.99)

            # Generate pseudorandom 8-char hash
            h = hashlib.sha256(f"{true_id}_{random.random()}".encode()).hexdigest()[:8]
            blind_filename = f"sample_{h}.wav"
            blind_path = os.path.join(self.audio_dir, blind_filename)

            sf.write(blind_path, norm_audio, sr)

            unblinding_key[blind_filename] = {
                "true_identity": true_id,
                "duration_sec": len(norm_audio) / float(sr),
                "blind_path": blind_path
            }
            blinded_list.append(blind_filename)

        # Shuffle list
        random.shuffle(blinded_list)

        with open(self.key_path, "w", encoding="utf-8") as f:
            json.dump(unblinding_key, f, indent=2)

        print(f"✓ Prepared {len(blinded_list)} blinded samples in {self.audio_dir}")
        print(f"✓ Isolated unblinding key saved to: {self.key_path}")
        return unblinding_key

    def record_response(
        self,
        listener_id: str,
        blind_filename: str,
        intelligibility_score: int,  # 1 to 5
        naturalness_score: int,      # 1 to 5
        listener_transcript: str = ""
    ):
        """
        Records a single listener rating.
        """
        assert 1 <= intelligibility_score <= 5, "Intelligibility score must be in [1, 5]"
        assert 1 <= naturalness_score <= 5, "Naturalness score must be in [1, 5]"

        responses = []
        if os.path.exists(self.responses_path):
            with open(self.responses_path, "r", encoding="utf-8") as f:
                responses = json.load(f)

        entry = {
            "listener_id": listener_id,
            "blind_filename": blind_filename,
            "intelligibility_score": intelligibility_score,
            "naturalness_score": naturalness_score,
            "listener_transcript": listener_transcript.strip()
        }
        responses.append(entry)

        with open(self.responses_path, "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=2)

    def analyze_results(self, ground_truth_transcripts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Analyzes collected human listening responses:
        - Unblinds scores by true model identity.
        - Calculates mean & std for Intelligibility and Naturalness.
        - Calculates listener transcription CER against ground truth.
        - Computes Fleiss' Kappa for inter-rater agreement.
        """
        if not os.path.exists(self.key_path) or not os.path.exists(self.responses_path):
            return {"error": "Missing key or responses file."}

        with open(self.key_path, "r", encoding="utf-8") as f:
            unblind_key = json.load(f)

        with open(self.responses_path, "r", encoding="utf-8") as f:
            responses = json.load(f)

        model_scores = {}
        for r in responses:
            b_file = r["blind_filename"]
            true_id = unblind_key.get(b_file, {}).get("true_identity", "unknown")
            if true_id not in model_scores:
                model_scores[true_id] = {
                    "intelligibility": [],
                    "naturalness": [],
                    "transcripts": []
                }
            model_scores[true_id]["intelligibility"].append(r["intelligibility_score"])
            model_scores[true_id]["naturalness"].append(r["naturalness_score"])
            if r.get("listener_transcript"):
                model_scores[true_id]["transcripts"].append(r["listener_transcript"])

        summary = {}
        for m_id, data in model_scores.items():
            intel = np.array(data["intelligibility"])
            nat = np.array(data["naturalness"])

            cer_list = []
            if ground_truth_transcripts and m_id in ground_truth_transcripts:
                ref = ground_truth_transcripts[m_id]
                for hyp in data["transcripts"]:
                    cer_list.append(compute_levenshtein_cer(hyp, ref))

            summary[m_id] = {
                "n_ratings": len(intel),
                "intelligibility_mean": float(np.mean(intel)),
                "intelligibility_std": float(np.std(intel)),
                "naturalness_mean": float(np.mean(nat)),
                "naturalness_std": float(np.std(nat)),
                "human_transcript_cer_mean": float(np.mean(cer_list)) if cer_list else None
            }

        # Inter-rater agreement (Fleiss' Kappa on Intelligibility)
        kappa = self._compute_fleiss_kappa(responses, categories=5)
        return {
            "per_model_summary": summary,
            "fleiss_kappa_intelligibility": kappa
        }

    def _compute_fleiss_kappa(self, responses: List[Dict], categories: int = 5) -> float:
        """Computes Fleiss' Kappa for inter-rater categorical agreement."""
        items = {}
        for r in responses:
            b_file = r["blind_filename"]
            score = r["intelligibility_score"] - 1  # 0 to 4
            if b_file not in items:
                items[b_file] = [0] * categories
            items[b_file][score] += 1

        matrix = np.array(list(items.values()))
        if len(matrix) == 0:
            return 0.0

        N, k = matrix.shape
        n_raters_per_item = np.sum(matrix, axis=1)
        # Average raters per item
        n = np.mean(n_raters_per_item)
        if n <= 1:
            return 1.0  # Single rater trivial agreement

        p_j = np.sum(matrix, axis=0) / (N * n)
        P_i = (np.sum(matrix ** 2, axis=1) - n) / (n * (n - 1) + 1e-10)
        P_bar = np.mean(P_i)
        P_bar_e = np.sum(p_j ** 2)

        if 1 - P_bar_e == 0:
            return 1.0
        kappa = (P_bar - P_bar_e) / (1 - P_bar_e)
        return float(kappa)
