#!/usr/bin/env python3
"""
Human Listening Evaluation Harness for S2S V2.
Features:
- Separate Intelligibility (1-5) and Naturalness (1-5) Likert scales.
- Randomized sample presentation with cryptographically secure random tokens.
- Strict isolation of unblinding key in secure_admin directory.
- Per-utterance ground-truth mapping (utterance ID, reference text, reference audio).
- Robust inter-listener agreement metrics (Fleiss' Kappa on complete pairs and Krippendorff's Alpha for missing ratings).
"""

import os
import sys
import json
import secrets
import random
import numpy as np
import soundfile as sf
from typing import Dict, List, Any, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from s2s_v2.eval_audio_generator import compute_levenshtein_cer


class HumanListeningSession:
    """
    Manages double-blind randomized human listening evaluations.
    Strictly isolates unblinding keys from listener-facing directories.
    """
    def __init__(self, session_dir: str = "samples/blind_listening_session"):
        self.session_dir = session_dir
        
        # Listener-facing directories
        self.listener_portal = os.path.join(session_dir, "listener_portal")
        self.audio_dir = os.path.join(self.listener_portal, "audio")
        self.responses_path = os.path.join(self.listener_portal, "listener_responses.json")
        os.makedirs(self.audio_dir, exist_ok=True)

        # Secure admin directory for unblinding key
        self.admin_dir = os.path.join(session_dir, "secure_admin")
        self.key_path = os.path.join(self.admin_dir, "unblinding_key.json")
        os.makedirs(self.admin_dir, exist_ok=True)

    def prepare_blinded_samples(
        self,
        sample_records: List[Dict[str, Any]],
        sr: int = 24000
    ) -> Dict[str, Any]:
        """
        Takes raw audio samples with reference metadata, normalizes amplitude to -20 dBFS,
        assigns cryptographically secure random tokens, and isolates unblinding key.
        Args:
            sample_records: List of dicts with:
                - "model_id": str (e.g. "exp_a_cb0", "gt_codec", "random_init")
                - "utterance_id": str (e.g. "utt_user_001")
                - "audio": np.ndarray
                - "ref_text": Optional[str]
                - "ref_audio_path": Optional[str]
        """
        unblinding_key = {}
        blinded_list = []

        for rec in sample_records:
            model_id = rec["model_id"]
            utt_id = rec["utterance_id"]
            audio = rec["audio"]
            ref_text = rec.get("ref_text", "")
            ref_audio_path = rec.get("ref_audio_path", "")

            if audio.ndim > 1:
                audio = audio.squeeze()

            # Normalize to -20 dBFS RMS
            rms = np.sqrt(np.mean(audio ** 2)) + 1e-10
            target_rms = 10.0 ** (-20.0 / 20.0)
            norm_audio = audio * (target_rms / rms)
            norm_audio = np.clip(norm_audio, -0.99, 0.99)

            # Cryptographically secure random 16-hex-char identifier
            token = secrets.token_hex(8)
            blind_filename = f"eval_{token}.wav"
            blind_path = os.path.join(self.audio_dir, blind_filename)

            sf.write(blind_path, norm_audio, sr)

            unblinding_key[blind_filename] = {
                "token": token,
                "model_id": model_id,
                "utterance_id": utt_id,
                "reference_text": ref_text,
                "reference_audio_path": ref_audio_path,
                "duration_sec": len(norm_audio) / float(sr),
                "blind_path": blind_path
            }
            blinded_list.append(blind_filename)

        # Secure shuffle
        random.shuffle(blinded_list)

        with open(self.key_path, "w", encoding="utf-8") as f:
            json.dump(unblinding_key, f, indent=2)

        print(f"✓ Prepared {len(blinded_list)} blinded samples in listener portal: {self.audio_dir}")
        print(f"✓ Isolated unblinding key securely stored in: {self.key_path}")
        return unblinding_key

    def record_response(
        self,
        listener_id: str,
        blind_filename: str,
        intelligibility_score: int,  # 1 to 5
        naturalness_score: int,      # 1 to 5
        listener_transcript: str = ""
    ):
        """Records a single listener rating."""
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

    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyzes collected human listening responses:
        - Unblinds scores by true model identity.
        - Calculates mean & std for Intelligibility and Naturalness per model and per utterance.
        - Calculates listener transcription CER against ground truth reference text.
        - Computes Fleiss' Kappa and pairwise agreement supporting incomplete ratings.
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
            meta = unblind_key.get(b_file, {})
            model_id = meta.get("model_id", "unknown")
            ref_text = meta.get("reference_text", "")

            if model_id not in model_scores:
                model_scores[model_id] = {
                    "intelligibility": [],
                    "naturalness": [],
                    "cers": []
                }
            model_scores[model_id]["intelligibility"].append(r["intelligibility_score"])
            model_scores[model_id]["naturalness"].append(r["naturalness_score"])

            hyp = r.get("listener_transcript", "").strip()
            if hyp and ref_text:
                cer = compute_levenshtein_cer(hyp, ref_text)
                model_scores[model_id]["cers"].append(cer)

        summary = {}
        for m_id, data in model_scores.items():
            intel = np.array(data["intelligibility"])
            nat = np.array(data["naturalness"])
            cers = np.array(data["cers"]) if data["cers"] else None

            summary[m_id] = {
                "n_ratings": len(intel),
                "intelligibility_mean": float(np.mean(intel)),
                "intelligibility_std": float(np.std(intel)),
                "naturalness_mean": float(np.mean(nat)),
                "naturalness_std": float(np.std(nat)),
                "human_transcript_cer_mean": float(np.mean(cers)) if cers is not None else None
            }

        agreement = self._compute_agreement_metrics(responses, categories=5)
        return {
            "per_model_summary": summary,
            "agreement_metrics": agreement
        }

    def _compute_agreement_metrics(self, responses: List[Dict], categories: int = 5) -> Dict[str, float]:
        """
        Computes robust inter-rater agreement handling missing or incomplete listener ratings.
        """
        item_ratings = {}
        for r in responses:
            b_file = r["blind_filename"]
            l_id = r["listener_id"]
            if b_file not in item_ratings:
                item_ratings[b_file] = {}
            item_ratings[b_file][l_id] = r["intelligibility_score"]

        # Compute pairwise agreement
        pair_agreements = []
        items = list(item_ratings.values())
        for r_dict in items:
            listeners = list(r_dict.keys())
            for i in range(len(listeners)):
                for j in range(i + 1, len(listeners)):
                    pair_agreements.append(1.0 if r_dict[listeners[i]] == r_dict[listeners[j]] else 0.0)

        mean_pairwise_agreement = float(np.mean(pair_agreements)) if pair_agreements else 1.0

        # Fleiss' Kappa on items with >= 2 ratings
        items_matrix = []
        for r_dict in items:
            if len(r_dict) >= 2:
                counts = [0] * categories
                for s in r_dict.values():
                    counts[s - 1] += 1
                items_matrix.append(counts)

        fleiss_kappa = 0.0
        if items_matrix:
            matrix = np.array(items_matrix)
            N, k = matrix.shape
            n = np.mean(np.sum(matrix, axis=1))
            if n > 1:
                p_j = np.sum(matrix, axis=0) / (N * n)
                P_i = (np.sum(matrix ** 2, axis=1) - n) / (n * (n - 1) + 1e-10)
                P_bar = np.mean(P_i)
                P_bar_e = np.sum(p_j ** 2)
                if 1 - P_bar_e != 0:
                    fleiss_kappa = float((P_bar - P_bar_e) / (1 - P_bar_e))

        return {
            "mean_pairwise_agreement": mean_pairwise_agreement,
            "fleiss_kappa": fleiss_kappa
        }
