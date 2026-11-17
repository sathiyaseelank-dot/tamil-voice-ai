# S2S V2 DATASET MANIFEST AUDIT & SPEAKER RECONCILIATION

**Document ID:** AUDIT-S2S-V2-MANIFEST-002  
**Target Repository:** `tamil-voice-ai`  
**Execution Date:** 2026-11-17  
**Status:** **TIER 1 AUDIT FAILED (MISSING UNDERLYING RECORDS) / TIER 2 VERIFIED (0.00% OVERLAP)**  

---

## 1. Executive Summary & Reconciliation

Following strict technical review of dataset split claims, a formal audit of dataset manifests and underlying records was conducted to verify cross-split speaker disjointness.

### Forensic Reconciliation Finding:
1. **Tier 1 — Full Production Training Corpus (IndicVoices-R Tamil):**
   - **Status:** **`AUDIT_FAILED_MISSING_UNDERLYING_SPEAKER_RECORDS`**
   - **Reason:** The manifest at `data/v2_p2_4/v2_p2_4_manifest.json` contains aggregate scalar summary fields (`train_speakers_count: 382`, `val_speakers_count: 25`, `train_items: 7679`, `val_items: 684`, `total_audio_hours: 20.73`), but lacks the explicit per-speaker lists (`train_speakers_list`, `val_speakers_list`). Furthermore, the referenced token cache files (`v2_p2_4_train_tokens.pt`, `v2_p2_4_val_tokens.pt`) do not exist on local disk.
   - **Policy Enforcement:** Per P0 review requirements, calculating disjointness against an empty speaker list or accepting unverified aggregate metadata is rejected. Tier 1 cannot be certified until raw per-utterance parquet or speaker metadata is restored and hashed on disk.

2. **Tier 2 — Local Forensic Benchmark Pool (Readiness Investigation):**
   - **Status:** **`VERIFIED_FROM_INVESTIGATION_RECORD` (0.00% Cross-Split Overlap)**
   - **Scope:** Rapid local diagnostic suite executed on cached `.wav` slices in `samples/` and tracked in `samples/s2s_v2_readiness/readiness_investigation_data.json`.
   - **Training Pool:** $50$ utterances across $7$ local speakers (`speaker_asst`, `speaker_demo`, `speaker_hf`, `speaker_indic1`, `speaker_indic2`, `speaker_indic3`, `speaker_wsp`).
   - **Validation Pool:** $32$ utterances across $3$ held-out local speakers (`speaker_indic4`: 7 utterances, `speaker_mms`: 8 utterances, `speaker_user`: 17 utterances).
   - **Intersection:** $\text{Train Speakers} \cap \text{Val Speakers} = \emptyset$. **Zero overlap dynamically computed.**

---

## 2. Granular Tier 1 Corpus Distribution & Status

| Metric | Manifest Value | Local Verification Status |
| :--- | :---: | :--- |
| **Duration (Hours)** | $20.73\text{ h}$ ($18.97\text{ h}$ train / $1.77\text{ h}$ val) | Unverified (audio files not stored locally) |
| **Utterance Count** | $8,363$ ($7,679$ train / $684$ val) | Unverified (per-utterance manifests missing) |
| **Speaker Count** | $407$ ($382$ train / $25$ val) | Unverified (explicit speaker lists missing) |
| **Speaker Overlap** | Claimed $0$ | **FAILED AUDIT: Cannot calculate without explicit records** |
| **Cache Files** | `v2_p2_4_train_tokens.pt`, `v2_p2_4_val_tokens.pt` | **MISSING FROM LOCAL DISK** |

---

## 3. Granular Tier 2 Local Forensic Benchmark Pool

| Speaker ID | Split | Utterance Count | Source File / Sample |
| :--- | :---: | :---: | :--- |
| `speaker_asst` | Train | 6 | `samples/asst_turn_*.wav` |
| `speaker_demo` | Train | 5 | `samples/demo_*.wav` |
| `speaker_hf` | Train | 8 | `samples/hf_tamil_*.wav` |
| `speaker_indic1` | Train | 9 | `samples/indic_sample_01.wav` |
| `speaker_indic2` | Train | 8 | `samples/indic_sample_02.wav` |
| `speaker_indic3` | Train | 7 | `samples/indic_sample_03.wav` |
| `speaker_wsp` | Train | 7 | `samples/wsp_tamil_response.wav` |
| **Total Train** | **Train** | **50** | — |
| `speaker_indic4` | Validation | 7 | `samples/tamil_speech_03_foundation.wav` |
| `speaker_mms` | Validation | 8 | `samples/mms_tamil_original.wav` |
| `speaker_user` | Validation | 17 | `samples/user_turn_0000.wav` to `user_turn_0005.wav` |
| **Total Validation**| **Validation**| **32** | — |

**Intersection:** $\text{Train Speakers} \cap \text{Val Speakers} = \emptyset$. **Zero overlap verified.**

---

## 4. Manifest Audit File Reference
The programmatic JSON manifest is saved at:
[`manifests/manifest_audit_summary.json`](file:///content/tamil-voice-ai/manifests/manifest_audit_summary.json)
