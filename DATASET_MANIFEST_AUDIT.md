# S2S V2 DATASET MANIFEST AUDIT & SPEAKER RECONCILIATION

**Document ID:** AUDIT-S2S-V2-MANIFEST-001  
**Target Repository:** `tamil-voice-ai`  
**Execution Date:** 2026-11-17  
**Status:** **AUDITED & VERIFIED (0.00% SPEAKER OVERLAP)**  

---

## 1. Executive Summary & Reconciliation

During the initial review of the Milestone A readiness reports, an apparent discrepancy was noted between:
1. The **407-speaker dataset audit** conducted in Experiment V2-P2.4 (`data/v2_p2_4/v2_p2_4_manifest.json`), representing $\approx 20.73$ hours of Tamil speech.
2. The **7-training / 3-validation speaker split** cited in `experiments/run_readiness_investigation.py` and `S2S_V2_MILESTONE_A_READINESS_REPORT.md`.

### Forensic Reconciliation Finding:
- **Tier 1 — Full Production Training Corpus (IndicVoices-R Tamil):**
  - **Total Audio:** **$20.73\text{ hours}$** ($74,641\text{ seconds}$).
  - **Total Speakers:** **407 unique Tamil speakers** ($382$ training speakers, $25$ validation speakers).
  - **Total Utterances:** **$8,363$ items** ($7,679$ training utterances, $684$ validation utterances).
  - **Speaker Disjointness:** **Strictly $0.00\%$ cross-split overlap**.
  - **Validation Anchor Speakers:** Evaluation speakers `S4258098500393504` and `S4259865400376309` are $100\%$ isolated in the validation split.

- **Tier 2 — Local Forensic Benchmark Pool (Readiness Investigation):**
  - **Scope:** Rapid local diagnostic suite executed on cached `.wav` slices in `samples/`.
  - **Training Pool:** $50$ utterances across $7$ local speakers (`speaker_asst`, `speaker_demo`, `speaker_hf`, `speaker_indic1`, `speaker_indic2`, `speaker_indic3`, `speaker_wsp`). Duration $\approx 2.5\text{ minutes}$.
  - **Validation Pool:** $32$ utterances across $3$ held-out local speakers (`speaker_indic4`: 7 utterances, `speaker_mms`: 8 utterances, `speaker_user`: 17 utterances). Duration $\approx 1.8\text{ minutes}$.
  - **Speaker Overlap:** **$0.00\%$** (Strictly disjoint).

### Audited Conclusion:
The 20 hours belong exclusively to the **407 speakers of the full IndicVoices-R Tamil corpus (Tier 1)**. The 7-speaker pool (Tier 2) was a micro-benchmark for rapid forensic probes. Going forward, the two tiers are formally separated and tracked in manifests to prevent nomenclature confusion.

---

## 2. Granular Tier 1 Corpus Distribution (IndicVoices-R Tamil)

| Metric | Training Split | Validation Split | Combined Total |
| :--- | :---: | :---: | :---: |
| **Duration (Hours)** | $18.97\text{ h}$ | $1.77\text{ h}$ | **$20.73\text{ h}$** |
| **Duration (Seconds)** | $68,286.4\text{ s}$ | $6,355.1\text{ s}$ | **$74,641.5\text{ s}$** |
| **Utterance Count** | $7,679$ | $684$ | **$8,363$** |
| **Mean Utterance Length** | $8.89\text{ s}$ | $9.29\text{ s}$ | $8.92\text{ s}$ |
| **Speaker Count** | $382$ | $25$ | **$407$** |
| **Speaker Overlap** | — | — | **$0.00\%$ (0 speakers)** |
| **Token Cache File** | `v2_p2_4_train_tokens.pt` ($44.3\text{ MB}$) | `v2_p2_4_val_tokens.pt` ($4.1\text{ MB}$) | $48.5\text{ MB}$ |

### Validation Speaker Cohort:
The 25 held-out validation speakers include the canonical evaluation speakers:
- `S4258098500393504`
- `S4259865400376309`
Plus 23 additional randomly isolated speakers. Zero frames or tokens from these 25 speakers ever enter the training stream.

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

**Intersection:** $\text{Train Speakers} \cap \text{Val Speakers} = \emptyset$. **Zero overlap.**

---

## 4. Manifest Audit File Reference
The programmatic JSON manifest is saved at:
[`manifests/manifest_audit_summary.json`](file:///content/tamil-voice-ai/manifests/manifest_audit_summary.json)
