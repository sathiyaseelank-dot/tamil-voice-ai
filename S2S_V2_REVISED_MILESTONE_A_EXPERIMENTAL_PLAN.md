# S2S V2 REVISED MILESTONE A EXPERIMENTAL PLAN: CONTROLLED THREE-STAGE ARCHITECTURAL VALIDATION

**Document ID:** PLAN-S2S-V2-002  
**Target Repository:** `tamil-voice-ai`  
**Branch:** `feat/s2s-v2-milestone-a`  
**Execution Date:** 2026-11-17  
**Status:** **SUBMITTED FOR REVIEW & APPROVAL (TRAINING HALTED PENDING SIGNOFF)**  

---

## 1. Executive Summary & Forensic Grounding

### 1.1 Forensic Review of Human Listening vs. Objective Metrics
A critical qualitative discrepancy emerged during the native listening assessment of the Milestone A readiness artifacts, prompting an immediate reassessment of our architectural assumptions:

| Audio Stream | Time-Aligned SNR | SI-SDR | Human Listening Assessment | Diagnostic Verdict |
| :--- | :---: | :---: | :--- | :--- |
| **GT Codec Reference** | $0.00\text{ dB}$ | $0.00\text{ dB}$ | **Clearly listenable Tamil speech** | Clean ground-truth ceiling |
| **Random Init (4-CB)** | $-2.54\text{ dB}$ | $-42.08\text{ dB}$ | **Robotic, but identifiable Tamil** | Coarse phonetics + acoustic harmonics intact |
| **MMS Init (4-CB)** | $-2.58\text{ dB}$ | $-59.97\text{ dB}$ | **Robotic, but identifiable Tamil** | Coarse phonetics + acoustic harmonics intact |
| **WSP Init (4-CB)** | $-2.35\text{ dB}$ | $-23.75\text{ dB}$ | **Robotic, but identifiable Tamil** | Coarse phonetics + acoustic harmonics intact |
| **WSP Init (CB0-Only)** | **$-1.92\text{ dB}$** | **$-26.71\text{ dB}$** | **Very difficult to identify** | **Intelligibility collapse despite higher SI-SDR** |

### 1.2 Why SI-SDR Failed as an Intelligibility Metric
In the initial readiness report, zeroing residual codebooks CB1–CB3 produced a **$+8.01\text{ dB}$ gain in SI-SDR** ($-26.71\text{ dB}$ vs. $-34.72\text{ dB}$) and a $+0.67\text{ dB}$ gain in SNR. This was initially interpreted as eliminating phase distortion. However, human perceptual evaluation revealed the reverse:
1. **Residual RVQ Harmonics:** In the Encodec 24 kHz architecture, Codebook 0 (CB0) captures low-frequency vocal tract resonance and broad formant envelopes. However, the residual codebooks (CB1–CB3) carry the high-frequency harmonics, pitch dynamics, and unvoiced consonant transients (fricatives, plosive bursts, sibilants) that are phonetically required to distinguish Tamil syllables (e.g., distinguishing `"நாளிக்கு"` from `"நாலக்கு"`).
2. **SI-SDR Energy Bias:** Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) optimizes for macroscopic energy alignment and penalizes high-frequency variance. When CB1–CB3 are zeroed, the resulting audio is smooth, low-pass, and free of high-frequency variance—which mathematically optimizes SI-SDR while stripping away the high-frequency formant cues necessary for human ear recognition.
3. **WSP CB0 Benchmark Reversal:** In expanded statistical benchmarking across 32 validation utterances, WSP CB0 Top-1 accuracy achieved **$30.52\%$**, falling below `random_init` ($32.90\%$) and `mms_init` ($33.80\%$). The previously observed WSP advantage on the initial 8-utterance probe was not replicated.

### 1.3 Revised Strategy: Controlled Empirical Validation
We **do not assume the two-stage strategy is already validated**. Instead, we establish a controlled three-experiment framework to systematically measure CB0 learnability, residual codebook contribution to intelligibility, and exposure bias mitigation before any full-scale production training run.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   CONTROLLED MILESTONE A EXPERIMENTAL SUITE                      │
└──────────────────────────────────────────────────────────────────────────────────┘
                                          │
    ┌─────────────────────────────────────┼────────────────────────────────────┐
    ▼                                     ▼                                    ▼
┌─────────────────────────┐   ┌─────────────────────────┐   ┌──────────────────────────┐
│      EXPERIMENT A       │   │      EXPERIMENT B       │   │       EXPERIMENT C       │
│    CB0-Only Learning    │   │  Residual RVQ Learning  │   │  Inference Exposure Bias │
├─────────────────────────┤   ├─────────────────────────┤   ├──────────────────────────┤
│• Train 100% loss on CB0 │   │• Freeze trained CB0     │   │• Condition residuals on  │
│• Encodec frozen         │   │• Train CB1..CB3 on GT   │   │  predicted (noisy) CB0   │
│• Checkpoints: 500..5000 │   │• Compare CB0 vs. 4-CB   │   │• Scheduled sampling /    │
│• Measure Intelligibility│   │• Verify if residuals    │   │  Soft-embedding study    │
│  & ASR CER (not SI-SDR) │   │  restore intelligibility│   │• Close train-eval gap    │
└─────────────────────────┘   └─────────────────────────┘   └──────────────────────────┘
```

---

## 2. Experiment A: CB0-Only Learning Architecture & Protocol

### 2.1 Experimental Objective
Determine whether a dedicated Transformer predictor trained with 100% loss concentration on Codebook 0 can achieve high phonetic accuracy ($> 60\%$) on the full streaming dataset (`SPRINGLab/IndicVoices-R_Tamil`), and evaluate whether CB0-only decoded audio can ever reach human Tamil intelligibility without residual assistance.

### 2.2 Mathematical Specification
- **Input Stream:** Continuous MMS-300M features $\mathbf{H}_{\text{mms}} \in \mathbb{R}^{B \times T_{\text{mms}} \times 1024}$ extracted at $50\text{ Hz}$ ($16\text{ kHz}$ speech).
- **Target Stream:** Encodec RVQ discrete tokens $\mathbf{Y} \in \{0, \dots, 1023\}^{B \times 4 \times T_{\text{codec}}}$ at $75\text{ Hz}$ ($24\text{ kHz}$ speech).
- **Continuous Alignment:**
  $$\mu(n) = \frac{2}{3} n - 0.29114583, \quad \mathbf{h}_{\text{interp}}(n) = (1 - w)\mathbf{H}_{\text{mms}}[\lfloor \mu \rfloor] + w \mathbf{H}_{\text{mms}}[\lceil \mu \rceil]$$
- **Loss Formulation:**
  $$\mathcal{L}_{\text{ExpA}} = -\frac{1}{T}\sum_{n=1}^{T} \log P(Y_{0, n} = y_{0, n}^* \mid \mathbf{h}_{\text{aligned}}(n))$$
  Loss weights: $\lambda_0 = 1.0, \; \lambda_1 = 0.0, \; \lambda_2 = 0.0, \; \lambda_3 = 0.0$.
- **Codec State:** Encodec 24 kHz RVQ encoder and decoder remain **100% frozen**.

### 2.3 Checkpoint & Progression Milestones
Training will run on the streaming dataset with checkpoints saved and evaluated at regular intervals:
- **Evaluation Points:** Steps 250, 500, 1000, 2000, 3000, 4000, 5000.
- **Audio Output:** At every checkpoint, generate and decode:
  1. `step_{N}_expA_cb0_only.wav` (Decoded with CB0 predicted, CB1–CB3 = 0).
  2. `step_{N}_expA_gt_ref.wav` (Ground-truth Encodec reference).

### 2.4 Experiment A Evaluation Criteria & Disqualification of SI-SDR
- **Primary Metric 1 — Human Intelligibility Score (1–5 MOS):** Blind evaluation by native Tamil speakers evaluating syllable identification.
- **Primary Metric 2 — Offline ASR CER (%):** Transcribe decoded audio via Whisper Large-v3 and MMS-Tamil ASR using full $> 3.0\text{ s}$ audio samples (with prosodic padding to prevent the short-audio decoder repetition loop).
- **Primary Metric 3 — Token Accuracy & Entropy:** Top-1 Accuracy, Top-5 Accuracy, Unigram Entropy ($H \ge 5.0\text{ bits}$), and n-gram repetition rate ($R_{\text{repeat}} < 5\%$).
- **Disqualified Metric:** **SI-SDR is explicitly disqualified as a primary pass/fail criterion.** It will be logged strictly for mathematical tracking, but no optimization decisions will be based on maximizing SI-SDR at the expense of intelligibility.

---

## 3. Experiment B: Residual Codebook Learning (Ground-Truth Conditioning)

### 3.1 Experimental Objective
Test the core hypothesis: **Do residual codebooks (CB1–CB3) restore Tamil speech intelligibility and phonetic crispness, or do they merely add high-frequency noise?**

### 3.2 Architectural Specification
- **Backbone State:** The Transformer encoder and CB0 prediction head trained in Experiment A are **frozen**.
- **Conditioning Input:** Ground-truth CB0 tokens ($Y_{0}^*$) are embedded through the DepthAudioHead:
  $$\mathbf{z}_0 = \mathbf{h}_{\text{latent}} + \mathbf{E}_{\text{depth}, 0}(Y_0^*)$$
  $$\mathbf{z}_1 = \mathbf{z}_0 + \mathbf{E}_{\text{depth}, 1}(Y_1^*)$$
  $$\mathbf{z}_2 = \mathbf{z}_1 + \mathbf{E}_{\text{depth}, 2}(Y_2^*)$$
- **Trainable Parameters:** Depth embeddings ($\mathbf{E}_{\text{depth}, 0..2}$) and linear classification heads for CB1, CB2, and CB3.
- **Loss Formulation:**
  $$\mathcal{L}_{\text{ExpB}} = \sum_{k=1}^{3} \lambda_k \cdot \text{CrossEntropy}(\hat{\mathbf{Y}}_k, \mathbf{Y}_k^*)$$
  Loss weights: $\lambda_0 = 0.0$ (frozen), $\lambda_1 = 1.0, \; \lambda_2 = 0.5, \; \lambda_3 = 0.25$.

### 3.3 Paired A/B Intelligibility Comparison Protocol
At each checkpoint (steps 500, 1000, 2000, 3000, 5000):
1. **Stream 1 (CB0-Only Baseline):** Decode with CB0 from frozen backbone, CB1–CB3 zeroed out:
   $$\hat{\mathbf{Y}}_{\text{cb0}} = [\hat{\mathbf{Y}}_0, \mathbf{0}, \mathbf{0}, \mathbf{0}]$$
2. **Stream 2 (4-Codebook Decoded):** Decode with CB0 and predicted residuals:
   $$\hat{\mathbf{Y}}_{\text{4cb}} = [\hat{\mathbf{Y}}_0, \hat{\mathbf{Y}}_1, \hat{\mathbf{Y}}_2, \hat{\mathbf{Y}}_3]$$
3. **Stream 3 (GT Codec Reference):** Decode ground-truth tokens:
   $$\mathbf{Y}^* = [\mathbf{Y}_0^*, \mathbf{Y}_1^*, \mathbf{Y}_2^*, \mathbf{Y}_3^*]$$

### 3.4 Key Decision Gate for Experiment B
- **Intelligibility Restoration Gate:** If Stream 2 (4-Codebook) achieves statistically significant improvements in native human intelligibility and lower Whisper ASR CER compared to Stream 1 (CB0-Only), the hypothesis is confirmed: **residual codebooks are mandatory for intelligible Tamil speech**.
- **Noise Failure Gate:** If Stream 2 adds harsh distortion without improving syllable recognition, the DepthAudioHead capacity or RVQ depth formulation must be redesigned prior to full training.

---

## 4. Experiment C: Inference-Condition Validation & Exposure Bias Mitigation

### 4.1 Problem Formulation: The Training vs. Inference Exposure Bias
In Experiment B, residual heads are trained using **ground-truth** preceding tokens ($Y_{k-1}^*$, teacher forcing). At inference time, however, ground truth is unavailable; residual heads must condition on the model's **predicted** tokens ($\hat{Y}_{k-1}$).

Because CB0 accuracy is imperfect ($35\% \sim 60\%$), $40\% \sim 65\%$ of tokens fed to the depth head at test time will be wrong. If the residual heads overfit to perfect ground-truth embeddings, inference on predicted tokens will suffer catastrophic cascading degradation.

### 4.2 Four Exposure-Bias Mitigation Strategies to Benchmark
We will implement and benchmark four distinct mitigation protocols under identical data and compute budgets:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   EXPOSURE BIAS MITIGATION PROTOCOLS (EXP C)                     │
└──────────────────────────────────────────────────────────────────────────────────┘

 [Protocol C1: Baseline Free-Run]
   Residuals conditioned naively on argmax(Y_0_pred). Quantifies degradation gap.

 [Protocol C2: Scheduled Sampling]
   Depth head trained with coin-flip: feed GT with prob p(t), predicted with (1 - p(t)).
   p(t) = max(0.40, 1.0 - t / 3000). Forces robustness to noisy tokens.

 [Protocol C3: Soft-Embedding Distribution Projection]
   Depth head receives softmax probability distribution:
   z = h + Sum_v [ P(Y_0 = v) * E_depth(v) ]
   Eliminates discrete argmax cliff; gradient flows smoothly across candidates.

 [Protocol C4: End-to-End Joint Multi-Task Fine-Tuning]
   Unfreeze backbone + depth heads with combined loss:
   L = 1.0 * L_CB0 + 0.5 * L_CB1 + 0.25 * L_CB2 + 0.125 * L_CB3.
```

### 4.3 Experiment C Validation Protocol
Each protocol will be evaluated under **strict inference conditions** (zero ground-truth tokens provided):
1. Measure the **Exposure Gap**:
   $$\Delta_{\text{exposure}} = \text{CER}(\text{Conditioned on Pred CB0}) - \text{CER}(\text{Conditioned on GT CB0})$$
   A lower $\Delta_{\text{exposure}}$ indicates successful mitigation of exposure bias.
2. Select the protocol with the lowest CER and highest human intelligibility under free-running generation.

---

## 5. Comprehensive Evaluation Framework & Controls

### 5.1 Speaker-Disjoint Partitioning (Zero Overlap)
To eliminate phonetic memorization and speaker leak, evaluation will strictly adhere to the disjoint splits:

| Split Role | Speakers Included | Utterances / Hours | Purpose |
| :--- | :--- | :---: | :--- |
| **Training Pool** | `speaker_asst`, `speaker_demo`, `speaker_hf`, `speaker_indic1`, `speaker_indic2`, `speaker_indic3`, `speaker_wsp` | ~20 hours continuous | Model training across diverse acoustic environments |
| **Held-Out Validation** | `speaker_indic4`, `speaker_mms`, `speaker_user` | 32 fixed utterances | Strict out-of-speaker phonetic generalization |

Cross-split speaker overlap is verified at **$0.00\%$**.

### 5.2 Multi-Seed Replication
To establish statistical significance, all experimental conditions will run across **three independent random seeds**:
- Seed 42
- Seed 101
- Seed 2024

Reported metrics will present the **mean $\pm$ standard error** and $95\%$ bootstrap confidence intervals ($B = 2,000$ resamples across utterances).

### 5.3 Baseline Controls
Every evaluation stage will be benchmarked simultaneously against three reference controls:
1. **`gt_codec`:** Uncompressed ground-truth Encodec 24 kHz audio (the acoustic ceiling).
2. **`random_init`:** S2STokenPredictor initialized with standard normal random weights.
3. **`mms_init`:** S2STokenPredictor initialized with weights from Meta MMS-300M layers 0–5.
4. *(Optional Reference)* `wsp_init`: Evaluated for continuity, but not given preferential status.

### 5.4 Granular Metric Reporting Table Template
To prevent aggregate metrics from masking failure modes, CB0 and residual codebooks will be reported separately:

| Experiment / Checkpoint | Seed | CB0 Top-1 | CB0 Top-5 | CB1 Top-1 | CB2 Top-1 | CB3 Top-1 | Whisper CER (Tamil) | MMS ASR CER | Repetition Rate | Human Intelligibility (1-5) | SI-SDR (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A: Step 1000** | 42 | ... | ... | N/A | N/A | N/A | ... | ... | ... | ... | ... |
| **Exp A: Step 5000** | 42 | ... | ... | N/A | N/A | N/A | ... | ... | ... | ... | ... |
| **Exp B: 4-CB (GT CB0)** | 42 | [Frozen] | [Frozen] | ... | ... | ... | ... | ... | ... | ... | ... |
| **Exp C: Free-Run (Pred)** | 42 | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 5.5 Double-Blind Native Human Listening Test Protocol
Automated metrics (CER, loss, accuracy) will be paired with subjective listening tests:
1. **Panel:** Minimum of 3 native Tamil speakers.
2. **Blinding:** Audio files will be normalized to $-20\text{ dBFS}$ RMS, assigned randomized alphanumeric hashes (e.g., `eval_sample_9a8f.wav`), and presented in shuffled order.
3. **Questions:**
   - **Q1 (Intelligibility):** *Can you understand what is being said?* (1: None, 2: Isolated words, 3: Partial sentence, 4: Clear with minor accents, 5: Flawless).
   - **Q2 (Transcript Match):** Listeners write down what they heard; transcript is compared against source text.
   - **Q3 (Acoustic Naturalness):** Rate harshness / phase distortion (1: Unbearable distortion, 5: Clean natural voice).

---

## 6. Strict Architectural & Operational Constraints

1. **No Autoregressive Conversational Generation (Phase F):**
   Milestone A is strictly confined to continuous-to-discrete speech unit prediction. No autoregressive causal masking across past conversational turns is permitted.
2. **Zero Text Conditioning:**
   No text tokens, phoneme strings, BPE tokenizers, or transcript embeddings are permitted in the input, latent space, or loss function. Transcripts are permitted **only for offline ASR evaluation of generated audio**.
3. **Frozen Encodec Codec:**
   The Encodec 24 kHz model (`facebook/encodec_24khz`) must remain strictly frozen. No fine-tuning of Encodec encoder or decoder weights.
4. **No Arbitrary Weight Averaging:**
   No unvalidated grafting or linear interpolation of weights between disparate architectures or models.
5. **Historical Checkpoint Immutability:**
   All historical checkpoints (`checkpoints/*.pt`) are preserved as read-only artifacts. New experimental runs must save to dedicated subdirectories (`checkpoints/exp_a/`, `checkpoints/exp_b/`, `checkpoints/exp_c/`).
6. **Pre-Training Gate:**
   **No GPU training will be initiated until this experimental plan receives formal review and explicit written sign-off from the user.**

---

## 7. Implementation Roadmap & Execution Checklist

- [x] Audit previous SI-SDR and listening discrepancies.
- [x] Document failure of WSP CB0 advantage to replicate on expanded 32-utterance validation pool.
- [x] Formulate controlled three-stage experimental plan (Exp A: CB0-only, Exp B: Residuals with GT conditioning, Exp C: Exposure-bias mitigation).
- [x] Specify speaker-disjoint validation split (7 training speakers vs. 3 held-out validation speakers).
- [x] Disqualify SI-SDR as a primary success metric; establish human listening + ASR CER as gold standards.
- [ ] **Awaiting User Review & Sign-Off on `S2S_V2_REVISED_MILESTONE_A_EXPERIMENTAL_PLAN.md`**.
- [ ] Prepare modular training scripts for Exp A, Exp B, and Exp C with scheduled sampling support.
- [ ] Execute Experiment A (CB0-Only Learning across 3 seeds).
- [ ] Execute Experiment B (Residual Codebook Learning with paired A/B listening tests).
- [ ] Execute Experiment C (Inference Exposure Bias benchmarking across Protocols C1–C4).
- [ ] Deliver Final Milestone A Consolidated Validation Report.
