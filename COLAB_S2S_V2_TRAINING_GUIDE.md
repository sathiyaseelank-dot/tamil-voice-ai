# GOOGLE COLAB S2S V2 MILESTONE A TRAINING GUIDE

This guide provides instructions to run the **Textless Tamil Speech-to-Speech (S2S V2)** training on Google Colab with GPU acceleration.

---

## 1. Quick Start with Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File > Open notebook > GitHub**.
3. Enter repository URL: `https://github.com/sathiyaseelank-dot/tamil-voice-ai`
4. Select the branch: `feat/s2s-v2-milestone-a`
5. Open the notebook: [`Colab_S2S_V2_Milestone_A.ipynb`](file:///home/inkyank-01/dev/tamil-voice-ai/Colab_S2S_V2_Milestone_A.ipynb)
6. Ensure GPU is enabled (**Runtime > Change runtime type > T4 GPU** or higher).

---

## 2. Key Commands in Colab

### Clone & Checkout the Milestone A Branch:
```bash
!git clone https://github.com/sathiyaseelank-dot/tamil-voice-ai.git
%cd tamil-voice-ai
!git checkout feat/s2s-v2-milestone-a
!pip install -q soundfile librosa encodec transformers datasets accelerate torchaudio einops
```

### Mount Google Drive (Recommended for Checkpoint Preservation):
```python
from google.colab import drive
drive.mount('/content/drive')
SAVE_DIR = '/content/drive/MyDrive/tamil_voice_ai_checkpoints/s2s_v2'
```

### Run Stage 1 Training (CB0 Phonetic Backbone):
```bash
!python train_s2s_v2_streaming.py \
    --stage 1 \
    --init_type mms \
    --max_steps 5000 \
    --lr 2e-4 \
    --eval_interval 250 \
    --save_interval 500 \
    --save_dir "$SAVE_DIR"
```

### Run Stage 2 Training (Residual Harmonics):
```bash
!python train_s2s_v2_streaming.py \
    --stage 2 \
    --resume "$SAVE_DIR/s2s_v2_stage1_latest.pt" \
    --max_steps 10000 \
    --lr 1e-4 \
    --eval_interval 250 \
    --save_interval 500 \
    --save_dir "$SAVE_DIR"
```

---

## 3. Two-Stage Codebook Rationale

As proven in [`S2S_V2_MILESTONE_A_READINESS_REPORT.md`](file:///home/inkyank-01/dev/tamil-voice-ai/S2S_V2_MILESTONE_A_READINESS_REPORT.md):
- When residual codebooks CB1–CB3 have low accuracy (~12%), they inject phase distortion that degrades scale-invariant SNR by **$8.01\text{ dB}$**.
- Encodec RVQ was designed with codebook dropout: **decoding with CB0 alone produces clean, warm formants and intelligible vowel/consonant envelopes**.
- **Stage 1** forces 100% gradient concentration on CB0 ($\lambda_0 = 1.0, \lambda_{1..3} = 0.0$).
- **Stage 2** refines the depth conditioning on residual codebooks after CB0 is established.
