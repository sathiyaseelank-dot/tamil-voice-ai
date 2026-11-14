# Textless Conversational Speech-to-Speech AI for Tamil (தமிழ்)
*Inspired by Block's Dessa, Kyutai's Moshi, and Meta's AudioCraft*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sathiyaseelank-dot/tamil-voice-ai/blob/main/Tamil_Voice_AI_Colab_Training.ipynb)

An end-to-end conversational voice AI model for the **Tamil language** that operates entirely on **raw audio without intermediate text transcription**. Accelerated with **Weight-Space Programming (WSP)** and streaming datasets.

---

## 💡 Why Textless Speech-to-Speech Matters for Tamil (Diglossia)

In Tamil, there is a fundamental linguistic divide:
- **Literary / Written Tamil (செந்தமிழ் / எழுத்துத் தமிழ்):** Formal grammatical structure used in textbooks, formal news broadcasts, and standard text corpora.
- **Spoken / Conversational Tamil (பேச்சுத் தமிழ்):** The dynamic, natural dialect spoken in daily life (e.g. Chennai, Madurai, Kongu, Nellai, Batticaloa/Jaffna). 

Traditional voice pipelines (**ASR $\rightarrow$ Text LLM $\rightarrow$ TTS**) fail conversational Tamil because:
1. Speech-to-text forces colloquial speech into awkward, formal literary text tokens.
2. Text models strip out acoustic nuances: emotional tone, rhythm, pitch inflections, backchannels (*"ஆமா"*, *"சரி"*, *"ம்ம்ம்"*), laughter, and question markers (*"தெரியுமா?"*).
3. TTS synthesizes formal text back into speech, producing robotic, unnatural cadences.

**This project bypasses text completely:**
$$\text{Raw Tamil Audio In} \xrightarrow{\text{Neural Codec}} \text{RVQ Tokens} \xrightarrow{\text{Audio Transformer LM}} \text{Predicted Tokens} \xrightarrow{\text{Codec Decoder}} \text{Raw Tamil Audio Out}$$

---

## 🏛️ System Architecture

```
[User Audio (Tamil)]
        │ (24kHz Mono Waveform)
        ▼
┌───────────────────────────────┐
│     Neural Audio Codec        │  (Encodec RVQ: 4 or 8 codebooks @ 75 Hz)
│   (Continuous -> Discrete)    │  Each frame: [c_0, c_1, c_2, c_3] ∈ [0..1023]
└──────────────┬────────────────┘
               │  Multi-codebook token stream
               ▼
┌───────────────────────────────┐
│  Dual-Stream Transformer LM   │
│  • User Stream (Listening)    │  Full-duplex temporal attention
│  • Asst Stream (Speaking)     │  Predicts next audio tokens autoregressively
│  • Hierarchical Depth Head    │
└──────────────┬────────────────┘
               │  Predicted RVQ token stream
               ▼
┌───────────────────────────────┐
│     Neural Codec Decoder      │  (Residual Vector Dequantization + ConvNet)
│   (Discrete -> Continuous)    │  Synthesizes raw audio waveform in real time
└──────────────┬────────────────┘
        │
        ▼
[Assistant Response Audio (Tamil)]
```

---

## ⚡ Google Colab & Streaming Dataset Training

Train directly on Google Colab using Hugging Face audio streaming without downloading 50GB+ audio datasets to disk:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sathiyaseelank-dot/tamil-voice-ai/blob/main/Tamil_Voice_AI_Colab_Training.ipynb)

```bash
# Streaming training on Hugging Face IndicVoices-R Tamil
python train_streaming.py \
    --dataset SPRINGLab/IndicVoices-R_Tamil \
    --checkpoint checkpoints/tamil_speech_lm_wsp_grafted.pt \
    --max_steps 1000 \
    --device cuda
```

---

## 📁 Repository Structure

```
tamil_voice_ai/
├── config.py                 # Audio, Model, and Training hyperparameters
├── requirements.txt          # Python dependencies
├── README.md                 # Architecture & guide
├── demo.py                   # Automated end-to-end verification script
├── train.py                  # Autoregressive Audio LM training script
├── inference.py              # Zero-text Speech-to-Speech inference pipeline
├── tokenizer/
│   ├── codec.py              # RVQ Tokenizer & Vocoder (Encodec 24kHz)
│   └── audio_io.py           # Robust audio loading and saving (soundfile)
├── model/
│   └── audio_transformer.py  # Dual-Stream Causal Speech-to-Speech Transformer
├── dataset/
│   ├── tamil_dataset.py      # Conversational dialogue dataset & token caching
│   └── download_tamil_audio.py # Audio generator & open dataset downloader
├── samples/                  # Audio input/output samples (.wav)
├── tokenized_data/           # Cached pre-tokenized tensors (.pt)
└── checkpoints/              # Model weights (.pt)
```

---

## 🚀 Quickstart & Verification

### 1. Set Up Environment
```bash
cd tamil_voice_ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Demo
To verify the complete pipeline (synthetic dialogue generation $\rightarrow$ audio tokenization $\rightarrow$ mini-training $\rightarrow$ autoregressive inference $\rightarrow$ speech synthesis):
```bash
python demo.py
```

### 3. Run Inference on Your Own Tamil Audio
```bash
python inference.py --input path/to/tamil_speech.wav --output response_tamil.wav --checkpoint checkpoints/tamil_speech_lm_latest.pt
```

---

## 🎙️ Sourcing Conversational Tamil Datasets for Scaling

To scale the model to production conversational fluency:
1. **OpenSLR SLR65 (Tamil Speech):** High-quality spoken Tamil audio corpus (~40 hours).
2. **Mozilla Common Voice (Tamil):** Thousands of crowd-sourced Tamil speakers.
3. **AI4Bharat IndicSpeech / IndicTTS:** Rich Tamil speech recordings across diverse accents.
4. **Conversational YouTube / Podcasts:** Tamil talk shows and conversational podcasts (resampled to 24kHz mono and segmented with Voice Activity Detection).
