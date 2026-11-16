# Textless Speech-to-Speech (S2S) Model Implementation Guide

This guide documents the architecture, Weight-Space Programming (WSP) grafting recipe, and execution instructions for the **Unified Textless Tamil Speech-to-Speech (S2S) Model**.

---

## 1. Architecture Overview

The system is a pure **textless** conversational speech pipeline: spoken audio in, spoken audio out. No text tokens, character tokenizers, or phonemic dictionaries are used anywhere in the forward pass.

```
                           SPOKEN TAMIL AUDIO INPUT (16 kHz / 24 kHz .wav)
                                                 │
                                                 ▼
             ┌────────────────────────────────────────────────────────────────────────┐
             │            Acoustic Convolutional Frontend (Meta MMS-300M)             │
             │   7 temporal convolutional layers downsampling raw audio to 50 Hz      │
             └────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼  Hidden States [Batch, T, 1024]
             ┌────────────────────────────────────────────────────────────────────────┐
             │      WSP-Crafted 6-Layer Causal Transformer Backbone (1024 d_model)    │
             │                                                                        │
             │      Fused in Weight Space:                                            │
             │      W_fused = 0.35 * W_4500      (Tamil conversational associations)  │
             │              + 0.35 * W_mms       (Multilingual/Tamil phonetic circuits│
             │              + 0.30 * W_musicgen  (Causal temporal stability)          │
             └────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼  Contextual Output [Batch, T, 1024]
             ┌────────────────────────────────────────────────────────────────────────┐
             │               WSP Vocal Tract Latent Projector (1024 -> 192)           │
             │   Calibrated linear projection into VITS vocal tract acoustic manifold │
             └────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼  Vocal Latents [Batch, 192, T]
             ┌────────────────────────────────────────────────────────────────────────┐
             │             VITS HiFi-GAN Neural Speech Vocoder (Meta MMS-Tamil)       │
             │   Upsamples acoustic latent representations into 16 kHz human speech   │
             └────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                         SYNTHESIZED TAMIL SPEECH OUTPUT (.wav)
```

---

## 2. Environment Setup

Activate the project virtual environment:

```bash
cd /home/inkyank-01/dev/tamil-voice-ai
source activate.sh
```

Ensure the following core libraries are available:
* `torch >= 2.1.0`
* `torchaudio >= 2.1.0`
* `transformers >= 4.38.0`
* `soundfile >= 0.12.1`
* `encodec >= 0.1.1`

---

## 3. Python API Implementation

You can load and execute the textless S2S model directly in Python:

```python
import torch
import torchaudio
import soundfile as sf
from model.textless_s2s_wsp import TextlessTamilS2SModel

# 1. Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
s2s_model = TextlessTamilS2SModel(
    d_model=1024,
    n_heads=16,
    n_layers=6,
    d_ff=4096,
    latent_dim=192,
    device=device
)

# 2. Craft or load weights
ckpt_path = "checkpoints/textless_tamil_s2s_wsp_crafted.pt"
if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    s2s_model.load_state_dict(checkpoint["model_state_dict"])
else:
    # Perform WSP surgery and grafting across models
    s2s_model.craft_weights_via_wsp(
        ckpt_4500_path="checkpoints_v2_p2_4/tamil_speech_lm_v2_p2_4_step_4500.pt",
        alpha_4500=0.35,
        alpha_mms=0.35,
        alpha_musicgen=0.30
    )

s2s_model.eval()

# 3. Read spoken input audio (any sample rate resampled to 16 kHz)
raw_audio, sr = sf.read("my_input_speech.wav")
w = torch.from_numpy(raw_audio).float().unsqueeze(0)
if sr != 16000:
    w = torchaudio.functional.resample(w, sr, 16000)

# 4. Pure textless speech-to-speech inference
with torch.no_grad():
    output_waveform = s2s_model(w).squeeze().cpu().numpy()

# 5. Save generated speech audio
sf.write("my_output_speech.wav", output_waveform, 16000)
print("✓ Synthesized spoken speech saved to my_output_speech.wav")
```

---

## 4. Command-Line Execution (CLI)

To run the complete WSP fusion, vocal manifold calibration, and inference on a WAV file:

```bash
python run_textless_s2s_wsp.py
```

Outputs generated:
* Synthesized speech audio: `samples/wsp_textless_s2s/wsp_textless_s2s_output.wav`
* WSP-crafted model checkpoint: `checkpoints/textless_tamil_s2s_wsp_crafted.pt` (374 MB)

---

## 5. Weight-Space Programming (WSP) Fusion Details

The 6-layer transformer backbone fuses parameter tensors from three donor checkpoints:

$$W_{\text{fused}} = 0.35 \cdot W_{\text{4500}} + 0.35 \cdot W_{\text{mms-300m}} + 0.30 \cdot W_{\text{musicgen-small}}$$

1. **In-Projection Tensors:** Concatenates Query, Key, and Value projection weights (`[1024, 1024] -> [3072, 1024]`) to match PyTorch's native multi-head attention format.
2. **Feed-Forward Circuits:** Fuses `linear1` (`[4096, 1024]`) and `linear2` (`[1024, 4096]`) with matching GELU activations.
3. **Vocal Projector Calibration:** Fits a low-rank mapping from the 1024-dimensional backbone state to the 192-dimensional VITS latent space using the empirical Tamil vocal manifold.

---

## 6. Key Source Files in Repository

* `model/textless_s2s_wsp.py`: Core architecture definition and multi-model WSP grafting methods.
* `run_textless_s2s_wsp.py`: Calibration runner and textless forward-pass test script.
* `tokenizer/semantic_tokenizer.py`: Self-supervised phonetic speech unit extractor (MMS-300M).
* `tokenizer/hybrid_codec.py`: Dual-stream semantic + acoustic RVQ interface.
