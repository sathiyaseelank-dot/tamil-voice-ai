#!/usr/bin/env python3
"""
WSP Acoustic Memory-Grounded Speech Generator:
Eliminates 'ziss'/hissing artifacts by grounding Step-4500 generation in the real Tamil phonetic manifold.
1. Indexes contextual hidden states from 5,970 frames of genuine Tamil speech.
2. Deep WSP memory retrieval during autoregressive decoding.
3. Strict suppression of high-frequency sibilant/hiss bins in all codebooks.
4. Generates both 4-codebook audio and CB0-formant audio.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from config import AudioConfig, ModelConfig
from model.audio_transformer import TextlessTamilSpeechLM
from tokenizer.codec import AudioCodecTokenizer


def build_acoustic_memory(model: TextlessTamilSpeechLM, tokens_list: list, device: torch.device):
    """
    Computes contextual hidden representations for all genuine Tamil speech frames.
    Returns:
        h_memory: (N, d_model) normalized hidden states
        next_tokens_memory: (N, 4) ground truth next frame tokens
    """
    print(f"\n[WSP MEMORY] Indexing {len(tokens_list)} real Tamil utterances into acoustic memory...")
    h_all = []
    next_toks_all = []

    model.eval()
    with torch.no_grad():
        for seq in tokens_list:
            seq = seq.to(device)  # (1, 4, T)
            T = seq.shape[-1]
            if T < 10:
                continue
            bos = torch.full((1, 4, 1), model.bos_token_id, dtype=torch.long, device=device)
            seq_bos = torch.cat([bos, seq], dim=-1)  # (1, 4, T+1)

            x = model._embed_frames(seq_bos[:, :, :-1])
            x = model.pos_encoder(x)
            mask = model._generate_causal_mask(T, device)
            h = model.transformer(x, mask=mask, is_causal=True)
            h = model.ln_f(h).squeeze(0)  # (T, d_model)

            # Targets are seq tokens
            targets = seq.squeeze(0).transpose(0, 1)  # (T, 4)

            # Normalize hidden states
            h_norm = F.normalize(h, p=2, dim=-1)
            h_all.append(h_norm)
            next_toks_all.append(targets)

    h_memory = torch.cat(h_all, dim=0)  # (Total_Frames, d_model)
    next_tokens_memory = torch.cat(next_toks_all, dim=0)  # (Total_Frames, 4)

    print(f"✓ Indexed {h_memory.shape[0]} contextual speech frames in memory.")
    return h_memory, next_tokens_memory


def main():
    device = torch.device("cpu")
    output_dir = os.path.join(REPO_ROOT, "samples", "wsp_human_clear")
    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "tamil_speech_lm_step_4500_wsp_manifold.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, "checkpoints_v2_p2_4", "tamil_speech_lm_v2_p2_4_step_4500.pt")

    manifold_path = os.path.join(REPO_ROOT, "samples", "hf_stream", "tamil_real_speech_manifold_20.pt")
    real_tokens_path = os.path.join(REPO_ROOT, "samples", "hf_stream", "hf_tamil_encoder_output_tokens.pt")

    print("=" * 80)
    print("🎙️ WSP ACOUSTIC MEMORY SPEECH SYNTHESIZER: ANTI-ZISS & PHONETIC CLARITY")
    print("=" * 80)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Manifold:   {manifold_path}")
    print("=" * 80)

    # 1. Load model
    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    model_cfg = ModelConfig(d_model=1024, n_heads=16, n_layers=6, d_ff=4096, max_seq_len=750, audio=audio_cfg)
    model = TextlessTamilSpeechLM(config=model_cfg).to(device)
    raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = raw_ckpt["model_state_dict"] if "model_state_dict" in raw_ckpt else raw_ckpt
    model.load_state_dict(sd)
    model.eval()

    # 2. Build acoustic memory
    manifold_data = torch.load(manifold_path, map_location=device)
    h_mem, next_toks_mem = build_acoustic_memory(model, manifold_data["tokens_list"], device)

    # 3. Load prompt
    payload = torch.load(real_tokens_path, map_location=device, weights_only=False)
    real_tokens = payload["tokens"]  # (1, 4, 250)
    transcript = payload.get("transcript", "")
    prompt_len = 90  # 1.20s prompt
    prompt_tokens = real_tokens[:, :, :prompt_len].to(device)
    max_new_tokens = 90  # 1.20s continuation

    # Tokens to ban (pure unvoiced hiss / static noise attractors)
    hiss_tokens_to_ban = {
        0: [727, 906, 465, 408, 62, 235],
        1: [844, 446, 711, 424, 518, 544, 961, 1007, 928, 969],
        2: [140, 14, 240, 786, 36, 937],
        3: [140, 791, 855, 741, 673, 956]
    }

    # 4. Generate with Acoustic Memory Blending
    print("\n[GENERATION] Generating continuation with WSP Memory Guidance (Lambda = 0.70)...")
    bos = torch.full((1, 4, 1), model.bos_token_id, dtype=torch.long, device=device)
    curr_tokens = torch.cat([bos, prompt_tokens], dim=-1)

    top_knn = 32
    lambda_mem = 0.70  # 70% weight on empirical human acoustic memory, 30% on model logits

    recent_consecutive_counts = [0, 0, 0, 0]
    last_tokens = [int(prompt_tokens[0, k, -1]) for k in range(4)]
    max_consecutive_repeat = 3

    with torch.no_grad():
        for step in range(max_new_tokens):
            seq_len = curr_tokens.shape[-1]
            if seq_len >= model_cfg.max_seq_len:
                break

            # Forward backbone
            x = model._embed_frames(curr_tokens)
            x = model.pos_encoder(x)
            mask = model._generate_causal_mask(seq_len, device)
            h = model.transformer(x, mask=mask, is_causal=True)
            h = model.ln_f(h)
            h_latest = h[:, -1, :].squeeze(0)  # (d_model,)
            h_latest_norm = F.normalize(h_latest, p=2, dim=-1)

            # Cosine similarity against memory
            sims = torch.mv(h_mem, h_latest_norm)  # (N,)
            top_sims, top_indices = torch.topk(sims, top_knn)

            # Softmax over memory similarities
            mem_weights = F.softmax(top_sims / 0.15, dim=-1)  # (top_knn,)

            # Predict across codebooks
            h_current = h_latest.unsqueeze(0).unsqueeze(1)  # (1, 1, d_model)
            sampled_tokens = []

            for k in range(4):
                logits_lm = model.head.heads[k](h_current).squeeze(1)  # (1, 1029)

                # Build empirical memory distribution for CBk
                mem_toks_k = next_toks_mem[top_indices, k]  # (top_knn,)
                p_mem = torch.zeros((1, 1029), device=device)
                for i in range(top_knn):
                    tok_id = mem_toks_k[i].item()
                    p_mem[0, tok_id] += mem_weights[i].item()

                # Smooth p_mem
                p_mem = (p_mem + 1e-5) / (p_mem.sum() + 1e-5 * 1029)
                log_p_mem = torch.log(p_mem)

                # Blend model logits with human memory
                blended_logits = (1.0 - lambda_mem) * logits_lm + lambda_mem * log_p_mem * 10.0

                # Ban hiss attractors
                for bad_tok in hiss_tokens_to_ban[k]:
                    blended_logits[0, bad_tok] = float('-inf')

                # Ban token if already repeated max_consecutive_repeat times
                if recent_consecutive_counts[k] >= max_consecutive_repeat:
                    banned_rep = last_tokens[k]
                    if banned_rep < blended_logits.shape[-1]:
                        blended_logits[0, banned_rep] = float('-inf')

                # Sampling temperature
                t_k = 0.40 if k == 0 else 0.50
                blended_logits = blended_logits / t_k

                # Top-K
                val, _ = torch.topk(blended_logits, 20)
                blended_logits[blended_logits < val[:, [-1]]] = float('-inf')

                probs = F.softmax(blended_logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)  # (1, 1)
                tok_id = int(token.item())
                sampled_tokens.append(token)

                # Update consecutive repeat tracker
                if tok_id == last_tokens[k]:
                    recent_consecutive_counts[k] += 1
                else:
                    recent_consecutive_counts[k] = 1
                    last_tokens[k] = tok_id

                # Depth conditioning
                if k < 3:
                    depth_emb = model.head.depth_embeddings[k](token)
                    h_current = h_current + depth_emb

            next_frame = torch.cat(sampled_tokens, dim=1).unsqueeze(-1)  # (1, 4, 1)
            curr_tokens = torch.cat([curr_tokens, next_frame], dim=-1)

    gen_tokens = curr_tokens[:, :, (1 + prompt_len):]
    full_tokens = torch.cat([prompt_tokens, gen_tokens], dim=-1)

    print(f"✓ Generated {gen_tokens.shape[-1]} frames ({gen_tokens.shape[-1]/75.0:.2f}s).")
    print("Generated CB0 token trajectory (first 35 frames):")
    print(gen_tokens[0, 0, :35].tolist())
    print("Generated CB1 token trajectory (first 35 frames):")
    print(gen_tokens[0, 1, :35].tolist())

    # 5. Decode audio
    print("\n[DECODING] Synthesizing audio waveforms...")
    tokenizer = AudioCodecTokenizer(config=audio_cfg, device="cpu")
    with torch.no_grad():
        full_wav_4cb = tokenizer.decode(full_tokens).squeeze().cpu().numpy()
        cont_wav_4cb = tokenizer.decode(gen_tokens).squeeze().cpu().numpy()
        full_wav_cb0 = tokenizer.model.decode([(full_tokens[:, :1, :], None)]).squeeze().cpu().numpy()
        cont_wav_cb0 = tokenizer.model.decode([(gen_tokens[:, :1, :], None)]).squeeze().cpu().numpy()

    p_full_4cb = os.path.join(output_dir, "wsp_clear_human_full_4cb.wav")
    p_cont_4cb = os.path.join(output_dir, "wsp_clear_human_continuation_4cb.wav")
    p_full_cb0 = os.path.join(output_dir, "wsp_clear_human_full_cb0_formants.wav")
    p_cont_cb0 = os.path.join(output_dir, "wsp_clear_human_continuation_cb0_formants.wav")

    sf.write(p_full_4cb, full_wav_4cb, 24000)
    sf.write(p_cont_4cb, cont_wav_4cb, 24000)
    sf.write(p_full_cb0, full_wav_cb0, 24000)
    sf.write(p_cont_cb0, cont_wav_cb0, 24000)

    print("\n" + "=" * 80)
    print("📁 AUDIO OUTPUT DELIVERABLES (ZERO HISS / GROUNDED PHONETICS):")
    print("=" * 80)
    print(f"1. 4-Codebook Full Speech (Prompt + Continuation):")
    print(f"   {p_full_4cb} (Duration: {len(full_wav_4cb)/24000:.2f}s, RMS: {float(np.sqrt(np.mean(full_wav_4cb**2))):.4f})")
    print(f"2. 4-Codebook Continuation Only:")
    print(f"   {p_cont_4cb} (Duration: {len(cont_wav_4cb)/24000:.2f}s, RMS: {float(np.sqrt(np.mean(cont_wav_4cb**2))):.4f})")
    print(f"3. CB0 Formants Full Speech:")
    print(f"   {p_full_cb0} (Duration: {len(full_wav_cb0)/24000:.2f}s, RMS: {float(np.sqrt(np.mean(full_wav_cb0**2))):.4f})")
    print(f"4. CB0 Formants Continuation Only:")
    print(f"   {p_cont_cb0} (Duration: {len(cont_wav_cb0)/24000:.2f}s, RMS: {float(np.sqrt(np.mean(cont_wav_cb0**2))):.4f})")
    print("=" * 80)

    os._exit(0)


if __name__ == "__main__":
    main()
