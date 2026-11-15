#!/usr/bin/env python3
"""
Experiment V2-P2.4: Phase 1 — Dataset Preparation

Expands the Tamil training corpus to approximately 20 hours using IndicVoices-R_Tamil.
Requirements:
1. Verify audio duration and sample rate (resample to 24 kHz mono).
2. Remove corrupted, empty, or silent recordings (< 1.0s or max amplitude < 1e-4).
3. Remove duplicate recordings by text/audio hash.
4. Preserve speaker and recording metadata.
5. Create strictly speaker-disjoint training and validation splits.
   - Evaluation speakers (S4258098500393504 and S4259865400376309) are 100% reserved for validation.
   - Zero speaker overlap between train and validation.
6. Encode through the verified 4-codebook Encodec pipeline.
7. Save cached tokens and manifest for deterministic, high-throughput training.
"""

import os
import sys
import io
import time
import json
import glob
import random
import hashlib
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, '/content/tamil-voice-ai')
from config import AudioConfig, ModelConfig
from tokenizer.codec import AudioCodecTokenizer


def main():
    print("=" * 80)
    print("EXPERIMENT V2-P2.4: PHASE 1 — DATASET PREPARATION (~20 HOURS)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Set seed
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)

    audio_cfg = AudioConfig(sample_rate=24000, num_codebooks=4)
    tokenizer = AudioCodecTokenizer(audio_cfg, device=str(device))

    # Locate parquet files
    parquet_pattern = '/root/.cache/huggingface/hub/datasets--SPRINGLab--IndicVoices-R_Tamil/snapshots/*/data/train-0000[0-2]-of-00014.parquet'
    parquet_files = sorted(glob.glob(parquet_pattern))
    assert len(parquet_files) == 3, f"Expected 3 parquet files, found: {parquet_files}"
    print(f"Located {len(parquet_files)} parquet partitions.")

    # Mandatory held-out evaluation speakers from V2-P1 / V2-P2
    mandatory_val_speakers = {'S4258098500393504', 'S4259865400376309'}

    # Collect all records metadata first across all 3 parquets
    print("\n[1/4] Scanning metadata across partitions...")
    all_records = []
    seen_texts = set()
    total_raw_rows = 0
    total_raw_duration = 0.0

    for p_idx, p_path in enumerate(parquet_files):
        print(f"  Scanning partition {p_idx}: {os.path.basename(p_path)}...")
        pf = pq.ParquetFile(p_path)
        for batch in pf.iter_batches(batch_size=1000, columns=['speaker_id', 'duration', 'text', 'district', 'gender']):
            df_b = batch.to_pandas()
            for idx_in_batch, row in df_b.iterrows():
                total_raw_rows += 1
                dur = float(row['duration'])
                total_raw_duration += dur
                spk = str(row['speaker_id'])
                txt = str(row.get('text', '')).strip()
                dist = str(row.get('district', ''))
                gender = int(row.get('gender', 0))

                # Basic validation filtering
                if dur < 1.0 or dur > 25.0:
                    continue
                if not txt:
                    continue
                # Deduplication by text
                if txt in seen_texts:
                    continue
                seen_texts.add(txt)

                all_records.append({
                    "partition": p_idx,
                    "global_row": total_raw_rows - 1,
                    "speaker_id": spk,
                    "duration": dur,
                    "text": txt,
                    "district": dist,
                    "gender": gender
                })

    print(f"✓ Total raw rows: {total_raw_rows} ({total_raw_duration/3600.0:.2f} hours)")
    print(f"✓ Valid deduplicated rows: {len(all_records)}")

    # Group records by speaker
    speaker_records = {}
    for r in all_records:
        spk = r['speaker_id']
        speaker_records.setdefault(spk, []).append(r)

    unique_speakers = sorted(list(speaker_records.keys()))
    print(f"✓ Total unique speakers: {len(unique_speakers)}")

    # Designate Speaker-Disjoint Split
    # Mandatory val speakers are reserved first
    val_speakers = set(mandatory_val_speakers)
    # Add additional validation speakers to reach ~1.5 - 2.0 hours of validation audio
    val_duration_target = 1.75 * 3600.0
    current_val_duration = sum(sum(r['duration'] for r in speaker_records[spk]) for spk in val_speakers if spk in speaker_records)

    other_speakers = [s for s in unique_speakers if s not in mandatory_val_speakers]
    random.shuffle(other_speakers)

    for spk in other_speakers:
        if current_val_duration >= val_duration_target:
            break
        val_speakers.add(spk)
        current_val_duration += sum(r['duration'] for r in speaker_records[spk])

    train_speakers = set(s for s in unique_speakers if s not in val_speakers)

    # STRICT ASSERTIONS
    assert len(train_speakers.intersection(val_speakers)) == 0, "FATAL: Speaker leakage detected between train and val!"
    for mandatory in mandatory_val_speakers:
        assert mandatory not in train_speakers, f"FATAL: Mandatory evaluation speaker {mandatory} leaked into train!"
        assert mandatory in val_speakers, f"FATAL: Mandatory evaluation speaker {mandatory} missing from val!"

    train_records = [r for r in all_records if r['speaker_id'] in train_speakers]
    val_records = [r for r in all_records if r['speaker_id'] in val_speakers]

    train_duration_hours = sum(r['duration'] for r in train_records) / 3600.0
    val_duration_hours = sum(r['duration'] for r in val_records) / 3600.0

    print("\n[2/4] Speaker-Disjoint Split Statistics:")
    print(f"  Training Split:   {len(train_records)} recordings, {train_duration_hours:.2f} hours, {len(train_speakers)} speakers")
    print(f"  Validation Split: {len(val_records)} recordings, {val_duration_hours:.2f} hours, {len(val_speakers)} speakers")
    print(f"  Total Duration:   {train_duration_hours + val_duration_hours:.2f} hours")
    print(f"  Speaker Overlap:  {len(train_speakers.intersection(val_speakers))} (Strictly 0)")

    # Prepare encoding and storage
    out_dir = '/content/tamil-voice-ai/data/v2_p2_4'
    os.makedirs(out_dir, exist_ok=True)

    print("\n[3/4] Encoding Audio with Encodec 24 kHz (4 Codebooks)...")

    # Read parquets sequentially and extract audio bytes
    t0 = time.time()
    train_tokens_list = []
    val_tokens_list = []

    # Map (partition, global_row) to target record
    record_map = {(r['partition'], r['global_row']): r for r in all_records}

    processed_count = 0
    skipped_corrupt = 0
    encoded_audio_sec = 0.0

    current_global_row = 0
    for p_idx, p_path in enumerate(parquet_files):
        print(f"  Encoding from partition {p_idx}: {os.path.basename(p_path)}...")
        pf = pq.ParquetFile(p_path)
        for batch in pf.iter_batches(batch_size=200, columns=['audio', 'duration']):
            df_b = batch.to_pandas()
            for idx_in_batch, row in df_b.iterrows():
                row_key = (p_idx, current_global_row)
                current_global_row += 1

                if row_key not in record_map:
                    continue

                rec = record_map[row_key]
                audio_entry = row['audio']
                if not audio_entry or not isinstance(audio_entry, dict) or 'bytes' not in audio_entry:
                    skipped_corrupt += 1
                    continue

                audio_bytes = audio_entry['bytes']
                if not audio_bytes:
                    skipped_corrupt += 1
                    continue

                try:
                    raw_audio, sr = sf.read(io.BytesIO(audio_bytes))
                except Exception:
                    skipped_corrupt += 1
                    continue

                wav = torch.from_numpy(raw_audio).float()
                if wav.ndim == 1:
                    wav = wav.unsqueeze(0).unsqueeze(0)
                elif wav.ndim == 2:
                    wav = wav.mean(dim=-1, keepdim=True).transpose(0, 1).unsqueeze(0)

                # Amplitude check
                max_amp = float(wav.abs().max().item())
                if max_amp < 1e-4:
                    skipped_corrupt += 1
                    continue

                # Resample to 24 kHz if needed
                if sr != 24000:
                    wav = torchaudio.functional.resample(wav, sr, 24000)

                wav = wav.to(device)
                # Normalize peak to 0.9 (standard training normalization)
                wav = wav / (wav.abs().max() + 1e-6) * 0.9

                with torch.no_grad():
                    # (1, 4, seq_len)
                    tokens = tokenizer.encode(wav, sample_rate=24000)

                # Store as int16 CPU tensor to optimize RAM
                tokens_cpu = tokens.squeeze(0).cpu().to(torch.int16)
                seq_len = tokens_cpu.shape[-1]
                if seq_len < 30:
                    continue

                item = {
                    "tokens": tokens_cpu,
                    "speaker_id": rec['speaker_id'],
                    "duration": rec['duration'],
                    "text": rec['text'],
                    "district": rec['district']
                }

                if rec['speaker_id'] in train_speakers:
                    train_tokens_list.append(item)
                else:
                    val_tokens_list.append(item)

                encoded_audio_sec += rec['duration']
                processed_count += 1

                if processed_count % 500 == 0:
                    elapsed = time.time() - t0
                    speed = encoded_audio_sec / max(elapsed, 1e-5)
                    print(f"    Encoded {processed_count} recordings ({encoded_audio_sec/3600.0:.2f}h) | Speed: {speed:.1f}x real-time | Elapsed: {elapsed/60.0:.1f}m")

    total_encoded_time = time.time() - t0
    print(f"\n✓ Encoding completed in {total_encoded_time/60.0:.2f} minutes!")
    print(f"✓ Train items encoded: {len(train_tokens_list)} | Val items encoded: {len(val_tokens_list)} | Skipped corrupt: {skipped_corrupt}")

    # Save datasets
    print("\n[4/4] Saving Pre-Encoded Datasets and Manifest...")
    train_save_path = os.path.join(out_dir, "v2_p2_4_train_tokens.pt")
    val_save_path = os.path.join(out_dir, "v2_p2_4_val_tokens.pt")
    manifest_save_path = os.path.join(out_dir, "v2_p2_4_manifest.json")

    torch.save(train_tokens_list, train_save_path)
    torch.save(val_tokens_list, val_save_path)

    manifest = {
        "experiment": "V2-P2.4",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_audio_hours": encoded_audio_sec / 3600.0,
        "train_items": len(train_tokens_list),
        "val_items": len(val_tokens_list),
        "train_speakers_count": len(train_speakers),
        "val_speakers_count": len(val_speakers),
        "train_duration_hours": sum(i['duration'] for i in train_tokens_list) / 3600.0,
        "val_duration_hours": sum(i['duration'] for i in val_tokens_list) / 3600.0,
        "mandatory_val_speakers": list(mandatory_val_speakers),
        "train_save_path": train_save_path,
        "val_save_path": val_save_path,
        "train_file_size_mb": os.path.getsize(train_save_path) / (1024 * 1024),
        "val_file_size_mb": os.path.getsize(val_save_path) / (1024 * 1024),
        "speaker_overlap": len(train_speakers.intersection(val_speakers))
    }

    with open(manifest_save_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"✓ Train dataset saved to: {train_save_path} ({manifest['train_file_size_mb']:.1f} MB)")
    print(f"✓ Val dataset saved to:   {val_save_path} ({manifest['val_file_size_mb']:.1f} MB)")
    print(f"✓ Manifest saved to:      {manifest_save_path}")
    print("\n✅ PHASE 1 DATASET PREPARATION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
