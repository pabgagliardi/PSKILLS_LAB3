import io
import json
import hashlib
import soundfile as sf
import numpy as np
from datasets import load_dataset, Audio
from pathlib import Path
import yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# --- Config ---
LANG = params["lang"]
LANG_MLS = params["lang_mls"]
N_SAMPLES = params["n_samples"]
OUT_WAV_DIR = Path(f"data/raw/{LANG}/wav")
OUT_MANIFEST = Path(f"data/manifests/{LANG}/clean.jsonl")
TMP_MANIFEST = OUT_MANIFEST.with_suffix(".jsonl.tmp")

# --- Setup ---
OUT_WAV_DIR.mkdir(parents=True, exist_ok=True)
OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

# --- Download data (streaming, no audio auto-decode) ---
print("Downloading MLS data...")
ds = load_dataset(
    "facebook/multilingual_librispeech",
    LANG_MLS,
    split="test",
    streaming=True,
)
# Disable automatic audio decoding → we handle it ourselves
ds = ds.cast_column("audio", Audio(decode=False))
ds = ds.take(N_SAMPLES)

# --- Process each utterance ---
records = []
for i, sample in enumerate(ds):
    stem = f"mls_{i:06d}"
    utt_id = f"{LANG}_{stem}"
    wav_path = OUT_WAV_DIR / f"{stem}.wav"

    # Decode audio from raw bytes ourselves (no FFmpeg needed)
    audio_bytes = sample["audio"]["bytes"]
    audio_array, sr = sf.read(io.BytesIO(audio_bytes))

    # Convert to mono if needed
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    # Resample to 16000 Hz if needed (wav2vec2 requires 16kHz)
    if sr != 16000:
        import torchaudio
        import torch
        waveform = torch.tensor(audio_array).unsqueeze(0).float()
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio_array = resampler(waveform).squeeze(0).numpy()
        sr = 16000

    # Save as wav
    sf.write(str(wav_path), audio_array, sr)

    # Compute MD5
    md5 = hashlib.md5(wav_path.read_bytes()).hexdigest()

    records.append({
        "utt_id": utt_id,
        "lang": LANG,
        "wav_path": str(wav_path).replace("\\", "/"),
        "ref_text": sample["transcript"],
        "ref_phon": None,
        "sr": sr,
        "duration_s": round(len(audio_array) / sr, 2),
        "audio_md5": md5,
        "snr_db": None
    })
    print(f"  [{i+1}/{N_SAMPLES}] {utt_id} — {sample['transcript'][:50]}")

# --- Write manifest atomically ---
with open(TMP_MANIFEST, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

TMP_MANIFEST.rename(OUT_MANIFEST)
print(f"\n✅ Manifest written to {OUT_MANIFEST}")