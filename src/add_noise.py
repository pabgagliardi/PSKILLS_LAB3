import json
import numpy as np
import soundfile as sf
from pathlib import Path

# --- Config ---
LANG = "fr"
IN_MANIFEST = Path(f"data/manifests/{LANG}/phonemized.jsonl")
SNR_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # dB

# --- Noise functions (provided in lab PDF) ---
def add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=signal.shape)
    return signal + noise

def add_noise_to_file(input_wav: str, output_wav: str, snr_db: float, seed: int = None) -> None:
    signal, sr = sf.read(input_wav)
    if signal.ndim != 1:
        raise ValueError("Only mono audio is supported")
    rng = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)
    sf.write(output_wav, noisy_signal, sr)

# --- Load clean manifest ---
with open(IN_MANIFEST, encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

# --- Process each SNR level ---
for snr in SNR_LEVELS:
    print(f"\nProcessing SNR = {snr} dB...")

    OUT_WAV_DIR = Path(f"data/raw/{LANG}/noisy_snr{snr}")
    OUT_MANIFEST = Path(f"data/manifests/{LANG}/noisy_snr{snr}.jsonl")
    TMP_MANIFEST = OUT_MANIFEST.with_suffix(".jsonl.tmp")
    OUT_WAV_DIR.mkdir(parents=True, exist_ok=True)

    noisy_records = []
    for record in records:
        input_wav = record["wav_path"]
        stem = Path(input_wav).stem
        output_wav = OUT_WAV_DIR / f"{stem}_snr{snr}.wav"

        # Add noise with a fixed seed for reproducibility
        seed = int(record["utt_id"].split("_")[-1])
        add_noise_to_file(input_wav, str(output_wav), snr_db=snr, seed=seed)

        # Create new record for noisy version
        noisy_record = record.copy()
        noisy_record["wav_path"] = str(output_wav).replace("\\", "/")
        noisy_record["snr_db"] = snr
        noisy_records.append(noisy_record)

    # Write manifest atomically
    with open(TMP_MANIFEST, "w", encoding="utf-8") as f:
        for r in noisy_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    TMP_MANIFEST.rename(OUT_MANIFEST)
    print(f"  ✅ Manifest written to {OUT_MANIFEST}")

print("\n✅ All noise levels done!")