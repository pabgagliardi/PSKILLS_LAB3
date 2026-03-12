import json
import torch
import soundfile as sf
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from phonemizer.backend import EspeakBackend

# Fix for Windows: tell phonemizer where espeak-ng is
EspeakBackend.set_library("C:\\Program Files\\eSpeak NG\\libespeak-ng.dll")

# --- Config ---
import yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)
LANG = params["lang"]
MODEL_ID = params["model_id"]
SNR_LEVELS = params["snr_levels"]

# --- Load model (only once) ---
print(f"Loading model {MODEL_ID}...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()
print("✅ Model loaded!")

# --- Process each SNR level ---
for snr in SNR_LEVELS:
    IN_MANIFEST = Path(f"data/manifests/{LANG}/noisy_snr{snr}.jsonl")
    OUT_MANIFEST = Path(f"data/manifests/{LANG}/predictions_snr{snr}.jsonl")
    TMP_MANIFEST = OUT_MANIFEST.with_suffix(".jsonl.tmp")

    with open(IN_MANIFEST, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    print(f"\nPredicting SNR = {snr} dB ({len(records)} utterances)...")
    predicted_records = []

    for i, record in enumerate(records):
        # Load audio
        audio, sr = sf.read(record["wav_path"])
        assert sr == 16000, f"Expected 16kHz, got {sr}Hz"

        # Run model
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**inputs).logits

        # Decode prediction
        predicted_ids = torch.argmax(logits, dim=-1)
        prediction = processor.decode(predicted_ids[0])

        # Save prediction in record
        record["hyp_phon"] = prediction
        predicted_records.append(record)
        print(f"  [{i+1}/{len(records)}] {record['utt_id']}")
        print(f"    ref:  {record['ref_phon'][:60]}")
        print(f"    hyp:  {prediction[:60]}")

    # Write manifest atomically
    with open(TMP_MANIFEST, "w", encoding="utf-8") as f:
        for r in predicted_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    TMP_MANIFEST.replace(OUT_MANIFEST)
    print(f"  ✅ Predictions written to {OUT_MANIFEST}")

print("\n✅ All predictions done!")