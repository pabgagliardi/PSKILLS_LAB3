import json
import subprocess
from pathlib import Path
import yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANG = params["lang"]
IN_MANIFEST = Path(f"data/manifests/{LANG}/clean.jsonl")
OUT_MANIFEST = Path(f"data/manifests/{LANG}/phonemized.jsonl")
TMP_MANIFEST = OUT_MANIFEST.with_suffix(".jsonl.tmp")

# espeak-ng language codes
ESPEAK_LANG = {
    "fr": "fr",
    "es": "es",
    "de": "de",
}

def phonemize(text: str, lang: str) -> str:
    """Convert text to phonemes using espeak-ng."""
    result = subprocess.run(
        ["espeak-ng", "-q", "--ipa", "-v", ESPEAK_LANG[lang], text],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    # Clean up whitespace
    return " ".join(result.stdout.strip().split())

# --- Process manifest ---
records = []
with open(IN_MANIFEST, encoding="utf-8") as f:
    lines = f.readlines()

print(f"Phonemizing {len(lines)} utterances...")
for i, line in enumerate(lines):
    record = json.loads(line)
    record["ref_phon"] = phonemize(record["ref_text"], record["lang"])
    records.append(record)
    print(f"  [{i+1}/{len(lines)}] {record['utt_id']}")
    print(f"    text:  {record['ref_text'][:60]}")
    print(f"    phon:  {record['ref_phon'][:60]}")

# --- Write manifest atomically ---
with open(TMP_MANIFEST, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

TMP_MANIFEST.replace(OUT_MANIFEST)
print(f"\n✅ Manifest written to {OUT_MANIFEST}")