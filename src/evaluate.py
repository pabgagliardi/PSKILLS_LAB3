import json
import matplotlib.pyplot as plt
from pathlib import Path
from jiwer import cer  # CER == PER when each phoneme is one character

# --- Config ---
import yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)
LANG = params["lang"]
SNR_LEVELS = params["snr_levels"]

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

def compute_per(ref: str, hyp: str) -> float:
    """Compute Phoneme Error Rate using character error rate."""
    # Normalize: remove spaces so each phoneme is treated as one unit
    ref_clean = " ".join(ref.replace(" ", ""))
    hyp_clean = " ".join(hyp.replace(" ", ""))
    return cer(ref_clean, hyp_clean)

# --- Compute PER for each SNR level ---
per_scores = {}

for snr in SNR_LEVELS:
    manifest = Path(f"data/manifests/{LANG}/predictions_snr{snr}.jsonl")

    with open(manifest, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    pers = []
    for record in records:
        ref = record.get("ref_phon", "")
        hyp = record.get("hyp_phon", "")
        if ref and hyp:
            pers.append(compute_per(ref, hyp))

    mean_per = sum(pers) / len(pers)
    per_scores[snr] = mean_per
    print(f"  SNR {snr:3d} dB → PER = {mean_per:.3f} ({mean_per*100:.1f}%)")

# --- Save metrics to JSON ---
metrics_path = OUT_DIR / f"per_{LANG}.json"
with open(metrics_path, "w") as f:
    json.dump({str(k): round(v, 4) for k, v in per_scores.items()}, f, indent=2)
print(f"\n✅ Metrics saved to {metrics_path}")

# --- Plot PER vs SNR ---
snrs = list(per_scores.keys())
pers = [per_scores[s] * 100 for s in snrs]  # convert to %

plt.figure(figsize=(8, 5))
plt.plot(snrs, pers, marker="o", linewidth=2, color="steelblue", label=f"{LANG.upper()}")
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("PER (%)", fontsize=12)
plt.title(f"Phoneme Error Rate vs Noise Level ({LANG.upper()})", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = OUT_DIR / f"per_vs_snr_{LANG}.png"
plt.savefig(str(plot_path), dpi=150)
plt.show()
print(f"✅ Plot saved to {plot_path}")