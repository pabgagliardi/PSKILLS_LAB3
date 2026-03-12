import json
import matplotlib.pyplot as plt
from pathlib import Path
from jiwer import cer
import yaml

# --- Config ---
with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANG = params["lang"]
SNR_LEVELS = params["snr_levels"]
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

def compute_per(ref: str, hyp: str) -> float:
    """Compute Phoneme Error Rate using character error rate."""
    ref_clean = " ".join(ref.replace(" ", ""))
    hyp_clean = " ".join(hyp.replace(" ", ""))
    return cer(ref_clean, hyp_clean)

# --- Compute PER for current language ---
print(f"Evaluating language: {LANG}")
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

# --- Save current language metrics ---
metrics_path = OUT_DIR / f"per_{LANG}.json"
with open(metrics_path, "w") as f:
    json.dump({str(k): round(v, 4) for k, v in per_scores.items()}, f, indent=2)
print(f"\n✅ Metrics saved to {metrics_path}")

# --- Load ALL language results for the final plot ---
all_results = {}
for metrics_file in OUT_DIR.glob("per_??.json"):
    lang_code = metrics_file.stem.replace("per_", "")
    with open(metrics_file) as f:
        data = json.load(f)
    # Convert string keys back to integers
    all_results[lang_code] = {int(k): v for k, v in data.items()}

print(f"\nLanguages found for plotting: {list(all_results.keys())}")

# --- Plot all languages + mean ---
plt.figure(figsize=(10, 6))
colors = ["steelblue", "tomato", "seagreen", "orange", "purple"]

all_pers_per_snr = {snr: [] for snr in SNR_LEVELS}

for i, (lang_code, scores) in enumerate(all_results.items()):
    snrs = sorted(scores.keys())
    pers = [scores[s] * 100 for s in snrs]

    # Collect for mean calculation
    for snr in SNR_LEVELS:
        if snr in scores:
            all_pers_per_snr[snr].append(scores[snr])

    color = colors[i % len(colors)]
    plt.plot(snrs, pers, marker="o", linewidth=2,
             color=color, label=lang_code.upper())

# Plot mean curve (only if more than one language)
if len(all_results) > 1:
    mean_pers = []
    for snr in SNR_LEVELS:
        values = all_pers_per_snr[snr]
        if values:
            mean_pers.append(sum(values) / len(values) * 100)
        else:
            mean_pers.append(None)

    plt.plot(SNR_LEVELS, mean_pers, marker="s", linewidth=2.5,
             linestyle="--", color="black", label="MEAN")

plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("PER (%)", fontsize=12)
plt.title("Phoneme Error Rate vs Noise Level", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = OUT_DIR / "per_vs_snr_all.png"
plt.savefig(str(plot_path), dpi=150)
plt.show()
print(f"✅ Final plot saved to {plot_path}")