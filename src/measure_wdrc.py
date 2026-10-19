import csv
import json
import numpy as np
import soundfile as sf
from pathlib import Path

from hearing_aid_wdrc import apply_wdrc

ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = ROOT / "features_prosodic_4s"
OUT = ROOT / "outputs_wdrc_eval" / "audio_measurements.csv"

CONDITIONS = [
    ("control_1to1", 1.0, 10.0, 100.0, False),
    ("wdrc_fast", 2.0, 10.0, 100.0, False),
    ("wdrc_fast_rms_matched", 2.0, 10.0, 100.0, True),
    ("wdrc_slow", 2.0, 50.0, 200.0, False),
    ("wdrc_slow_rms_matched", 2.0, 50.0, 200.0, True),
]

def measurements(x, sr):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    eps = 1e-12
    rms = np.sqrt(np.mean(x * x) + eps)
    peak = np.max(np.abs(x)) if len(x) else 0.0

    # RMS envelope in 10 ms frames
    frame = max(1, int(sr * 0.01))
    usable = (len(x) // frame) * frame
    frames = x[:usable].reshape(-1, frame) if usable else x.reshape(1, -1)
    env = np.sqrt(np.mean(frames * frames, axis=1) + eps)
    env_db = 20 * np.log10(env + eps)

    return {
        "rms_dbfs": 20 * np.log10(rms + eps),
        "peak_dbfs": 20 * np.log10(peak + eps),
        "crest_factor_db": 20 * np.log10((peak + eps) / (rms + eps)),
        "envelope_sd_db": float(np.std(env_db)),
        "envelope_p95_minus_p05_db": float(
            np.percentile(env_db, 95) - np.percentile(env_db, 5)
        ),
        "clipped_percent": 100.0 * np.mean(np.abs(x) >= 0.999),
    }

index = json.loads((FEATURE_DIR / "index.json").read_text(encoding="utf-8"))
items = index if isinstance(index, list) else index.get("items", [])

rows = []
for item in items:
    # Support common index layouts; skip records without an audio path.
    rel = item.get("source_path") or item.get("audio_path") or item.get("path") or item.get("wav")
    if not rel:
        continue

    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        continue

    signal, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    base = measurements(signal, sr)
    for name, ratio, attack, release, match_rms in CONDITIONS:
        processed, _ = apply_wdrc(
            signal, sr,
            ratio=ratio,
            attack_ms=attack,
            release_ms=release,
            knee_dbfs=-30.0,
            low_level_gain_db=0.0,
        )
        processed = np.asarray(processed, dtype=np.float32)

        if match_rms:
            before = np.sqrt(np.mean(signal.astype(np.float64)**2) + 1e-20)
            after = np.sqrt(np.mean(processed.astype(np.float64)**2) + 1e-20)
            if after > 1e-12:
                processed *= before / after

        m = measurements(processed, sr)
        rows.append({
            "filename": path.name,
            "condition": name,
            **{f"input_{k}": v for k, v in base.items()},
            **m,
            "rms_change_db": m["rms_dbfs"] - base["rms_dbfs"],
        })

if not rows:
    raise RuntimeError(
        "No audio paths found in index.json. No output written; inspect its structure."
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Measured {len(rows)} processed recordings.")
print(f"Saved: {OUT}")
print("Mean results by condition:")
for name, *_ in CONDITIONS:
    subset = [r for r in rows if r["condition"] == name]
    if subset:
        print(
            f"{name:26s} "
            f"RMS change={np.mean([r['rms_change_db'] for r in subset]):7.3f} dB | "
            f"envelope SD={np.mean([r['envelope_sd_db'] for r in subset]):6.2f} dB | "
            f"crest={np.mean([r['crest_factor_db'] for r in subset]):6.2f} dB"
        )
