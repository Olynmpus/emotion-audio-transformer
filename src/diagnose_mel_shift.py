import argparse
import csv
import json
from pathlib import Path

import numpy as np
from audio_features import load_mono
from evaluate_channels import (
    ROOT, resolve_index_path, audio_to_model_features, bandlimit
)


def save_csv(path, rows):
    if not rows:
        print(f"No rows to write: {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default="features_prosodic_4s")
    parser.add_argument("--test-speakers", default="10,14,17,19")
    parser.add_argument("--output-dir", default="outputs_mel_shift")
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    feature_dir = (ROOT / args.feature_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(
        (feature_dir / "index.json").read_text(encoding="utf-8")
    )
    labels = set(payload["labels"])
    speakers = {s.strip() for s in args.test_speakers.split(",") if s.strip()}

    items = [
        x for x in payload["items"]
        if str(x.get("speaker_id")) in speakers
        and x.get("label") in labels
        and resolve_index_path(x.get("feature_path", "")).exists()
        and resolve_index_path(x.get("source_path", "")).exists()
    ]
    items.sort(key=lambda x: (str(x.get("speaker_id")), x.get("filename", "")))

    if args.max_files:
        items = items[:args.max_files]
    if not items:
        raise SystemExit("No eligible recordings found.")

    conditions = {
        "bandpass_300_3400": (300, 3400),
        "bandpass_500_2500": (500, 2500),
    }

    # Per-mel-bin comparisons. Mel features occupy columns 0..39.
    bin_rows = []
    summary_rows = []
    recording_rows = []

    for i, item in enumerate(items, 1):
        with np.load(resolve_index_path(item["feature_path"])) as data:
            cached = np.clip(data["feature"].astype(np.float32), -5, 5)

        clean_audio = load_mono(resolve_index_path(item["source_path"]))
        clean_features = audio_to_model_features(clean_audio)
        clean_mel = clean_features[:, :40]
        cached_mel = cached[:, :40]

        # Check cached versus freshly extracted clean mel features.
        parity = cached_mel - clean_mel
        recording_rows.append({
            "filename": item["filename"],
            "speaker_id": item["speaker_id"],
            "label": item["label"],
            "condition": "clean_cache_vs_recomputed",
            "mel_mae": float(np.mean(np.abs(parity))),
            "mel_rmse": float(np.sqrt(np.mean(parity ** 2))),
            "mel_max_abs_diff": float(np.max(np.abs(parity))),
            "mel_changed_over_0_1_fraction": float(np.mean(np.abs(parity) > 0.1)),
        })

        for condition, (low, high) in conditions.items():
            filtered_audio = bandlimit(clean_audio, low, high)
            filtered_mel = audio_to_model_features(filtered_audio)[:, :40]
            delta = filtered_mel - clean_mel

            recording_rows.append({
                "filename": item["filename"],
                "speaker_id": item["speaker_id"],
                "label": item["label"],
                "condition": condition,
                "mel_mae": float(np.mean(np.abs(delta))),
                "mel_rmse": float(np.sqrt(np.mean(delta ** 2))),
                "mel_max_abs_diff": float(np.max(np.abs(delta))),
                "mel_changed_over_0_1_fraction": float(np.mean(np.abs(delta) > 0.1)),
            })

            for b in range(40):
                c = clean_mel[:, b].reshape(-1)
                f = filtered_mel[:, b].reshape(-1)
                d = f - c
                bin_rows.append({
                    "condition": condition,
                    "mel_bin": b,
                    "clean_mean": float(np.mean(c)),
                    "filtered_mean": float(np.mean(f)),
                    "mean_shift": float(np.mean(d)),
                    "clean_std": float(np.std(c)),
                    "filtered_std": float(np.std(f)),
                    "std_ratio_filtered_over_clean": float(
                        np.std(f) / max(float(np.std(c)), 1e-8)
                    ),
                    "mean_absolute_change": float(np.mean(np.abs(d))),
                    "rmse_change": float(np.sqrt(np.mean(d ** 2))),
                    "fraction_abs_change_over_0_1": float(np.mean(np.abs(d) > 0.1)),
                })

        if i % 25 == 0 or i == len(items):
            print(f"Processed {i}/{len(items)} recordings")

    save_csv(output_dir / "mel_bin_statistics.csv", bin_rows)
    save_csv(output_dir / "recording_differences.csv", recording_rows)

    # Aggregate recording-level feature changes by condition.
    for condition in ["clean_cache_vs_recomputed", *conditions.keys()]:
        rows = [r for r in recording_rows if r["condition"] == condition]
        summary_rows.append({
            "condition": condition,
            "n_recordings": len(rows),
            "mean_mel_mae": float(np.mean([r["mel_mae"] for r in rows])),
            "median_mel_mae": float(np.median([r["mel_mae"] for r in rows])),
            "mean_mel_rmse": float(np.mean([r["mel_rmse"] for r in rows])),
            "mean_max_abs_diff": float(np.mean([r["mel_max_abs_diff"] for r in rows])),
            "mean_fraction_abs_change_over_0_1": float(
                np.mean([r["mel_changed_over_0_1_fraction"] for r in rows])
            ),
        })
    save_csv(output_dir / "mel_distribution_summary.csv", summary_rows)

    # Rank the bins with the largest average absolute changes.
    ranked_rows = []
    for condition in conditions:
        rows = [r for r in bin_rows if r["condition"] == condition]
        rows.sort(key=lambda r: r["mean_absolute_change"], reverse=True)
        for rank, row in enumerate(rows, 1):
            ranked_rows.append({
                "condition": condition,
                "rank": rank,
                **row,
            })
    save_csv(output_dir / "mel_bins_ranked.csv", ranked_rows)

    print(f"\nSaved diagnostic results to: {output_dir}")
    print("\nSummary:")
    for row in summary_rows:
        print(
            f'{row["condition"]:28s} '
            f'MAE={row["mean_mel_mae"]:.6f} '
            f'RMSE={row["mean_mel_rmse"]:.6f} '
            f'fraction>|0.1|={row["mean_fraction_abs_change_over_0_1"]:.3%}'
        )
    print("\nLargest mel-bin changes:")
    for condition in conditions:
        rows = [r for r in ranked_rows if r["condition"] == condition]
        print(condition)
        for row in rows[:8]:
            print(
                f'  bin {row["mel_bin"]:2d}: '
                f'MAE={row["mean_absolute_change"]:.4f}, '
                f'mean shift={row["mean_shift"]:+.4f}, '
                f'std ratio={row["std_ratio_filtered_over_clean"]:.3f}'
            )


if __name__ == "__main__":
    main()
