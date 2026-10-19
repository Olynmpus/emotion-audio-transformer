import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

from audio_features import load_mono
from evaluate_channels import (
    ROOT,
    resolve_index_path,
    audio_to_model_features,
    run_model,
    bandlimit,
    match_rms,
    rms,
)
from train_prosodic import CNNTransformerAttention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default="features_prosodic_4s")
    parser.add_argument("--checkpoint",
                        default="outputs_prosodic_4s/best_model.pt")
    parser.add_argument("--output-dir",
                        default="outputs_feature_ablation")
    parser.add_argument("--test-speakers", default="10,14,17,19")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    feature_dir = (ROOT / args.feature_dir).resolve()
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = feature_dir / "index.json"
    if not index_path.exists():
        raise SystemExit(f"Missing index: {index_path}")
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    labels = payload["labels"]
    label_to_id = {label: i for i, label in enumerate(labels)}
    speakers = {x.strip() for x in args.test_speakers.split(",") if x.strip()}

    items = [
        item for item in payload["items"]
        if str(item.get("speaker_id")) in speakers
        and item.get("label") in label_to_id
        and resolve_index_path(item.get("feature_path", "")).exists()
        and resolve_index_path(item.get("source_path", "")).exists()
    ]
    items.sort(key=lambda x: (str(x.get("speaker_id")), x.get("filename", "")))

    if args.max_files > 0:
        items = items[:args.max_files]
    if not items:
        raise SystemExit("No eligible test recordings found.")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if list(checkpoint.get("labels", labels)) != list(labels):
        raise SystemExit("Label order differs between index and checkpoint.")

    model = CNNTransformerAttention(
        n_mels=44, d_model=64, n_heads=4, layers=2,
        n_classes=len(labels), max_len=201,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test recordings: {len(items)}")
    print(f"Test speakers: {sorted(speakers)}")

    rows = []

    def record(item, condition, ablation, level_mode, feature):
        pred_id, confidence, probs = run_model(model, feature, device)
        true_label = item["label"]
        rows.append({
            "filename": item["filename"],
            "speaker_id": item["speaker_id"],
            "true_label": true_label,
            "condition": condition,
            "ablation": ablation,
            "level_mode": level_mode,
            "predicted_label": labels[pred_id],
            "confidence": confidence,
            "correct": int(labels[pred_id] == true_label),
            "probabilities": json.dumps(
                {label: float(p) for label, p in zip(labels, probs)}
            ),
        })

    for i, item in enumerate(items, 1):
        with np.load(resolve_index_path(item["feature_path"])) as data:
            cached_clean = np.clip(
                data["feature"].astype(np.float32), -5.0, 5.0
            )

        clean_audio = load_mono(resolve_index_path(item["source_path"]))
        clean_features = audio_to_model_features(clean_audio)
        clean_mel = clean_features[:, :40]
        clean_prosody = clean_features[:, 40:]

        # Baseline uses the exact cached inputs from the original evaluation.
        record(
            item, "clean_baseline", "all_features",
            "native_level", cached_clean
        )

        filtered_conditions = {
            "bandpass_300_3400": bandlimit(clean_audio, 300, 3400),
            "bandpass_500_2500": bandlimit(clean_audio, 500, 2500),
        }

        for condition, filtered_audio in filtered_conditions.items():
            for level_mode in ("native_level", "rms_matched"):
                audio = (
                    filtered_audio
                    if level_mode == "native_level"
                    else match_rms(filtered_audio, rms(clean_audio))
                )

                filtered_features = audio_to_model_features(audio)
                filtered_mel = filtered_features[:, :40]
                filtered_prosody = filtered_features[:, 40:]

                # 1. Both feature groups come from filtered audio.
                record(
                    item, condition, "all_filtered_features",
                    level_mode, filtered_features
                )

                # 2. Filtered mel + original clean-audio prosody.
                hybrid_clean_prosody = np.concatenate(
                    [filtered_mel, clean_prosody], axis=1
                )
                hybrid_clean_prosody = np.clip(
                    hybrid_clean_prosody, -5.0, 5.0
                )
                record(
                    item, condition, "filtered_mel_clean_prosody",
                    level_mode, hybrid_clean_prosody
                )

                # 3. Original clean mel + filtered-audio prosody.
                hybrid_clean_mel = np.concatenate(
                    [clean_mel, filtered_prosody], axis=1
                )
                hybrid_clean_mel = np.clip(
                    hybrid_clean_mel, -5.0, 5.0
                )
                record(
                    item, condition, "clean_mel_filtered_prosody",
                    level_mode, hybrid_clean_mel
                )

        if i % 25 == 0 or i == len(items):
            print(f"Processed {i}/{len(items)} recordings")

    predictions_path = output_dir / "predictions.csv"
    fieldnames = [
        "filename", "speaker_id", "true_label", "condition", "ablation",
        "level_mode", "predicted_label", "confidence", "correct",
        "probabilities",
    ]
    with predictions_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for condition in sorted({r["condition"] for r in rows}):
        for ablation in sorted({
            r["ablation"] for r in rows if r["condition"] == condition
        }):
            for level_mode in sorted({
                r["level_mode"] for r in rows
                if r["condition"] == condition
                and r["ablation"] == ablation
            }):
                subset = [
                    r for r in rows
                    if r["condition"] == condition
                    and r["ablation"] == ablation
                    and r["level_mode"] == level_mode
                ]
                y_true = [r["true_label"] for r in subset]
                y_pred = [r["predicted_label"] for r in subset]

                entry = {
                    "condition": condition,
                    "ablation": ablation,
                    "level_mode": level_mode,
                    "n": len(subset),
                    "accuracy": accuracy_score(y_true, y_pred),
                    "macro_f1": f1_score(
                        y_true, y_pred, labels=labels,
                        average="macro", zero_division=0
                    ),
                    "mean_confidence": float(np.mean([
                        r["confidence"] for r in subset
                    ])),
                }
                summary.append(entry)

                print(
                    f"{condition:22s} | {ablation:29s} | "
                    f"{level_mode:12s} | "
                    f"acc={entry['accuracy']:.3f} "
                    f"macro-F1={entry['macro_f1']:.3f} "
                    f"conf={entry['mean_confidence']:.3f}"
                )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    report_path = output_dir / "classification_reports.txt"
    with report_path.open("w", encoding="utf-8") as f:
        for entry in summary:
            subset = [
                r for r in rows
                if r["condition"] == entry["condition"]
                and r["ablation"] == entry["ablation"]
                and r["level_mode"] == entry["level_mode"]
            ]
            f.write(
                f"\n{'=' * 78}\n"
                f"{entry['condition']} | {entry['ablation']} | "
                f"{entry['level_mode']}\n\n"
            )
            f.write(classification_report(
                [r["true_label"] for r in subset],
                [r["predicted_label"] for r in subset],
                labels=labels, zero_division=0
            ))

    print(f"\nSaved predictions: {predictions_path}")
    print(f"Saved summary:     {summary_path}")
    print(f"Saved reports:     {report_path}")


if __name__ == "__main__":
    main()
