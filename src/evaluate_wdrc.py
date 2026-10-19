import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from audio_features import load_mono, waveform_to_logmel
from prepare_prosodic_features import prosodic_features
from train_prosodic import CNNTransformerAttention
from hearing_aid_wdrc import apply_wdrc


ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = ROOT / "features_prosodic_4s"
CHECKPOINT_PATH = ROOT / "outputs_prosodic_4s" / "best_model.pt"
OUTPUT_DIR = ROOT / "outputs_wdrc_eval"
SEED = 42

DEVICE = torch.device("cpu")
torch.set_num_threads(2)


def make_features(audio):
    """Use the same frontend and feature ordering as Experiment C."""
    mel = waveform_to_logmel(audio)
    prosody = prosodic_features(audio)

    combined = np.concatenate([mel, prosody], axis=1)
    if combined.shape != (201, 44):
        raise ValueError(f"Unexpected feature shape: {combined.shape}")

    # Match the float16 storage and clipping used during training.
    return np.clip(combined.astype(np.float16).astype(np.float32),
                   -5.0, 5.0)


def predict(model, features):
    x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return int(np.argmax(probs)), probs


def rms(signal):
    signal = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean(signal * signal) + 1e-20))


def process_to_features(signal, sr, ratio, attack_ms, release_ms,
                        temp_path, match_rms=False):
    """Apply WDRC at the source sample rate, then run the normal frontend."""
    processed, _stats = apply_wdrc(
        signal,
        sr,
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms,
        knee_dbfs=-30.0,
        low_level_gain_db=0.0,
    )

    processed = np.asarray(processed, dtype=np.float32)

    if match_rms:
        before = rms(signal)
        after = rms(processed)
        if after > 1e-12:
            processed = processed * (before / after)

    # FLOAT preserves the processed waveform without PCM16 re-quantisation.
    sf.write(str(temp_path), processed, sr, subtype="FLOAT")
    audio = load_mono(temp_path)
    return make_features(audio)


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    index_path = FEATURE_DIR / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    labels = payload["labels"]
    label_to_id = {label: i for i, label in enumerate(labels)}

    # Match the filtering and speaker split used by train_prosodic.py.
    items = [
        item for item in payload["items"]
        if item.get("label") in label_to_id
        and item.get("speaker_id") is not None
        and Path(item["feature_path"]).exists()
        and Path(item["source_path"]).exists()
    ]

    if len(items) < 30:
        raise RuntimeError(f"Only {len(items)} usable indexed files found.")

    groups = [item["speaker_id"] for item in items]
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.30, random_state=SEED
    )
    train_idx, remaining_idx = next(
        splitter.split(items, groups=groups)
    )
    remaining_items = [items[i] for i in remaining_idx]
    remaining_groups = [item["speaker_id"] for item in remaining_items]

    splitter2 = GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=SEED + 1
    )
    _, test_idx = next(
        splitter2.split(remaining_items, groups=remaining_groups)
    )
    test_items = [remaining_items[i] for i in test_idx]

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    if config.get("n_mels") != 44:
        raise ValueError(f"Unexpected checkpoint config: {config}")

    model = CNNTransformerAttention(
        n_mels=44,
        d_model=64,
        n_heads=4,
        layers=2,
        n_classes=len(labels),
        max_len=201,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Checkpoint seed: {checkpoint.get('seed')}")
    print(f"Indexed files: {len(items)}")
    print(f"Held-out files: {len(test_items)}")
    print("Held-out speakers:",
          sorted({item["speaker_id"] for item in test_items}))
    print("Conditions: saved baseline, recomputed baseline, 1:1 control,")
    print("            fast/slow WDRC, and RMS-matched fast/slow WDRC")

    conditions = [
        "baseline_saved",
        "baseline_recomputed",
        "control_1to1",
        "wdrc_fast",
        "wdrc_fast_rms_matched",
        "wdrc_slow",
        "wdrc_slow_rms_matched",
    ]
    truth = []
    predictions = {name: [] for name in conditions}
    prediction_rows = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="wdrc_eval_") as temp_dir:
        temp_dir = Path(temp_dir)

        for n, item in enumerate(test_items, start=1):
            source_path = Path(item["source_path"])
            y = label_to_id[item["label"]]
            truth.append(y)

            # Exact baseline representation that the model was trained on.
            with np.load(item["feature_path"]) as data:
                saved_features = np.clip(
                    data["feature"].astype(np.float32), -5.0, 5.0
                )
            pred, probs = predict(model, saved_features)
            predictions["baseline_saved"].append(pred)
            prediction_rows.append([
                item["filename"], item["speaker_id"], item["label"],
                "baseline_saved", labels[pred], int(pred == y),
                float(np.max(probs))
            ])

            # Recompute features from the original waveform as a sanity check.
            original_features = make_features(load_mono(source_path))
            pred, probs = predict(model, original_features)
            predictions["baseline_recomputed"].append(pred)
            prediction_rows.append([
                item["filename"], item["speaker_id"], item["label"],
                "baseline_recomputed", labels[pred], int(pred == y),
                float(np.max(probs))
            ])

            raw, sr = sf.read(
                str(source_path), dtype="float32", always_2d=True
            )
            signal = raw.mean(axis=1).astype(np.float32)

            # 1:1 is a control for the processing code and feature pipeline.
            # Fast and slow conditions use the same original waveform.
            variants = [
                ("control_1to1", 1.0, 10.0, 100.0, False),
                ("wdrc_fast", 2.0, 10.0, 100.0, False),
                ("wdrc_fast_rms_matched", 2.0, 10.0, 100.0, True),
                ("wdrc_slow", 2.0, 50.0, 200.0, False),
                ("wdrc_slow_rms_matched", 2.0, 50.0, 200.0, True),
            ]

            for name, ratio, attack, release, matched in variants:
                temp_path = temp_dir / f"{n}_{name}.wav"
                features = process_to_features(
                    signal, sr, ratio, attack, release,
                    temp_path, match_rms=matched
                )
                pred, probs = predict(model, features)
                predictions[name].append(pred)
                prediction_rows.append([
                    item["filename"], item["speaker_id"], item["label"],
                    name, labels[pred], int(pred == y),
                    float(np.max(probs))
                ])

            if n % 25 == 0 or n == len(test_items):
                print(f"Processed {n}/{len(test_items)} held-out files")

    summary_rows = []
    report_parts = []

    for name in conditions:
        pred = predictions[name]
        acc = accuracy_score(truth, pred)
        macro_f1 = f1_score(
            truth, pred, labels=list(range(len(labels))),
            average="macro", zero_division=0
        )
        summary_rows.append([name, len(truth), acc, macro_f1])

        report_parts.append(
            f"=== {name} ===\n"
            f"Accuracy: {acc:.4f}\n"
            f"Macro-F1: {macro_f1:.4f}\n"
            + classification_report(
                truth, pred, labels=list(range(len(labels))),
                target_names=labels, zero_division=0
            )
        )

        print(f"{name:26s} accuracy={acc:.4f} macro-F1={macro_f1:.4f}")

    saved = np.asarray(predictions["baseline_saved"])
    recomputed = np.asarray(predictions["baseline_recomputed"])
    agreement = float(np.mean(saved == recomputed))
    print(f"\nSaved-vs-recomputed baseline prediction agreement: {agreement:.1%}")

    with (OUTPUT_DIR / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "n_test", "accuracy", "macro_f1"])
        writer.writerows(summary_rows)

    with (OUTPUT_DIR / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "speaker_id", "true_label", "condition",
            "predicted_label", "correct", "confidence"
        ])
        writer.writerows(prediction_rows)

    (OUTPUT_DIR / "classification_reports.txt").write_text(
        "\n\n".join(report_parts)
        + f"\n\nSaved-vs-recomputed baseline prediction agreement: {agreement:.4f}\n",
        encoding="utf-8"
    )

    print(f"\nResults saved in: {OUTPUT_DIR}")
    print("Files: summary.csv, predictions.csv, classification_reports.txt")
    if agreement < 0.98:
        print(
            "WARNING: baseline agreement is below 98%. Inspect the feature "
            "pipeline and saved-feature consistency before interpreting WDRC."
        )


if __name__ == "__main__":
    main()
