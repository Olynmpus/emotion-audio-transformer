"""Train the prosodic CNN-Transformer with waveform-level bandpass augmentation."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_features import load_mono, waveform_to_logmel, SAMPLE_RATE
from prepare_prosodic_features import prosodic_features
from train_prosodic import CNNTransformerAttention, evaluate


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_index_path(value):
    """Resolve Windows absolute paths stored in the index to WSL paths."""
    raw = str(value or "")
    slash = chr(92)

    if len(raw) >= 3 and raw[1] == ":" and raw[2] in (slash, "/"):
        drive = raw[0].lower()
        tail = raw[3:].replace(slash, "/")
        candidate = Path(f"/mnt/{drive}/{tail}")
        if candidate.exists():
            return candidate

    return Path(raw).expanduser()


def bandlimit(audio, low_hz, high_hz):
    """Fourth-order Butterworth bandpass at the model sample rate."""
    audio = np.asarray(audio, dtype=np.float32)
    sos = butter(
        4, [low_hz, high_hz],
        btype="bandpass", fs=SAMPLE_RATE, output="sos"
    )
    return sosfiltfilt(sos, audio).astype(np.float32)


def waveform_to_features(audio):
    """Recompute 40 Mel + 4 prosodic features from the same waveform."""
    mel = waveform_to_logmel(audio).astype(np.float32)
    prosody = prosodic_features(audio, sr=SAMPLE_RATE).astype(np.float32)
    combined = np.concatenate([mel, prosody], axis=1).astype(np.float32)

    if combined.shape != (201, 44):
        raise ValueError(f"Expected (201, 44), got {combined.shape}")

    return np.clip(combined, -5.0, 5.0).astype(np.float32)


def read_cached_feature(item):
    path = resolve_index_path(item["feature_path"])
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    with np.load(path) as data:
        feature = data["feature"].astype(np.float32)

    if feature.shape != (201, 44):
        raise ValueError(f"Expected (201, 44) in {path}, got {feature.shape}")

    return np.clip(feature, -5.0, 5.0).astype(np.float32)


class BandpassAugmentedDataset(Dataset):
    """
    Training samples randomly use clean, 300-3400 Hz, or 500-2500 Hz audio.
    Features for both filtered conditions are precomputed once and cached in RAM.
    """

    def __init__(self, items, label_to_id, augment=False):
        self.items = items
        self.label_to_id = label_to_id
        self.augment = augment
        self.features = []

        for i, item in enumerate(items, start=1):
            clean = read_cached_feature(item)

            if augment:
                source = resolve_index_path(item["source_path"])
                if not source.exists():
                    raise FileNotFoundError(f"Source audio not found: {source}")

                audio = load_mono(str(source))
                if audio is None or len(audio) == 0:
                    raise ValueError(f"Could not load usable audio: {source}")

                filtered_300_3400 = bandlimit(audio, 300.0, 3400.0)
                filtered_500_2500 = bandlimit(audio, 500.0, 2500.0)

                feature_300_3400 = waveform_to_features(filtered_300_3400)
                feature_500_2500 = waveform_to_features(filtered_500_2500)

                self.features.append({
                    "clean": clean,
                    "band_300_3400": feature_300_3400,
                    "band_500_2500": feature_500_2500,
                })
            else:
                self.features.append({"clean": clean})

            if i % 200 == 0 or i == len(items):
                print(
                    f"Prepared {'training' if augment else 'validation'} "
                    f"features: {i}/{len(items)}",
                    flush=True,
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        available = self.features[idx]

        if self.augment:
            # Every epoch can expose the recording to a different condition.
            condition = np.random.choice(
                ["clean", "band_300_3400", "band_500_2500"],
                p=[0.40, 0.30, 0.30],
            )
        else:
            condition = "clean"

        x = available[condition].copy()

        # Preserve the original training script's feature-domain augmentation.
        if self.augment:
            if np.random.rand() < 0.8:
                noise_std = np.random.uniform(0.02, 0.15)
                x += np.random.normal(
                    0.0, noise_std, size=x.shape
                ).astype(np.float32)

            if np.random.rand() < 0.5:
                n_mels = 40
                max_width = max(3, n_mels // 5)
                width = np.random.randint(2, max_width + 1)
                start = np.random.randint(0, n_mels - width + 1)
                attenuation = np.random.uniform(0.5, 0.9)
                x[:, start:start + width] *= attenuation

        x = np.clip(x, -5.0, 5.0).astype(np.float32)
        return torch.from_numpy(x), self.label_to_id[item["label"]]


def print_split_summary(name, items):
    speakers = sorted({str(item["speaker_id"]) for item in items})
    print(
        f"{name}: {len(items)} recordings; "
        f"speakers={speakers}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train emotion model with bandpass waveform augmentation."
    )
    parser.add_argument("--feature-dir", default="features_prosodic_4s")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs_bandpass_aug")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cpu")

    feature_dir = Path(args.feature_dir)
    if not feature_dir.is_absolute():
        feature_dir = ROOT / feature_dir

    index_path = feature_dir / "index.json"
    if not index_path.exists():
        raise SystemExit(f"Missing feature index: {index_path}")

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    labels = payload["labels"]
    label_to_id = {label: i for i, label in enumerate(labels)}

    items = sorted(
        payload["items"],
        key=lambda item: (
            item["speaker_id"], item["filename"], item["label"]
        ),
    )

    # Resolve paths before filtering, because the index stores Windows paths.
    items = [
        item for item in items
        if item.get("label") in label_to_id
        and item.get("speaker_id") is not None
        and resolve_index_path(item["feature_path"]).exists()
        and resolve_index_path(item["source_path"]).exists()
    ]

    if args.max_files > 0:
        items = items[:args.max_files]

    if len(items) < 30:
        raise SystemExit(f"Only {len(items)} usable recordings found.")

    groups = [item["speaker_id"] for item in items]

    # Match the original speaker-independent split exactly.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.30, random_state=args.seed
    )
    train_idx, remaining_idx = next(splitter.split(items, groups=groups))
    train_items = [items[i] for i in train_idx]
    remaining_items = [items[i] for i in remaining_idx]
    remaining_groups = [item["speaker_id"] for item in remaining_items]

    splitter2 = GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=args.seed + 1
    )
    val_idx, test_idx = next(
        splitter2.split(remaining_items, groups=remaining_groups)
    )
    val_items = [remaining_items[i] for i in val_idx]
    test_items = [remaining_items[i] for i in test_idx]

    train_speakers = {item["speaker_id"] for item in train_items}
    val_speakers = {item["speaker_id"] for item in val_items}
    test_speakers = {item["speaker_id"] for item in test_items}

    assert train_speakers.isdisjoint(val_speakers)
    assert train_speakers.isdisjoint(test_speakers)
    assert val_speakers.isdisjoint(test_speakers)

    print("\nSpeaker-independent data split")
    print_split_summary("Train", train_items)
    print_split_summary("Validation", val_items)
    print_split_summary("Test (held out)", test_items)
    print(f"Labels: {labels}")
    print("Training condition probabilities: clean=0.40, 300-3400=0.30, 500-2500=0.30")
    print("Validation condition: clean only")

    train_ds = BandpassAugmentedDataset(
        train_items, label_to_id, augment=True
    )
    val_ds = BandpassAugmentedDataset(
        val_items, label_to_id, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False
    )

    model = CNNTransformerAttention(
        n_mels=44, d_model=64, n_heads=4, layers=2,
        n_classes=len(labels), max_len=201
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.csv"
    checkpoint_path = out_dir / "best_model.pt"
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss",
            "val_accuracy", "val_macro_f1", "learning_rate"
        ])

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * x.size(0)

            train_loss = total_loss / len(train_ds)
            val_loss, val_acc, val_f1, _, _ = evaluate(
                model, val_loader, device, criterion
            )
            scheduler.step(val_f1)
            current_lr = optimizer.param_groups[0]["lr"]

            writer.writerow([
                epoch, train_loss, val_loss, val_acc, val_f1, current_lr
            ])
            f.flush()

            print(
                f"Epoch {epoch:02d}/{args.epochs} "
                f"| train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f} "
                f"| val_acc={val_acc:.3f} "
                f"| val_macroF1={val_f1:.3f} "
                f"| lr={current_lr:.2e}",
                flush=True,
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                epochs_without_improvement = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "labels": labels,
                        "config": {
                            "architecture": "CNNTransformerAttentionProsodic",
                            "n_mels": 44,
                            "d_model": 64,
                            "layers": 2,
                            "heads": 4,
                            "training_augmentation": [
                                "clean", "bandpass_300_3400",
                                "bandpass_500_2500"
                            ],
                        },
                        "epoch": epoch,
                        "val_macro_f1": val_f1,
                        "seed": args.seed,
                    },
                    checkpoint_path,
                )
                print(f"Saved best checkpoint: {checkpoint_path}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print("Early stopping: validation macro-F1 did not improve.")
                break

    print("\nTraining complete")
    print(f"Best epoch: {best_epoch}")
    print(f"Best clean validation macro-F1: {best_f1:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics: {metrics_path}")
    print("Held-out test recordings were not used for model selection.")


if __name__ == "__main__":
    main()
