import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FeatureDataset(Dataset):
    def __init__(self, items, label_to_id, augment=False):
        self.items = items
        self.label_to_id = label_to_id
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        with np.load(item["feature_path"]) as data:
            clean = data["feature"].astype(np.float32)

        clean = np.clip(clean, -5, 5)
        x = clean.copy()

        if self.augment:
            # Moderate feature-domain augmentation.
            if np.random.rand() < 0.8:
                noise_std = np.random.uniform(0.02, 0.15)
                x += np.random.normal(
                    0, noise_std, size=x.shape
                ).astype(np.float32)

            if np.random.rand() < 0.5:
                n_mels = x.shape[1]
                max_width = max(3, n_mels // 5)
                width = np.random.randint(2, max_width + 1)
                start = np.random.randint(0, max(1, n_mels - width + 1))
                attenuation = np.random.uniform(0.5, 0.9)
                x[:, start:start + width] *= attenuation

        return (
            torch.from_numpy(x),
            self.label_to_id[item["label"]],
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=201):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError(
                f"Input has {x.shape[1]} frames but positional encoding "
                f"supports {self.pe.shape[1]}."
            )
        return x + self.pe[:, :x.shape[1]]


class CNNTransformerMeanPool(nn.Module):
    """
    Classification-focused acoustic model:
    CNN time-frequency features -> Transformer temporal context
    -> learned temporal attention pooling -> emotion logits.
    Input shape: [batch, time_frames, mel_bands].
    """

    def __init__(
        self,
        n_mels=40,
        d_model=64,
        n_heads=4,
        layers=2,
        n_classes=7,
        max_len=201,
        dropout=0.2,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            # Downsample frequency while preserving the time axis.
            nn.Conv2d(
                16, 32, kernel_size=3, stride=(1, 2), padding=1
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # For 40 Mel bands, the stride-(1, 2) convolution produces 20 bins.
        self.input_proj = nn.Linear(32 * ((n_mels + 1) // 2), d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers
        )

        # Experiment B: no learned temporal attention. Every frame contributes
        # equally to the utterance representation through mean pooling.

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x, return_attention=False):
        # [B, T, F] -> [B, 1, T, F]
        x = x.unsqueeze(1)
        x = self.cnn(x)                    # [B, C, T, F']
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.flatten(start_dim=2)         # [B, T, C*F']
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)

        pooled = x.mean(dim=1)  # [B, D]: equal weight for every time frame
        logits = self.classifier(pooled)

        if return_attention:
            # Preserve the evaluation/export interface: these are uniform
            # pooling weights, not learned attention scores.
            weights = torch.full(
                (x.shape[0], x.shape[1]),
                1.0 / x.shape[1],
                dtype=x.dtype,
                device=x.device,
            )
            return logits, weights
        return logits


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            labels = labels.to(device)
            logits = model(x)
            loss = criterion(logits, labels)
            total_loss += loss.item() * x.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    average_loss = total_loss / max(1, len(loader.dataset))
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    return average_loss, accuracy, macro_f1, y_true, y_pred


def print_split_summary(name, items):
    speakers = {item["speaker_id"] for item in items}
    counts = {}
    for item in items:
        label = item["label"]
        counts[label] = counts.get(label, 0) + 1
    print(f"{name}: {len(items)} files, {len(speakers)} speakers")
    print(f"  Class counts: {counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs_attention")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cpu")

    feature_dir = Path(args.feature_dir)
    index_path = feature_dir / "index.json"
    if not index_path.exists():
        raise SystemExit(
            f"Missing {index_path}. Run prepare_features.py first."
        )

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    items = sorted(
        payload["items"],
        key=lambda item: (
            item["speaker_id"], item["filename"], item["label"]
        ),
    )
    labels = payload["labels"]
    label_to_id = {label: i for i, label in enumerate(labels)}

    if args.max_files > 0:
        items = items[:args.max_files]

    items = [
        item for item in items
        if item.get("label") in label_to_id
        and item.get("speaker_id") is not None
        and Path(item["feature_path"]).exists()
    ]
    if len(items) < 30:
        raise SystemExit(f"Only {len(items)} usable files. Prepare more data.")

    groups = [item["speaker_id"] for item in items]
    unique_speakers = set(groups)
    if len(unique_speakers) < 3:
        raise SystemExit("At least 3 distinct speaker IDs are required.")

    # Keep the same speaker-independent split strategy as the baseline.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.30, random_state=args.seed
    )
    train_idx, remaining_idx = next(splitter.split(items, groups=groups))
    train_items = [items[i] for i in train_idx]
    remaining_items = [items[i] for i in remaining_idx]
    remaining_groups = [item["speaker_id"] for item in remaining_items]

    if len(set(remaining_groups)) < 2:
        raise SystemExit(
            "Not enough remaining speakers for separate validation/test sets."
        )

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
    print_split_summary("Test", test_items)
    print(f"Labels (class index order): {labels}")

    train_ds = FeatureDataset(train_items, label_to_id, augment=True)
    val_ds = FeatureDataset(val_items, label_to_id, augment=False)
    test_ds = FeatureDataset(test_items, label_to_id, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False
    )

    model = CNNTransformerMeanPool(
        n_mels=40,
        d_model=64,
        n_heads=4,
        layers=2,
        n_classes=len(labels),
        max_len=201,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    out_dir = Path(args.output_dir)
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
                x = x.to(device)
                y = y.to(device)
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
                f"| lr={current_lr:.2e}"
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
                            "architecture": "CNNTransformerMeanPool",
                            "n_mels": 40,
                            "d_model": 64,
                            "layers": 2,
                            "heads": 4,
                        },
                        "epoch": epoch,
                        "val_macro_f1": val_f1,
                        "seed": args.seed,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping: validation macro-F1 did not improve "
                    f"for {args.patience} consecutive epochs."
                )
                break

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_acc, test_f1, y_true, y_pred = evaluate(
        model, test_loader, device, criterion
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
    )
    report_text = (
        f"Best epoch: {checkpoint['epoch']}\n"
        f"Validation macro-F1 at best epoch: "
        f"{checkpoint['val_macro_f1']:.4f}\n"
        f"Speaker-independent test accuracy: {test_acc:.4f}\n"
        f"Speaker-independent test macro-F1: {test_f1:.4f}\n\n"
        "Limitations:\n"
        "- Test metrics can vary substantially with few held-out speakers.\n"
        "- Results assume speaker_id correctly identifies unique speakers.\n"
        "- Verify dataset provenance and speaker-ID uniqueness.\n"
        "- Acted-emotion classification is not a direct measure of real-world "
        "emotion perception or hearing impairment.\n\n"
        + report
    )
    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(labels)))
    )
    with (out_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["true\\predicted"] + labels)
        for label, row in zip(labels, cm):
            writer.writerow([label] + row.tolist())

    # Export uniform pooling weights; these are identical for every frame.
    model.eval()
    pooling_rows = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            logits, weights = model(x.to(device), return_attention=True)
            weights = weights.cpu().numpy()
            for row_idx, item_idx in enumerate(
                range(batch_idx * args.batch_size,
                      min((batch_idx + 1) * args.batch_size, len(test_items)))
            ):
                item = test_items[item_idx]
                for frame_idx, weight in enumerate(weights[row_idx]):
                    pooling_rows.append(
                        [item["filename"], item["speaker_id"],
                         item["label"], frame_idx, float(weight)]
                    )

    with (out_dir / "test_pooling_weights.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "speaker_id", "true_label", "frame_index",
            "pooling_weight"
        ])
        writer.writerows(pooling_rows)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best checkpoint: {checkpoint_path.resolve()}")
    print(f"Metrics: {metrics_path.resolve()}")
    print(f"Report: {report_path.resolve()}")
    print(f"Confusion matrix: {(out_dir / 'confusion_matrix.csv').resolve()}")
    print(
        f"Uniform pooling weights: "
        f"{(out_dir / 'test_attention_weights.csv').resolve()}"
    )
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test macro-F1: {test_f1:.4f}")
    print("Reminder: verify dataset provenance and speaker IDs.")


if __name__ == "__main__":
    main()
