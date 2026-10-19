
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
from sklearn.metrics import classification_report, f1_score, accuracy_score


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
            # Feature-domain corruption. This is a stress test,
            # not a physical hearing-loss simulation.
            if np.random.rand() < 0.9:
                noise_std = np.random.uniform(0.03, 0.30)
                x += np.random.normal(
                    0, noise_std, size=x.shape
                ).astype(np.float32)

            if np.random.rand() < 0.7:
                n_mels = x.shape[1]
                width = np.random.randint(
                    2, max(3, n_mels // 4)
                )
                start = np.random.randint(
                    0, max(1, n_mels - width + 1)
                )
                attenuation = np.random.uniform(0.2, 0.8)
                x[:, start:start + width] *= attenuation

        return (
            torch.from_numpy(x),
            torch.from_numpy(clean),
            self.label_to_id[item["label"]],
        )

# Sinusoidal Positional Encoding -
# 2017 paper "Attention Is All You Need", creating the Transformer architecture.
# Provides sequential order

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=201):
        super().__init__()

        pos = torch.arange(
            max_len, dtype=torch.float32
        ).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(
            pos * div[:pe[:, 1::2].shape[1]]
        )

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


class TinyEmotionTransformer(nn.Module):
    def __init__(
        self,
        n_mels=40,
        d_model=48,
        n_heads=4,
        layers=2,
        n_classes=7,
    ):
        super().__init__()

        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=96,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=layers
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=96,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=layers
        )

        self.reconstruction = nn.Linear(d_model, n_mels)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes),
        )

    def forward(self, degraded):
        embedded = self.pos(self.input_proj(degraded))
        memory = self.encoder(embedded)

        # The degraded sequence is the decoder query;
        # encoder memory provides context for reconstruction.
        decoded = self.decoder(
            tgt=embedded,
            memory=memory,
        )

        reconstructed = self.reconstruction(decoded)
        pooled = memory.mean(dim=1)
        logits = self.classifier(pooled)

        return reconstructed, logits


def evaluate(
    model,
    loader,
    device,
    criterion_recon,
    criterion_cls,
    recon_weight=0.5,
):
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, clean, labels in loader:
            x = x.to(device)
            clean = clean.to(device)
            labels = labels.to(device)

            recon, logits = model(x)

            loss = (
                recon_weight * criterion_recon(recon, clean)
                + criterion_cls(logits, labels)
            )

            total_loss += loss.item() * x.size(0)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    average_loss = total_loss / max(1, len(loader.dataset))
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred,
        average="macro",
        zero_division=0,
    )

    return average_loss, accuracy, macro_f1, y_true, y_pred


def print_split_summary(name, items):
    speakers = {item["speaker_id"] for item in items}
    counts = {}

    for item in items:
        label = item["label"]
        counts[label] = counts.get(label, 0) + 1

    print(
        f"{name}: {len(items)} files, "
        f"{len(speakers)} speakers"
    )
    print(f"  Class counts: {counts}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--recon-weight", type=float, default=0.5)

    args = parser.parse_args()

    seed_everything(args.seed)

    # Keep CPU usage manageable on a small PC.
    torch.set_num_threads(2)
    device = torch.device("cpu")

    feature_dir = Path(args.feature_dir)
    index_path = feature_dir / "index.json"

    if not index_path.exists():
        raise SystemExit(
            f"Missing {index_path}. Run prepare_features.py first."
        )

    payload = json.loads(
        index_path.read_text(encoding="utf-8")
    )

    items = sorted(payload["items"], key=lambda item: (item["speaker_id"], item["filename"], item["label"]))
    labels = payload["labels"]
    label_to_id = {
        label: i for i, label in enumerate(labels)
    }

    if args.max_files > 0:
        items = items[:args.max_files]

    # Keep only usable features with known labels and speaker IDs.
    items = [
        item for item in items
        if item.get("label") in label_to_id
        and item.get("speaker_id") is not None
        and Path(item["feature_path"]).exists()
    ]

    if len(items) < 30:
        raise SystemExit(
            f"Only {len(items)} usable files. Prepare more data first."
        )

    # Speaker-independent split:
    # approximately 70% train, 15% validation, 15% test.
    # GroupShuffleSplit keeps each speaker within one partition.
    groups = [item["speaker_id"] for item in items]
    unique_speakers = set(groups)

    if len(unique_speakers) < 3:
        raise SystemExit(
            "At least 3 distinct speaker IDs are needed "
            "for train/validation/test splitting."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=args.seed,
    )

    train_idx, remaining_idx = next(
        splitter.split(items, groups=groups)
    )

    train_items = [items[i] for i in train_idx]
    remaining_items = [items[i] for i in remaining_idx]

    remaining_groups = [
        item["speaker_id"] for item in remaining_items
    ]

    if len(set(remaining_groups)) < 2:
        raise SystemExit(
            "Not enough remaining speakers to create "
            "separate validation and test sets."
        )

    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=args.seed + 1,
    )

    val_idx, test_idx = next(
        splitter2.split(
            remaining_items,
            groups=remaining_groups,
        )
    )

    val_items = [remaining_items[i] for i in val_idx]
    test_items = [remaining_items[i] for i in test_idx]

    # Check that speaker IDs do not overlap between partitions.
    train_speakers = {
        item["speaker_id"] for item in train_items
    }
    val_speakers = {
        item["speaker_id"] for item in val_items
    }
    test_speakers = {
        item["speaker_id"] for item in test_items
    }

    assert train_speakers.isdisjoint(val_speakers)
    assert train_speakers.isdisjoint(test_speakers)
    assert val_speakers.isdisjoint(test_speakers)

    print("\nSpeaker-independent data split")
    print_split_summary("Train", train_items)
    print_split_summary("Validation", val_items)
    print_split_summary("Test", test_items)

    # Warn if a partition has no examples of one or more classes.
    for name, partition in [
        ("Train", train_items),
        ("Validation", val_items),
        ("Test", test_items),
    ]:
        present = {item["label"] for item in partition}
        missing = sorted(set(labels) - present)

        if missing:
            print(
                f"WARNING: {name} has no examples for: "
                f"{', '.join(missing)}"
            )

    train_ds = FeatureDataset(
        train_items, label_to_id, augment=True
    )
    val_ds = FeatureDataset(
        val_items, label_to_id, augment=False
    )
    test_ds = FeatureDataset(
        test_items, label_to_id, augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    model = TinyEmotionTransformer(
        n_classes=len(labels)
    ).to(device)

    criterion_recon = nn.SmoothL1Loss()
    criterion_cls = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    # Create output directory BEFORE training and evaluation.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    metrics_path = out_dir / "metrics.csv"
    checkpoint_path = out_dir / "best_model.pt"

    with metrics_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
        ])

        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0

            for x, clean, y in train_loader:
                x = x.to(device)
                clean = clean.to(device)
                y = y.to(device)

                optimizer.zero_grad(set_to_none=True)

                recon, logits = model(x)

                loss_recon = criterion_recon(recon, clean)
                loss_cls = criterion_cls(logits, y)

                loss = (
                    args.recon_weight * loss_recon
                    + loss_cls
                )

                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )

                optimizer.step()
                total += loss.item() * x.size(0)

            train_loss = total / len(train_ds)

            val_loss, val_acc, val_f1, _, _ = evaluate(
                model,
                val_loader,
                device,
                criterion_recon,
                criterion_cls,
                args.recon_weight,
            )

            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                val_acc,
                val_f1,
            ])
            f.flush()

            print(
                f"Epoch {epoch:02d}/{args.epochs} "
                f"| train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f} "
                f"| val_acc={val_acc:.3f} "
                f"| val_macroF1={val_f1:.3f}"
            )

            # Select the checkpoint using validation macro-F1 only.
            if val_f1 > best_f1:
                best_f1 = val_f1

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "labels": labels,
                        "config": {
                            "n_mels": 40,
                            "d_model": 48,
                            "layers": 2,
                            "heads": 4,
                        },
                        "epoch": epoch,
                        "val_macro_f1": val_f1,
                        "seed": args.seed,
                    },
                    checkpoint_path,
                )

    # Load the checkpoint selected on validation speakers.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate the test speakers only after model selection.
    _, test_acc, test_f1, y_true, y_pred = evaluate(
        model,
        test_loader,
        device,
        criterion_recon,
        criterion_cls,
        args.recon_weight,
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
        "Important limitations:\n"
        "- Test metrics may be noisy when few speakers are held out.\n"
        "- The split assumes speaker_id reliably identifies speakers.\n"
        "- Verify original dataset provenance and speaker-ID uniqueness.\n"
        "- This is acted-emotion classification, not a direct measure "
        "of human emotion perception or hearing impairment.\n\n"
        + report
    )

    report_path = out_dir / "classification_report.txt"
    report_path.write_text(
        report_text, encoding="utf-8"
    )

    print(f"\nBest checkpoint: {checkpoint_path.resolve()}")
    print(f"Metrics: {metrics_path.resolve()}")
    print(f"Report: {report_path.resolve()}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test macro-F1: {test_f1:.4f}")
    print(
        "Reminder: verify dataset provenance and speaker IDs "
        "before scientific interpretation."
    )


if __name__ == "__main__":
    main()
