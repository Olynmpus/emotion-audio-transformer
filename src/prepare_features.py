import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from audio_features import extract_feature
from dataset_utils import LABELS, discover_audio


def balanced_limit(files, limit):
    """Select up to limit files round-robin across labels for a useful pilot."""
    if limit <= 0 or len(files) <= limit:
        return files
    by_label = defaultdict(deque)
    for path in files:
        by_label[path.parent.name.lower()].append(path)
    selected = []
    labels = [label for label in LABELS if by_label[label]]
    while len(selected) < limit and labels:
        remaining_labels = []
        for label in labels:
            if by_label[label] and len(selected) < limit:
                selected.append(by_label[label].popleft())
            if by_label[label]:
                remaining_labels.append(label)
        labels = remaining_labels
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Extract features only from valid RAVDESS-style WAVs in the seven emotion folders."
    )
    parser.add_argument("--data-dir", required=True, help="Root containing the seven emotion folders")
    parser.add_argument("--feature-dir", required=True, help="Output folder for feature files")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional total file limit for a pilot; selection is round-robin across labels")
    args = parser.parse_args()

    root = Path(args.data_dir).expanduser().resolve()
    out_root = Path(args.feature_dir).expanduser()
    if not out_root.is_absolute():
        out_root = (root / out_root).resolve()

    try:
        files, rejected, mismatches = discover_audio(root)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    files = balanced_limit(files, args.limit)
    if not files:
        raise SystemExit(
            "No WAV files matched the exact filename pattern and folder/emotion-code rule."
        )

    print(f"Dataset root: {root}")
    print(f"Selected valid files: {len(files)}")
    print(f"Rejected WAV files (pattern or emotion mismatch): {rejected}")
    if mismatches:
        print(f"Emotion-code/folder mismatches: {len(mismatches)} (excluded)")

    index = []
    errors = 0
    for i, path in enumerate(files, 1):
        label = path.parent.name.lower()
        try:
            feat = extract_feature(path)
            # Store individual compressed arrays to keep RAM use low.
            rel = path.relative_to(root)
            dest = (out_root / rel).with_suffix(".npz")
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, feature=feat.astype(np.float16))
            index.append({
                "feature_path": str(dest.resolve()),
                "label": label,
                "source_path": str(path.resolve()),
                "filename": path.name,
                "speaker_id": path.stem.split("-")[-1],
                "source_format": "ravdess_style",
            })
        except Exception as exc:
            errors += 1
            print(f"[WARN] {path}: {exc}")
        if i % 250 == 0 or i == len(files):
            print(f"Processed {i}/{len(files)}; saved={len(index)}; errors={errors}")

    out_root.mkdir(parents=True, exist_ok=True)
    index_path = out_root / "index.json"
    index_path.write_text(json.dumps({"labels": LABELS, "items": index}, indent=2), encoding="utf-8")
    print(f"\nSaved index: {index_path.resolve()}")
    print(f"Features saved: {len(index)}; errors: {errors}")
    print("Only matching filenames with emotion codes consistent with their folder were processed.")


if __name__ == "__main__":
    main()
