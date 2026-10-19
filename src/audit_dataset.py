import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from dataset_utils import discover_audio


def main():
    parser = argparse.ArgumentParser(
        description="Audit only matching RAVDESS-style WAVs in the seven emotion folders."
    )
    parser.add_argument("--data-dir", required=True, help="Root containing the seven emotion folders")
    parser.add_argument("--report", default="outputs/dataset_audit.csv")
    args = parser.parse_args()

    root = Path(args.data_dir).expanduser().resolve()
    try:
        files, rejected, mismatches = discover_audio(root)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    rows = []
    label_counts = Counter()
    sr_counts = Counter()
    errors = []
    seen_fingerprints = defaultdict(list)

    print(f"Dataset root: {root}")
    print(f"Matching filenames with folder/emotion agreement: {len(files)}")
    print(f"Rejected matching-folder WAVs (pattern or emotion mismatch): {rejected}")
    if mismatches:
        print(f"Encoded emotion / folder mismatches: {len(mismatches)}")
        for path, code, folder_label, encoded_label in mismatches[:10]:
            print(f"  {path.name}: code={code}, folder={folder_label}, code_label={encoded_label}")

    for path in files:
        label = path.parent.name.lower()
        try:
            info = sf.info(str(path))
            audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
            if sr <= 0:
                raise ValueError(f"Invalid sample rate: {sr}")
            duration = info.frames / info.samplerate

            # A deterministic, coarse fingerprint for a duplicate *flag* only.
            stride = max(1, len(audio) // 2048)
            sample = np.ascontiguousarray(audio[::stride][:2048])
            fingerprint = __import__("hashlib").sha1(sample.tobytes()).hexdigest()
            groups = path.stem.split("-")
            row = {
                "path": str(path.resolve()),
                "label": label,
                "ravdess_emotion_code": groups[2],
                "sample_rate": int(sr),
                "channels": int(audio.shape[1]),
                "duration_seconds": round(float(duration), 3),
                "frames": int(len(audio)),
                "fingerprint_sha1": fingerprint,
            }
            rows.append(row)
            label_counts[label] += 1
            sr_counts[int(sr)] += 1
            seen_fingerprints[fingerprint].append(str(path))
        except Exception as exc:
            errors.append((str(path), str(exc)))

    if not rows:
        raise SystemExit(
            "No readable WAV files matched the exact seven-group filename pattern "
            "and folder/emotion-code rule. Check filenames and folder labels."
        )

    report = Path(args.report)
    if not report.is_absolute():
        report = root / report
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    durations = np.array([r["duration_seconds"] for r in rows], dtype=float)
    duplicate_groups = [paths for paths in seen_fingerprints.values() if len(paths) > 1]

    print(f"\nAudio files indexed: {len(rows)}")
    print("\nCounts by emotion folder:")
    for label in ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]:
        print(f"  {label:12s} {label_counts[label]:6d}")
    print("\nSample rates:")
    for sr, count in sorted(sr_counts.items()):
        print(f"  {sr:6d} Hz: {count}")
    print(f"\nDuration seconds: median={np.median(durations):.2f}, "
          f"90th percentile={np.percentile(durations, 90):.2f}, max={durations.max():.2f}")
    print(f"Potential duplicate fingerprint groups: {len(duplicate_groups)} "
          "(rough flag only; inspect manually)")
    print(f"Unreadable matching files: {len(errors)}")
    if errors[:10]:
        print("First errors:")
        for path, err in errors[:10]:
            print(f"  {path}: {err}")
    print(f"\nCSV report: {report.resolve()}")
    print("Note: matching the filename format does not itself guarantee speaker-independent evaluation.")


if __name__ == "__main__":
    main()
