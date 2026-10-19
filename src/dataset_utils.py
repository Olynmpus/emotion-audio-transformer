"""Shared dataset discovery for the seven-class RAVDESS-style subset."""
from pathlib import Path
import re

EMOTION_FOLDERS = {
    "Angry": "angry",
    "Disgusted": "disgusted",
    "Fearful": "fearful",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
    "Surprised": "surprised",
}

# Exact filename: 2 digits, then six more two-digit groups, .wav extension.
FILENAME_PATTERN = re.compile(r"^\d{2}(?:-\d{2}){6}\.wav$", re.IGNORECASE)

# RAVDESS filename group 3: emotion code. Calm (02) is intentionally excluded.
EMOTION_CODE_TO_LABEL = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgusted",
    "08": "surprised",
}

LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]


def discover_audio(data_dir):
    """Return (accepted_paths, rejected_count, mismatch_paths).

    Only searches WAV files immediately inside the seven emotion folders. It
    never recursively searches the project root, .venv, features, or outputs.
    A file is accepted only when its name matches the exact pattern and its
    encoded RAVDESS emotion agrees with the parent folder.
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    accepted = []
    rejected = 0
    mismatches = []

    for folder, expected_label in EMOTION_FOLDERS.items():
        emotion_dir = root / folder
        if not emotion_dir.is_dir():
            continue

        # Direct children only; do not recurse into any nested directories.
        for path in sorted(emotion_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".wav":
                continue
            if FILENAME_PATTERN.fullmatch(path.name) is None:
                rejected += 1
                continue

            groups = path.stem.split("-")
            encoded_label = EMOTION_CODE_TO_LABEL.get(groups[2])
            if encoded_label != expected_label:
                rejected += 1
                mismatches.append((path, groups[2], expected_label, encoded_label))
                continue
            accepted.append(path)

    return accepted, rejected, mismatches
