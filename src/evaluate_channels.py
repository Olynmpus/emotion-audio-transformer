"""
Communication-channel stress test for the Emotion project.

Runs the existing frozen CNN-Transformer emotion classifier on the SAME
speaker-held-out recordings under:
  - clean baseline
  - telephone-like band limitation (300-3400 Hz)
  - severe band limitation (500-2500 Hz)
  - Opus at 24, 16 and 12 kbps (requires FFmpeg with libopus)
  - telephone-like band limitation followed by Opus 16 kbps

Each processed condition is evaluated at:
  - native_level: preserve the level produced by the channel
  - rms_matched: rescale the processed waveform to the clean waveform RMS

Outputs:
  outputs_channel_eval/predictions.csv
  outputs_channel_eval/summary.csv
  outputs_channel_eval/classification_reports.txt

Run a small pilot first:
  .\\.venv\\Scripts\\python.exe src\\evaluate_channels.py --max-files 20

Then run all held-out test recordings:
  .\\.venv\\Scripts\\python.exe src\\evaluate_channels.py

Notes:
- This is a machine-based acted-emotion experiment, not a direct human-perception test.
- Codec conditions require ffmpeg.exe on PATH and an FFmpeg build with libopus.
- The held-out speakers default to 10,14,17,19 to match the current project split.
"""

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_features import load_mono, waveform_to_logmel, SAMPLE_RATE, HOP_LENGTH
from prepare_prosodic_features import prosodic_features
from train_prosodic import CNNTransformerAttention


DEFAULT_FEATURE_DIR = ROOT / "features_prosodic_4s"
DEFAULT_CHECKPOINT = ROOT / "outputs_prosodic_4s" / "best_model.pt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs_channel_eval"


def resolve_index_path(value):
    """Resolve Windows-index paths to files mounted in WSL."""
    raw = str(value or "")
    slash = chr(92)

    if len(raw) >= 3 and raw[1] == ":" and raw[2] in (slash, "/"):
        drive = raw[0].lower()
        tail = raw[3:].replace(slash, "/")
        candidate = Path(f"/mnt/{drive}/{tail}")
        if candidate.exists():
            return candidate

    return Path(raw).expanduser()


def rms(audio):
    audio = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(audio * audio) + 1e-12))


def db(value):
    return float(20.0 * np.log10(max(float(value), 1e-12)))


def match_rms(audio, target_rms):
    current = rms(audio)
    if current < 1e-8:
        return audio.copy()
    scaled = audio * (float(target_rms) / current)
    # Avoid creating out-of-range samples; this should rarely trigger.
    peak = float(np.max(np.abs(scaled))) if len(scaled) else 0.0
    if peak > 0.999:
        scaled = scaled * (0.999 / peak)
    return scaled.astype(np.float32)


def bandlimit(audio, low_hz, high_hz, sample_rate=SAMPLE_RATE):
    nyquist = sample_rate / 2.0
    high_hz = min(float(high_hz), nyquist * 0.98)
    low_hz = max(float(low_hz), 1.0)
    sos = butter(
        4,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    return sosfiltfilt(sos, np.asarray(audio, dtype=np.float32)).astype(np.float32)


def fit_length(audio, target_length):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) > target_length:
        return audio[:target_length]
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio))).astype(np.float32)
    return audio


def opus_roundtrip(audio, bitrate_kbps, ffmpeg_path):
    """Encode and decode one utterance through FFmpeg/libopus."""
    with tempfile.TemporaryDirectory(prefix="emotion_codec_") as temp_dir:
        temp_dir = Path(temp_dir)
        input_wav = temp_dir / "input.wav"
        encoded = temp_dir / "encoded.ogg"
        decoded_wav = temp_dir / "decoded.wav"

        sf.write(input_wav, np.asarray(audio, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")

        encode_cmd = [
            ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(input_wav),
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k",
            "-vbr", "off", "-application", "voip",
            str(encoded),
        ]
        subprocess.run(encode_cmd, check=True, capture_output=True)

        decode_cmd = [
            ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(encoded),
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(decoded_wav),
        ]
        subprocess.run(decode_cmd, check=True, capture_output=True)

        decoded, sr = sf.read(decoded_wav, dtype="float32", always_2d=False)
        if decoded.ndim > 1:
            decoded = decoded.mean(axis=1)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"Decoded sample rate {sr}, expected {SAMPLE_RATE}")
        return fit_length(decoded, len(audio))


def acoustic_metrics(audio):
    """Descriptive signal measures; these are not direct measures of emotion."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) == 0:
        return {
            "rms_dbfs": np.nan, "envelope_sd_db": np.nan,
            "crest_factor_db": np.nan, "spectral_centroid_hz": np.nan,
            "energy_above_3400_ratio": np.nan, "spectral_flatness": np.nan,
        }

    r = rms(audio)
    rms_dbfs = db(r)
    peak = float(np.max(np.abs(audio)))
    crest = db(peak / max(r, 1e-12))

    frame_len = max(1, int(0.020 * SAMPLE_RATE))
    n_frames = max(1, int(np.ceil(len(audio) / frame_len)))
    padded = np.pad(audio, (0, n_frames * frame_len - len(audio)))
    frames = padded.reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    envelope_sd = float(np.std(20.0 * np.log10(np.maximum(frame_rms, 1e-8))))

    # Mean spectrum from overlapping 25-ms Hann frames.
    n_fft = 512
    hop = 160
    if len(audio) < n_fft:
        audio_fft = np.pad(audio, (0, n_fft - len(audio)))
    else:
        audio_fft = audio
    starts = list(range(0, max(1, len(audio_fft) - n_fft + 1), hop))
    if not starts:
        starts = [0]
    window = np.hanning(n_fft).astype(np.float32)
    spectra = []
    for start in starts:
        frame = audio_fft[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spectra.append(np.abs(np.fft.rfft(frame * window)) ** 2)
    power = np.mean(np.asarray(spectra), axis=0) + 1e-12
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SAMPLE_RATE)
    magnitude = np.sqrt(power)
    centroid = float(np.sum(freqs * magnitude) / max(np.sum(magnitude), 1e-12))
    total_energy = float(np.sum(power))
    high_energy = float(np.sum(power[freqs >= 3400.0]))
    high_ratio = high_energy / max(total_energy, 1e-12)
    flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))

    return {
        "rms_dbfs": rms_dbfs,
        "envelope_sd_db": envelope_sd,
        "crest_factor_db": crest,
        "spectral_centroid_hz": centroid,
        "energy_above_3400_ratio": high_ratio,
        "spectral_flatness": flatness,
    }


def audio_to_model_features(audio):
    mel = waveform_to_logmel(audio).astype(np.float32)
    prosody = prosodic_features(audio, sr=SAMPLE_RATE).astype(np.float32)
    combined = np.concatenate([mel, prosody], axis=1).astype(np.float32)
    if combined.shape != (201, 44):
        raise ValueError(f"Expected feature shape (201,44), got {combined.shape}")
    return np.clip(combined, -5.0, 5.0).astype(np.float32)


def run_model(model, feature, device):
    x = torch.from_numpy(np.asarray(feature, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_id = int(np.argmax(probs))
    return pred_id, float(np.max(probs)), probs


def condition_waveforms(clean_audio, ffmpeg_path):
    """Return condition_name -> processed waveform, with one encoding per codec."""
    conditions = {
        "bandpass_300_3400": bandlimit(clean_audio, 300, 3400),
        "bandpass_500_2500": bandlimit(clean_audio, 500, 2500),
    }

    if ffmpeg_path:
        conditions["opus_24kbps"] = opus_roundtrip(clean_audio, 24, ffmpeg_path)
        conditions["opus_16kbps"] = opus_roundtrip(clean_audio, 16, ffmpeg_path)
        conditions["opus_12kbps"] = opus_roundtrip(clean_audio, 12, ffmpeg_path)
        telephone = conditions["bandpass_300_3400"]
        conditions["telephone_plus_opus_16kbps"] = opus_roundtrip(
            telephone, 16, ffmpeg_path
        )
    return conditions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default=str(DEFAULT_FEATURE_DIR))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--test-speakers", default="10,14,17,19")
    parser.add_argument("--max-files", type=int, default=0,
                        help="For a pilot, use 20. Zero means all selected test files.")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--skip-codecs", action="store_true",
                        help="Run filter conditions only; no FFmpeg required.")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    feature_dir = Path(args.feature_dir).expanduser().resolve()
    index_path = feature_dir / "index.json"
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not index_path.exists():
        raise SystemExit(f"Missing feature index: {index_path}")
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing model checkpoint: {checkpoint_path}")

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    labels = payload["labels"]
    label_to_id = {label: i for i, label in enumerate(labels)}
    test_speakers = {x.strip() for x in args.test_speakers.split(",") if x.strip()}

    items = [
        item for item in payload["items"]
        if str(item.get("speaker_id")) in test_speakers
        and item.get("label") in label_to_id
        and resolve_index_path(item.get("feature_path", "")).exists()
        and resolve_index_path(item.get("source_path", "")).exists()
    ]
    items.sort(key=lambda x: (str(x.get("speaker_id")), x.get("filename", "")))

    if args.max_files > 0:
        items = items[:args.max_files]
    if not items:
        raise SystemExit(
            f"No test items found for speaker IDs {sorted(test_speakers)}. "
            "Check --test-speakers against features_prosodic_4s/index.json."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_labels = checkpoint.get("labels", labels)
    if list(model_labels) != list(labels):
        raise SystemExit(
            f"Label order mismatch. Index labels={labels}; checkpoint labels={model_labels}"
        )

    model = CNNTransformerAttention(
        n_mels=44, d_model=64, n_heads=4, layers=2,
        n_classes=len(labels), max_len=201,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ffmpeg_path = None if args.skip_codecs else shutil.which("ffmpeg")
    if not args.skip_codecs and not ffmpeg_path:
        print("WARNING: ffmpeg.exe was not found on PATH. Codec conditions will be skipped.")
        print("Install FFmpeg with libopus and add ffmpeg.exe to PATH to test Opus conditions.")

    print(f"Project root: {ROOT}")
    print(f"Feature index: {index_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Selected held-out recordings: {len(items)}")
    print(f"Speaker IDs: {sorted(test_speakers)}")
    print(f"FFmpeg: {ffmpeg_path or 'not available; codecs skipped'}")

    prediction_rows = []
    baseline_by_key = {}
    clean_audio_by_key = {}
    clean_metrics_by_key = {}

    # First pass: baseline predictions from the exact precomputed model inputs.
    model.eval()
    for i, item in enumerate(items, 1):
        key = item["filename"]
        with np.load(resolve_index_path(item["feature_path"])) as npz:
            clean_feature = np.clip(npz["feature"].astype(np.float32), -5.0, 5.0)
        pred_id, confidence, probs = run_model(model, clean_feature, device)
        true_label = item["label"]
        baseline = {
            "predicted_label": labels[pred_id],
            "confidence": confidence,
            "correct": int(labels[pred_id] == true_label),
            "probs": probs,
        }
        baseline_by_key[key] = baseline

        clean_audio = load_mono(resolve_index_path(item["source_path"]))
        clean_audio_by_key[key] = clean_audio
        clean_metrics_by_key[key] = acoustic_metrics(clean_audio)

        prediction_rows.append({
            "filename": item["filename"],
            "speaker_id": item["speaker_id"],
            "true_label": true_label,
            "condition": "clean_baseline",
            "level_mode": "native_level",
            "predicted_label": labels[pred_id],
            "confidence": confidence,
            "correct": int(labels[pred_id] == true_label),
            "baseline_predicted_label": labels[pred_id],
            "baseline_correct": int(labels[pred_id] == true_label),
            "prediction_changed": 0,
            "transition": "baseline",
            **clean_metrics_by_key[key],
            "delta_rms_db": 0.0,
            "delta_envelope_sd_db": 0.0,
            "delta_crest_factor_db": 0.0,
            "delta_spectral_centroid_hz": 0.0,
            "delta_energy_above_3400_ratio": 0.0,
            "delta_spectral_flatness": 0.0,
        })
        if i % 50 == 0 or i == len(items):
            print(f"Baseline inference: {i}/{len(items)}")

    # Process each utterance through each channel; do not retrain the model.
    for i, item in enumerate(items, 1):
        key = item["filename"]
        clean_audio = clean_audio_by_key[key]
        clean_rms = rms(clean_audio)
        clean_metrics = clean_metrics_by_key[key]
        true_label = item["label"]
        baseline = baseline_by_key[key]

        try:
            transformed_conditions = condition_waveforms(clean_audio, ffmpeg_path)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                "FFmpeg codec operation failed. Confirm FFmpeg includes libopus. "
                f"Error: {exc.stderr.decode(errors='replace') if exc.stderr else exc}"
            )

        for condition, transformed in transformed_conditions.items():
            for level_mode in ("native_level", "rms_matched"):
                audio = transformed if level_mode == "native_level" else match_rms(transformed, clean_rms)
                feature = audio_to_model_features(audio)
                pred_id, confidence, probs = run_model(model, feature, device)
                pred_label = labels[pred_id]
                correct = int(pred_label == true_label)

                if baseline["correct"] and not correct:
                    transition = "correct_to_wrong"
                elif not baseline["correct"] and correct:
                    transition = "wrong_to_correct"
                elif baseline["correct"] and correct:
                    transition = "both_correct"
                else:
                    transition = "both_wrong"

                metrics = acoustic_metrics(audio)
                prediction_rows.append({
                    "filename": item["filename"],
                    "speaker_id": item["speaker_id"],
                    "true_label": true_label,
                    "condition": condition,
                    "level_mode": level_mode,
                    "predicted_label": pred_label,
                    "confidence": confidence,
                    "correct": correct,
                    "baseline_predicted_label": baseline["predicted_label"],
                    "baseline_correct": baseline["correct"],
                    "prediction_changed": int(pred_label != baseline["predicted_label"]),
                    "transition": transition,
                    **metrics,
                    "delta_rms_db": metrics["rms_dbfs"] - clean_metrics["rms_dbfs"],
                    "delta_envelope_sd_db": metrics["envelope_sd_db"] - clean_metrics["envelope_sd_db"],
                    "delta_crest_factor_db": metrics["crest_factor_db"] - clean_metrics["crest_factor_db"],
                    "delta_spectral_centroid_hz": metrics["spectral_centroid_hz"] - clean_metrics["spectral_centroid_hz"],
                    "delta_energy_above_3400_ratio": metrics["energy_above_3400_ratio"] - clean_metrics["energy_above_3400_ratio"],
                    "delta_spectral_flatness": metrics["spectral_flatness"] - clean_metrics["spectral_flatness"],
                })

        if i % 10 == 0 or i == len(items):
            print(f"Channel processing: {i}/{len(items)}")

    # Summary table: all conditions compared on the same utterances.
    group_keys = []
    for row in prediction_rows:
        k = (row["condition"], row["level_mode"])
        if k not in group_keys:
            group_keys.append(k)

    summary_rows = []
    report_lines = []
    for condition, level_mode in group_keys:
        rows = [r for r in prediction_rows if r["condition"] == condition and r["level_mode"] == level_mode]
        y_true = [r["true_label"] for r in rows]
        y_pred = [r["predicted_label"] for r in rows]
        baseline_pred = [r["baseline_predicted_label"] for r in rows]
        transitions = {name: sum(r["transition"] == name for r in rows) for name in
                       ("correct_to_wrong", "wrong_to_correct", "both_correct", "both_wrong")}
        summary_rows.append({
            "condition": condition,
            "level_mode": level_mode,
            "n": len(rows),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "mean_confidence": float(np.mean([r["confidence"] for r in rows])),
            "prediction_change_rate_vs_clean": float(np.mean([r["prediction_changed"] for r in rows])),
            **transitions,
            "mean_delta_rms_db": float(np.mean([r["delta_rms_db"] for r in rows])),
            "mean_delta_envelope_sd_db": float(np.mean([r["delta_envelope_sd_db"] for r in rows])),
            "mean_delta_crest_factor_db": float(np.mean([r["delta_crest_factor_db"] for r in rows])),
            "mean_delta_spectral_centroid_hz": float(np.mean([r["delta_spectral_centroid_hz"] for r in rows])),
            "mean_delta_energy_above_3400_ratio": float(np.mean([r["delta_energy_above_3400_ratio"] for r in rows])),
            "mean_delta_spectral_flatness": float(np.mean([r["delta_spectral_flatness"] for r in rows])),
        })

        report_lines.append(f"\n{'=' * 78}\n{condition} | {level_mode}\n")
        report_lines.append(
            f"Accuracy: {accuracy_score(y_true, y_pred):.4f}; "
            f"Macro-F1: {f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.4f}\n"
        )
        report_lines.append(
            f"Prediction changed vs clean: {sum(r['prediction_changed'] for r in rows)}/{len(rows)}\n"
        )
        report_lines.append(
            "Transitions vs clean: " + ", ".join(f"{k}={v}" for k, v in transitions.items()) + "\n"
        )
        report_lines.append(
            classification_report(y_true, y_pred, labels=labels, target_names=labels, zero_division=0)
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report_lines.append("\nConfusion matrix (rows=true, columns=predicted):\n")
        report_lines.append(",".join(["true\\predicted"] + list(labels) + ["\n"]))
        for label, cm_row in zip(labels, cm):
            report_lines.append(",".join([label] + [str(int(v)) for v in cm_row] + ["\n"]))

    prediction_path = output_dir / "predictions.csv"
    summary_path = output_dir / "summary.csv"
    report_path = output_dir / "classification_reports.txt"

    if prediction_rows:
        with prediction_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()))
            writer.writeheader()
            writer.writerows(prediction_rows)

    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    report_path.write_text("".join(report_lines), encoding="utf-8")

    print("\nCompleted.")
    print(f"Predictions: {prediction_path}")
    print(f"Summary: {summary_path}")
    print(f"Reports: {report_path}")
    print("\nMain metrics:")
    for row in summary_rows:
        print(
            f"{row['condition']:32s} {row['level_mode']:12s} "
            f"acc={row['accuracy']:.3f} macro-F1={row['macro_f1']:.3f} "
            f"changed={row['prediction_change_rate_vs_clean']:.3f}"
        )


if __name__ == "__main__":
    main()
