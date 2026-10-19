"""
Prepare Experiment C features by appending four frame-aligned prosodic features
to the existing 40-bin log-Mel arrays. Existing features are not modified.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from audio_features import load_mono, SAMPLE_RATE, N_FFT, HOP_LENGTH, MAX_FRAMES


def prosodic_features(audio, sr=SAMPLE_RATE):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    pad = N_FFT // 2
    padded = np.pad(audio, (pad, pad))
    n_frames = 1 + max(0, (len(padded) - N_FFT) // HOP_LENGTH)
    frames = np.stack([
        padded[i * HOP_LENGTH:i * HOP_LENGTH + N_FFT]
        for i in range(n_frames)
    ]).astype(np.float32)

    # Fixed dBFS scaling; no validation/test-set fitting is performed.
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    energy_db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    energy_scaled = np.clip((energy_db + 50.0) / 40.0, -1.0, 1.0)

    # FFT autocorrelation estimates F0 efficiently for every frame.
    window = np.hanning(N_FFT).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], n=1024, axis=1)
    autocorr = np.fft.irfft(np.abs(spectrum) ** 2, n=1024, axis=1)[:, :N_FFT]
    autocorr_norm = autocorr / np.maximum(autocorr[:, :1], 1e-8)

    min_lag = max(1, int(sr / 400.0))
    max_lag = min(N_FFT - 2, int(sr / 60.0))
    candidate = autocorr_norm[:, min_lag:max_lag + 1]
    best_lag = np.argmax(candidate, axis=1) + min_lag
    confidence = np.max(candidate, axis=1)

    voiced = (confidence >= 0.30) & (energy_db > -50.0)
    f0 = np.zeros(len(frames), dtype=np.float32)
    f0[voiced] = sr / best_lag[voiced].astype(np.float32)

    # Scaled log-F0 in roughly [-1, 1]; zero denotes unvoiced.
    log_f0 = np.zeros_like(f0)
    log_f0[voiced] = np.log(np.clip(f0[voiced], 60.0, 400.0))
    lo, hi = np.log(60.0), np.log(400.0)
    pitch_scaled = np.zeros_like(f0)
    pitch_scaled[voiced] = 2.0 * (log_f0[voiced] - lo) / (hi - lo) - 1.0

    confidence_scaled = np.clip(confidence, 0.0, 1.0).astype(np.float32)
    confidence_scaled[~voiced] = 0.0

    pitch_delta = np.zeros_like(f0)
    adjacent_voiced = voiced[1:] & voiced[:-1]
    raw_delta = np.diff(log_f0)
    pitch_delta[1:][adjacent_voiced] = np.clip(
        raw_delta[adjacent_voiced] / 0.15, -1.0, 1.0
    )

    prosody = np.stack([
        pitch_scaled, energy_scaled.astype(np.float32),
        confidence_scaled, pitch_delta
    ], axis=1).astype(np.float32)

    if prosody.shape[0] > MAX_FRAMES:
        prosody = prosody[:MAX_FRAMES]
    if prosody.shape[0] < MAX_FRAMES:
        prosody = np.pad(prosody, ((0, MAX_FRAMES - prosody.shape[0]), (0, 0)))
    return prosody


def main():
    parser = argparse.ArgumentParser(
        description="Append four frame-aligned prosodic features to log-Mel features."
    )
    parser.add_argument("--base-feature-dir", required=True)
    parser.add_argument("--output-feature-dir", required=True)
    args = parser.parse_args()

    base_dir = Path(args.base_feature_dir).expanduser().resolve()
    out_dir = Path(args.output_feature_dir).expanduser().resolve()
    index_path = base_dir / "index.json"
    if not index_path.exists():
        raise SystemExit(f"Missing index: {index_path}")

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    out_items, errors = [], 0
    print(f"Base feature directory: {base_dir}")
    print(f"Output feature directory: {out_dir}")
    print(f"Items to process: {len(payload['items'])}")

    for i, item in enumerate(payload["items"], 1):
        try:
            with np.load(item["feature_path"]) as data:
                mel = data["feature"].astype(np.float32)
            if mel.shape != (MAX_FRAMES, 40):
                raise ValueError(f"Expected {(MAX_FRAMES, 40)}, got {mel.shape}")

            audio = load_mono(Path(item["source_path"]))
            prosody = prosodic_features(audio)
            combined = np.concatenate([mel, prosody], axis=1).astype(np.float32)
            if combined.shape != (MAX_FRAMES, 44) or not np.isfinite(combined).all():
                raise ValueError(f"Invalid combined feature array: {combined.shape}")

            base_path = Path(item["feature_path"]).resolve()
            try:
                rel = base_path.relative_to(base_dir)
            except ValueError:
                rel = Path(item["label"]) / (Path(item["filename"]).stem + ".npz")
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, feature=combined.astype(np.float16))

            new_item = dict(item)
            new_item["feature_path"] = str(dest.resolve())
            new_item["feature_type"] = "logmel40_plus_prosody4"
            out_items.append(new_item)
        except Exception as exc:
            errors += 1
            print(f"[WARN] {item.get('filename', '?')}: {exc}")

        if i % 250 == 0 or i == len(payload["items"]):
            print(f"Processed {i}/{len(payload['items'])}; saved={len(out_items)}; errors={errors}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_index = out_dir / "index.json"
    out_index.write_text(
        json.dumps({"labels": payload["labels"], "items": out_items}, indent=2),
        encoding="utf-8"
    )
    print(f"\nSaved index: {out_index}")
    print(f"Features saved: {len(out_items)}; errors: {errors}")
    if errors:
        print("WARNING: inspect warnings before training.")


if __name__ == "__main__":
    main()
