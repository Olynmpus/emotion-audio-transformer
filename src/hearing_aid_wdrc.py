
"""
hearing_aid_wdrc.py

Experimental waveform-level WDRC simulator.
Not a clinically calibrated hearing-aid prescription.

Input: mono or stereo WAV
Output: processed mono WAV

Example:
python src/hearing_aid_wdrc.py input.wav output.wav \
    --ratio 2.0 --attack-ms 10 --release-ms 100 --knee-dbfs -30
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def db_to_amplitude(db):
    return 10.0 ** (db / 20.0)


def amplitude_to_db(amplitude, floor=1e-10):
    return 20.0 * np.log10(np.maximum(amplitude, floor))


def smooth_power_envelope(
    signal: np.ndarray,
    sample_rate: int,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    """Estimate signal power with separate attack/release time constants."""
    if attack_ms <= 0 or release_ms <= 0:
        raise ValueError("Attack and release times must be positive.")

    attack_alpha = np.exp(-1.0 / (sample_rate * attack_ms / 1000.0))
    release_alpha = np.exp(-1.0 / (sample_rate * release_ms / 1000.0))

    power = signal.astype(np.float64) ** 2
    envelope = np.empty_like(power)
    state = 0.0

    for i, current_power in enumerate(power):
        # Attack when power rises; release when power falls.
        alpha = attack_alpha if current_power > state else release_alpha
        state = alpha * state + (1.0 - alpha) * current_power
        envelope[i] = state

    return envelope


def apply_wdrc(
    signal: np.ndarray,
    sample_rate: int,
    ratio: float = 2.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    knee_dbfs: float = -30.0,
    low_level_gain_db: float = 0.0,
    max_gain_db: float = 20.0,
) -> tuple[np.ndarray, dict]:
    """
    Apply a simple broadband compressor.

    ratio=1 means no compression above the kneepoint.
    ratio=2 means a 2:1 input-to-output level slope above the knee.
    low_level_gain_db is the gain below the kneepoint.
    """
    if ratio < 1.0:
        raise ValueError("Compression ratio must be >= 1.")
    if attack_ms <= 0 or release_ms <= 0:
        raise ValueError("Attack and release times must be positive.")

    x = np.asarray(signal, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("Expected mono audio. Convert stereo to mono first.")

    # Track short-term power and convert it to an amplitude level in dBFS.
    power_envelope = smooth_power_envelope(
        x, sample_rate, attack_ms, release_ms
    )
    rms_envelope = np.sqrt(np.maximum(power_envelope, 1e-12))
    level_dbfs = amplitude_to_db(rms_envelope)

    # Below the knee, use the configured low-level gain.
    # Above the knee, reduce the level slope according to the ratio.
    over_knee_db = np.maximum(level_dbfs - knee_dbfs, 0.0)
    gain_db = low_level_gain_db - over_knee_db * (1.0 - 1.0 / ratio)

    # Keep gain bounded to avoid unexpectedly large amplification.
    gain_db = np.clip(gain_db, -60.0, max_gain_db)

    y = x * db_to_amplitude(gain_db)

    # No automatic peak normalisation: it could confound comparisons.
    stats = {
        "sample_rate": sample_rate,
        "ratio": ratio,
        "attack_ms": attack_ms,
        "release_ms": release_ms,
        "knee_dbfs": knee_dbfs,
        "low_level_gain_db": low_level_gain_db,
        "input_rms_dbfs": float(amplitude_to_db(np.sqrt(np.mean(x**2) + 1e-12))),
        "output_rms_dbfs": float(amplitude_to_db(np.sqrt(np.mean(y**2) + 1e-12))),
        "input_peak": float(np.max(np.abs(x))) if x.size else 0.0,
        "output_peak": float(np.max(np.abs(y))) if y.size else 0.0,
        "mean_gain_db": float(np.mean(gain_db)) if gain_db.size else 0.0,
        "gain_std_db": float(np.std(gain_db)) if gain_db.size else 0.0,
        "fraction_compressed": float(np.mean(level_dbfs > knee_dbfs))
        if level_dbfs.size else 0.0,
    }

    return y.astype(np.float32), stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_wav", type=Path)
    parser.add_argument("output_wav", type=Path)
    parser.add_argument("--ratio", type=float, default=2.0)
    parser.add_argument("--attack-ms", type=float, default=10.0)
    parser.add_argument("--release-ms", type=float, default=100.0)
    parser.add_argument("--knee-dbfs", type=float, default=-30.0)
    parser.add_argument("--low-level-gain-db", type=float, default=0.0)
    parser.add_argument("--max-gain-db", type=float, default=20.0)
    args = parser.parse_args()

    audio, sample_rate = sf.read(args.input_wav, dtype="float32")

    # Match the mono input expected by the current feature pipeline.
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    processed, stats = apply_wdrc(
        audio,
        sample_rate,
        ratio=args.ratio,
        attack_ms=args.attack_ms,
        release_ms=args.release_ms,
        knee_dbfs=args.knee_dbfs,
        low_level_gain_db=args.low_level_gain_db,
        max_gain_db=args.max_gain_db,
    )

    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_wav, processed, sample_rate, subtype="PCM_16")

    print(f"Saved: {args.output_wav}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
