
from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 320
N_MELS = 40
MAX_SECONDS = 4.0
MAX_FRAMES = 201

# Silence-trimming settings
TRIM_SILENCE = True
COMPRESS_LONG_PAUSES = False
FRAME_MS = 20
SILENCE_THRESHOLD_DB = -40.0
RELATIVE_THRESHOLD_DB = 25.0
PADDING_MS = 100


# ============================================================
# Mel filterbank
# ============================================================

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (
        10.0 ** (np.asarray(mel) / 2595.0) - 1.0
    )


def mel_filterbank(
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=N_MELS,
    fmin=60.0,
    fmax=7600.0,
):
    mel_points = np.linspace(
        hz_to_mel(fmin),
        hz_to_mel(fmax),
        n_mels + 2,
    )

    hz_points = mel_to_hz(mel_points)

    bins = np.floor(
        (n_fft + 1) * hz_points / sr
    ).astype(int)

    bank = np.zeros(
        (n_mels, n_fft // 2 + 1),
        dtype=np.float32,
    )

    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        center = max(center, left + 1)
        right = max(right, center + 1)

        for k in range(left, min(center, bank.shape[1])):
            bank[m - 1, k] = (
                (k - left) / max(1, center - left)
            )

        for k in range(center, min(right, bank.shape[1])):
            bank[m - 1, k] = (
                (right - k) / max(1, right - center)
            )

    return bank


_MEL_BANK = mel_filterbank()
_HANN = np.hanning(N_FFT).astype(np.float32)


# ============================================================
# Silence trimming
# ============================================================


def trim_silence_edges(
    audio,
    sample_rate,
    frame_ms=20,
    threshold_db=-60.0,
    relative_threshold_db=30.0,
    padding_ms=100,
    min_duration_s=0.50,
):
    """
    Conservatively trim leading/trailing silence.

    The activity threshold is the greater of:
      - threshold_db (the absolute noise floor)
      - peak frame RMS minus relative_threshold_db

    Internal pauses are preserved. If the detected region is
    suspiciously short, preserve the original recording rather
    than risk deleting most of the utterance.
    """

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    metadata = {
        "original_samples": len(audio),
        "trimmed_start_ms": 0.0,
        "trimmed_end_ms": 0.0,
        "effective_threshold_db": None,
        "status": "unchanged",
    }

    if len(audio) == 0:
        metadata["status"] = "empty_audio"
        return audio, metadata

    frame_length = max(
        1,
        int(sample_rate * frame_ms / 1000),
    )

    n_frames = int(np.ceil(len(audio) / frame_length))

    padded = np.pad(
        audio,
        (0, n_frames * frame_length - len(audio)),
    )

    frames = padded.reshape(n_frames, frame_length)

    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10))

    peak_frame_db = float(np.max(rms_db))

    effective_threshold = max(
        threshold_db,
        peak_frame_db - relative_threshold_db,
    )

    metadata["effective_threshold_db"] = effective_threshold

    active = rms_db >= effective_threshold

    if not np.any(active):
        metadata["status"] = "no_activity_detected_preserved"
        return audio, metadata

    first_active = int(np.argmax(active))
    last_active = int(len(active) - 1 - np.argmax(active[::-1]))

    padding_samples = int(
        sample_rate * padding_ms / 1000
    )

    start_sample = max(
        0,
        first_active * frame_length - padding_samples,
    )

    end_sample = min(
        len(audio),
        (last_active + 1) * frame_length + padding_samples,
    )

    if end_sample <= start_sample:
        metadata["status"] = "invalid_region_preserved"
        return audio, metadata

    proposed = audio[start_sample:end_sample]

    # Safety check: do not discard most of an utterance
    # because of a brief high-energy section.
    if len(proposed) < min_duration_s * sample_rate:
        metadata["status"] = "short_detection_preserved_original"
        return audio, metadata

    metadata["trimmed_start_ms"] = (
        1000.0 * start_sample / sample_rate
    )

    metadata["trimmed_end_ms"] = (
        1000.0 * (len(audio) - end_sample) / sample_rate
    )

    metadata["status"] = (
        "trimmed"
        if start_sample > 0 or end_sample < len(audio)
        else "no_edge_silence_removed"
    )

    return proposed.astype(np.float32), metadata


def compress_long_pauses(
    audio,
    sample_rate,
    frame_ms=20,
    threshold_db=-45.0,
    min_pause_s=1.0,
    keep_pause_s=0.30,
):
    """
    Shorten long internal low-energy pauses while preserving
    leading and trailing audio and the speech around each pause.

    This is an experimental preprocessing method for emotion
    recognition. Inspect the output before using it for training.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if len(audio) == 0:
        return audio

    frame_length = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = int(np.ceil(len(audio) / frame_length))

    padded = np.pad(
        audio,
        (0, n_frames * frame_length - len(audio)),
    )
    frames = padded.reshape(n_frames, frame_length)

    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    silent = rms_db < threshold_db

    min_frames = max(1, int(min_pause_s * sample_rate / frame_length))
    keep_frames = max(1, int(keep_pause_s * sample_rate / frame_length))

    # Find continuous low-energy regions.
    changes = np.diff(np.r_[False, silent, False].astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    # Only compress long pauses that are internal to the recording.
    regions = [
        (start, end)
        for start, end in zip(starts, ends)
        if start > 0
        and end < n_frames
        and end - start >= min_frames
    ]

    # Build output from the original samples, working in time order.
    pieces = []
    cursor = 0

    for start, end in regions:
        start_sample = start * frame_length
        end_sample = min(len(audio), end * frame_length)

        if start_sample > cursor:
            pieces.append(audio[cursor:start_sample])

        pause = audio[start_sample:end_sample]
        target_samples = min(
            len(pause),
            keep_frames * frame_length,
        )

        # Retain a central slice of the quiet region.
        offset = max(0, (len(pause) - target_samples) // 2)
        pieces.append(pause[offset:offset + target_samples])

        cursor = end_sample

    if cursor < len(audio):
        pieces.append(audio[cursor:])

    return np.concatenate(pieces).astype(np.float32)

# ============================================================
# Audio loading and resampling
# ============================================================

def load_mono(path, return_metadata=False):
    """
    Load audio, convert to mono, trim edge silence, resample
    to 16 kHz and enforce the existing three-second limit.
    """

    audio, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    # Convert multichannel audio to mono.
    audio = audio.mean(axis=1).astype(np.float32)

    original_sample_rate = int(sr)
    original_duration = len(audio) / sr if sr else 0.0

    # Trim at the original sample rate.
    if TRIM_SILENCE:
        audio, trim_metadata = trim_silence_edges(
            audio,
            sr,
        )
    else:
        trim_metadata = {
            "trimmed_start_ms": 0.0,
            "trimmed_end_ms": 0.0,
            "status": "trimming_disabled",
        }

        # Optionally compress long internal pauses.
    if COMPRESS_LONG_PAUSES:
        audio = compress_long_pauses(
            audio,
            sr,
            threshold_db=-45.0,
            min_pause_s=1.0,
            keep_pause_s=0.30,
        )

    duration_after_trim = len(audio) / sr if sr else 0.0

    # Resample to the model sample rate.
    if sr != SAMPLE_RATE:
        g = gcd(int(sr), SAMPLE_RATE)

        audio = resample_poly(
            audio,
            SAMPLE_RATE // g,
            sr // g,
        ).astype(np.float32)

        sr = SAMPLE_RATE

    # Keep the existing duration policy.
    max_samples = int(MAX_SECONDS * SAMPLE_RATE)

    was_truncated = len(audio) > max_samples

    if was_truncated:
        audio = audio[:max_samples]

    # Prevent downstream FFT errors for extremely short audio.
    if len(audio) < N_FFT:
        audio = np.pad(
            audio,
            (0, N_FFT - len(audio)),
        )

    audio = audio.astype(np.float32)

    metadata = {
        "path": str(path),
        "original_sample_rate": original_sample_rate,
        "original_duration_s": original_duration,
        "duration_after_trim_s": duration_after_trim,
        "duration_after_limit_s": len(audio) / SAMPLE_RATE,
        "trimmed_start_ms": trim_metadata["trimmed_start_ms"],
        "trimmed_end_ms": trim_metadata["trimmed_end_ms"],
        "effective_threshold_db": trim_metadata.get(
            "effective_threshold_db"
        ),
        "trim_status": trim_metadata["status"],
        "truncated_to_max_seconds": was_truncated,
    }

    if return_metadata:
        return audio, metadata

    return audio


# ============================================================
# Log-Mel feature extraction
# ============================================================

def waveform_to_logmel(audio):
    """
    Convert waveform to a [MAX_FRAMES, N_MELS] log-Mel array.
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    ).reshape(-1)

    # Centered framing.
    pad = N_FFT // 2

    padded = np.pad(
        audio,
        (pad, pad),
    )

    n_frames = 1 + max(
        0,
        (len(padded) - N_FFT) // HOP_LENGTH,
    )

    frames = np.stack([
        padded[
            i * HOP_LENGTH:
            i * HOP_LENGTH + N_FFT
        ] * _HANN
        for i in range(n_frames)
    ])

    # Short-time Fourier transform.
    spectrum = np.fft.rfft(
        frames,
        n=N_FFT,
        axis=1,
    )

    power = (
        np.abs(spectrum) ** 2
    ).astype(np.float32)

    # Map FFT power to Mel bands.
    mel_power = power @ _MEL_BANK.T

    logmel = np.log(
        np.maximum(mel_power, 1e-8)
    ).astype(np.float32)

    # Preserve the existing maximum frame policy.
    if logmel.shape[0] > MAX_FRAMES:
        logmel = logmel[:MAX_FRAMES]

    # Pad to a fixed input size.
    if logmel.shape[0] < MAX_FRAMES:
        logmel = np.pad(
            logmel,
            (
                (0, MAX_FRAMES - logmel.shape[0]),
                (0, 0),
            ),
        )

    # Per-utterance normalization.
    # Note: this removes absolute loudness information.
    mean = float(logmel.mean())
    std = float(logmel.std())

    logmel = (
        (logmel - mean) / max(std, 1e-4)
    )

    return np.clip(
        logmel,
        -5.0,
        5.0,
    ).astype(np.float32)


# ============================================================
# Public feature extraction function
# ============================================================

def extract_feature(path):
    """
    Return the fixed-size log-Mel features expected by the
    existing Transformer training pipeline.
    """

    audio = load_mono(path)

    return waveform_to_logmel(audio)


def extract_feature_with_metadata(path):
    """
    Optional diagnostic version returning both features
    and audio trimming metadata.
    """

    audio, metadata = load_mono(
        path,
        return_metadata=True,
    )

    features = waveform_to_logmel(audio)

    return features, metadata
