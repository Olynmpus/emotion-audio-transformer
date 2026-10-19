# Project B — Emotion-Aware Auditory Representation

A CPU-first PyTorch project for a Windows computer with 8 GB RAM and no dedicated GPU.

## Research question

Can a small Transformer encoder–decoder reconstruct emotion-related speech features after simulated acoustic degradation, while retaining useful information for emotion classification?

This first version operates on log-Mel spectrograms rather than raw waveforms. It is intentionally small and is a learning/research prototype, not a clinical tool or a validated model of human emotion perception.

## Hardware-conscious defaults

- CPU only; no CUDA required
- 16 kHz audio
- 40 Mel frequency bins
- Maximum 128 time frames per utterance
- Batch size 8
- Transformer width 48, 2 encoder layers, 2 decoder layers, 4 attention heads
- Features stored individually on disk as float16; they are loaded per batch rather than all into RAM

## 1. Install

Open PowerShell in this folder. Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run commands using `.\.venv\Scripts\python.exe` instead.

## 2. Put the dataset in place

Extract the dataset so the layout is:

```text
data/
  raw/
    Angry/
      some_file.wav
    Disgusted/
      some_file.wav
    Fearful/
    Happy/
    Neutral/
    Sad/
    Surprised/   # "Suprised" is also accepted
```

The folder names are used as labels. Keep the original filenames if possible; they may contain speaker/source identifiers.

## 3. Audit the dataset first

```powershell
python src\audit_dataset.py --data-dir .
```

This checks file counts, durations, sample rates, labels, and likely duplicate audio. Review the report before training. The merged collection combines four datasets, so recording conditions and actor demographics may differ.

## 4. Prepare compact features

```powershell
python src\prepare_features.py --data-dir . --feature-dir features
```

This can take a while on a CPU. It writes one small `.npz` file per WAV, so it does not load the whole dataset into RAM. It uses 3 seconds maximum audio duration and pads/truncates features to 128 frames.

To test the pipeline first:

```powershell
python src\prepare_features.py --data-dir . --feature-dir features --limit 200
```

The `--limit 200` option selects files round-robin across emotion classes for a useful small pilot. Use a separate feature directory for the pilot, or remove the pilot `features` directory before preparing the full dataset.

## 5. Train a small model

Start with a subset to estimate runtime:

```powershell
python src\train.py --feature-dir features --epochs 3 --max-files 1000
```

Then use the prepared full dataset:

```powershell
python src\train.py --feature-dir features --epochs 15
```

Optional controls: `--batch-size 4`, `--workers 0`, `--seed 42`. Start with batch size 4 if Windows becomes sluggish.

## What the model does

1. Encoder: represents the degraded log-Mel sequence using self-attention.
2. Decoder: attends to the encoder representation and reconstructs the clean log-Mel sequence.
3. Emotion head: predicts one of seven folder labels from the encoded sequence.
4. Training objective: combines reconstruction loss and emotion cross-entropy.

During training, the input is randomly corrupted in feature space with additive noise and frequency-band attenuation. This is a controlled first experiment, not a physically exact simulation of hearing loss. A later version can degrade the waveform before feature extraction.

## Important scientific safeguards

- Do not treat acted emotion labels as ground truth for how real listeners perceive emotion.
- Do not claim generalisation based on a random file split alone. The four source datasets have different speakers and protocols. Inspect filenames and source IDs; ideally split by speaker and evaluate on a held-out source dataset.
- The current starter uses a deterministic file-level split so the pipeline runs immediately. It is a baseline only and may leak speaker identity across splits.
- Compare against a simple baseline (e.g., a small CNN or logistic regression on pooled log-Mel features) before attributing improvements to Transformers.
- The model is not a clinical diagnostic system.

## Output

Training writes:
- `outputs/best_model.pt`
- `outputs/metrics.csv`
- `outputs/classification_report.txt`

Keep the dataset itself out of version control.
