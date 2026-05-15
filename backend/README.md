# Backend

Inference server. Tested on Ubuntu; other Linux distributions may also work.

## Recommended environment

- Python 3.12+
- NVIDIA GPU (VRAM 30GB+)
- PyTorch 2.8 + CUDA 12.8
- `ffmpeg` on PATH

## Setup

```bash
bash backend/scripts/setup.sh
```

Installs dependencies and downloads SAM2 / Casper weights in one go. Safe to re-run — it resumes from where it left off.

## Run

From the repository root:

```bash
python -m backend.main
```

The backend is ready when the following log message appears:

```
[INFO] backend.predictors.casper: casper adapter loaded
```
