"""SAM2 / Casper のモデル重みをダウンロードする。

実行方法:
    cd backend
    python scripts/download_models.py
"""

import os
import sys
import urllib.request

from huggingface_hub import snapshot_download
import gdown


_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_HERE)


SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM2_DEST = os.path.join(_BACKEND_DIR, "models", "sam2", "sam2.1_hiera_large.pt")

CASPER_HF_REPO = "alibaba-pai/Wan2.1-Fun-1.3B-InP"
CASPER_HF_DEST = os.path.join(
    _BACKEND_DIR, "vendor", "gen-omnimatte-public", "models",
    "Diffusion_Transformer", "Wan2.1-Fun-1.3B-InP",
)

CASPER_GDRIVE_ID = "1n3Sv4d0pbTjfa5UhypEhTaylrSy2X4C1"
CASPER_GDRIVE_DEST = os.path.join(
    _BACKEND_DIR, "vendor", "gen-omnimatte-public", "models",
    "Casper", "wan2.1_fun_1.3b_casper.safetensors",
)


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(downloaded * 100 / total_size, 100.0)
    mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    sys.stdout.write(f"\r  {pct:5.1f}%  ({mb:.1f} / {total_mb:.1f} MiB)")
    sys.stdout.flush()


def download_sam2() -> None:
    if os.path.exists(SAM2_DEST):
        print(f"[sam2] already exists: {SAM2_DEST}")
        return
    print(f"[sam2] downloading {SAM2_URL}")
    os.makedirs(os.path.dirname(SAM2_DEST), exist_ok=True)
    urllib.request.urlretrieve(SAM2_URL, SAM2_DEST, _progress_hook)
    sys.stdout.write("\n")
    print(f"[sam2] saved to {SAM2_DEST}")


def download_casper_hf() -> None:
    print(f"[casper-hf] downloading {CASPER_HF_REPO} -> {CASPER_HF_DEST}")
    os.makedirs(CASPER_HF_DEST, exist_ok=True)
    snapshot_download(repo_id=CASPER_HF_REPO, local_dir=CASPER_HF_DEST)
    print(f"[casper-hf] done")


def download_casper_gdrive() -> None:
    if os.path.exists(CASPER_GDRIVE_DEST):
        print(f"[casper-safetensors] already exists: {CASPER_GDRIVE_DEST}")
        return
    print(f"[casper-safetensors] downloading gdrive:{CASPER_GDRIVE_ID}")
    os.makedirs(os.path.dirname(CASPER_GDRIVE_DEST), exist_ok=True)
    gdown.download(id=CASPER_GDRIVE_ID, output=CASPER_GDRIVE_DEST, quiet=False)
    print(f"[casper-safetensors] saved to {CASPER_GDRIVE_DEST}")


def main() -> None:
    download_sam2()
    download_casper_hf()
    download_casper_gdrive()


if __name__ == "__main__":
    main()
