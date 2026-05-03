"""SAM3 セットアップ補助スクリプト.

行うこと:
  1. `git submodule update --init vendor/sam3` を実行（未取得なら）
  2. `pip install -e ./vendor/sam3` を実行（任意、--no-install で skip）
  3. AEmotionStudio/sam3 (HF, non-gated) から `sam3.safetensors` を
     `vendor/sam3/checkpoints/sam3.safetensors` にダウンロード

使い方:
  python scripts/setup_sam3.py
  python scripts/setup_sam3.py --no-install   # pip install をスキップ
  python scripts/setup_sam3.py --skip-weights # 重みダウンロードをスキップ
"""
import argparse
import os
import subprocess
import sys


REPO_ID = "AEmotionStudio/sam3"
REPO_URL = "https://github.com/facebookresearch/sam3"
WEIGHT_FILE = "sam3.safetensors"


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_submodule(root: str) -> None:
    sam3_dir = os.path.join(root, "vendor", "sam3")
    sentinel = os.path.join(sam3_dir, "pyproject.toml")
    if os.path.exists(sentinel):
        print(f"[ok] vendor/sam3 already present at {sam3_dir}")
        return

    # 1. submodule として登録済みなら update --init で取得
    print("[setup] trying git submodule update --init vendor/sam3...")
    result = subprocess.run(
        ["git", "submodule", "update", "--init", "vendor/sam3"],
        cwd=root, capture_output=True, text=True,
    )
    if result.returncode == 0 and os.path.exists(sentinel):
        print(f"[ok] vendor/sam3 initialized via submodule")
        return
    if result.stderr:
        print(result.stderr.rstrip())

    # 2. 未登録（.gitmodules にエントリはあるが gitlink が無い等）なら直接 clone
    print("[setup] submodule not registered; cloning vendor/sam3 directly...")
    os.makedirs(os.path.dirname(sam3_dir), exist_ok=True)
    _run(["git", "clone", REPO_URL, sam3_dir], cwd=root)


def pip_install_sam3(root: str) -> None:
    sam3_dir = os.path.join(root, "vendor", "sam3")
    print("[setup] pip install -e vendor/sam3 ...")
    _run([sys.executable, "-m", "pip", "install", "-e", sam3_dir], cwd=root)


def download_weights(root: str) -> None:
    from huggingface_hub import hf_hub_download

    out_dir = os.path.join(root, "vendor", "sam3", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, WEIGHT_FILE)
    if os.path.exists(out_path):
        print(f"[ok] weights already present at {out_path}")
        return
    print(f"[setup] downloading {WEIGHT_FILE} from {REPO_ID}...")
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=WEIGHT_FILE,
        local_dir=out_dir,
    )
    # local_dir に直接ファイル名で配置されることを期待。違う場合はリンクを張る
    if os.path.abspath(downloaded) != os.path.abspath(out_path):
        if os.path.exists(out_path):
            os.unlink(out_path)
        try:
            os.symlink(downloaded, out_path)
        except OSError:
            import shutil
            shutil.copy2(downloaded, out_path)
    print(f"[ok] weights at {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-install", action="store_true", help="skip pip install -e vendor/sam3")
    parser.add_argument("--skip-weights", action="store_true", help="skip weights download")
    args = parser.parse_args()

    root = _project_root()
    ensure_submodule(root)
    if not args.no_install:
        pip_install_sam3(root)
    if not args.skip_weights:
        download_weights(root)
    print("[done] SAM3 setup complete. Run with: OMNIMATTE_SAM_VERSION=3 python -m server.main")


if __name__ == "__main__":
    main()
