"""sidecar 内に保持する Casper 出力 mp4 のキャッシュ。

`/preload` で先回り計算した結果を保存し、`/run` で同じ (video, mask) ペアが
来たらキャッシュ hit でその mp4 バイトを即返す。

キャッシュ key = `(video bytes md5, mask bytes md5)`。単一スロット（base video が
変わったら新しい hash が来てキャッシュは自然に置き換わる）。
"""
import hashlib
import threading
from dataclasses import dataclass


def hash_file(path: str) -> bytes:
    """ファイルの md5 ダイジェストを bytes で返す。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.digest()


@dataclass
class CacheEntry:
    video_hash: bytes
    mask_hash: bytes
    mp4_bytes: bytes


class OutputCache:
    """単一スロット。`/preload` と `/run` の双方からアクセスされる。"""

    def __init__(self) -> None:
        self._entry: CacheEntry | None = None
        self._lock = threading.Lock()

    def set(self, video_hash: bytes, mask_hash: bytes, mp4_bytes: bytes) -> None:
        with self._lock:
            self._entry = CacheEntry(video_hash, mask_hash, mp4_bytes)

    def get(self, video_hash: bytes, mask_hash: bytes) -> bytes | None:
        with self._lock:
            if self._entry is None:
                return None
            if self._entry.video_hash == video_hash and self._entry.mask_hash == mask_hash:
                return self._entry.mp4_bytes
            return None

    def clear(self) -> None:
        with self._lock:
            self._entry = None


output_cache = OutputCache()
