"""Casper パイプラインのエントリポイント。

`server.model.CASPER_BACKEND` の値（`"wan"` | `"cogvideox"`）に応じて、
`pipeline_wan` か `pipeline_cogvideox` のどちらかから関数を re-export する。
本サーバや sidecar の上位レイヤは常に `casper_server.pipeline` から関数を参照すれば
バックエンドの違いを意識しなくてよい。
"""
from server.model import CASPER_BACKEND


if CASPER_BACKEND == "wan":
    from casper_server.pipeline_wan import (
        build_default_config,
        load_pipeline,
        run_one_seq,
    )
elif CASPER_BACKEND == "cogvideox":
    from casper_server.pipeline_cogvideox import (
        build_default_config,
        load_pipeline,
        run_one_seq,
    )
else:
    raise ValueError(f"unsupported CASPER_BACKEND: {CASPER_BACKEND!r}")


__all__ = ["build_default_config", "load_pipeline", "run_one_seq"]
