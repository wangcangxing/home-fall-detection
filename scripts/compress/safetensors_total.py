#!/usr/bin/env python
"""量一个 HF 快照目录里真正被加载的权重体积（safetensors 分片之和）。

不量目录总大小：目录里还有 assets/、.cache/、streammind_gate.safetensors 等
非主干权重，混在一起会把「压缩前体积」这个分母算错。
"""
from __future__ import annotations

import json
from pathlib import Path


def weights_bytes(model_dir: Path) -> int:
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    files: list[Path] = []
    if index.exists():
        meta = json.loads(index.read_text(encoding="utf-8"))
        shards = sorted(set(meta["weight_map"].values()))
        files = [model_dir / s for s in shards]
    else:
        files = sorted(model_dir.glob("model*.safetensors"))
    return sum(f.stat().st_size for f in files if f.exists())


if __name__ == "__main__":
    import sys

    d = Path(sys.argv[1] if len(sys.argv) > 1 else r"E:\MageVL\Mage-VL")
    n = weights_bytes(d)
    print(f"{d} -> {n} bytes = {n/1024**3:.3f} GiB = {n/1e9:.3f} GB")
