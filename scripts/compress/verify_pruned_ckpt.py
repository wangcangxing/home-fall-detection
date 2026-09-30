#!/usr/bin/env python
"""校验剪枝/量化产物的 checkpoint 是否与预期一致（逐 key 比对，CPU 只读）。

用途：产物出现「能加载、但输出乱码」时，用来排除「存盘存坏了」这一类原因。
逐 key 惰性读取，不整份载入内存。

⚠️ 关键口径：剪枝会**重编号层**（丢掉第 5 层后，原第 6 层变成第 0..? 号），
所以**不能按 key 名直接比对**——那会把正常的重编号误报成"数值不一致"（第一版就犯过这个错，
误报 220 项）。必须先用 `--drop-layers` 给出被丢的原始层号，把基线的层号映射到产物层号再比。

只读主干分片（依据 `model.safetensors.index.json` 的 weight_map），
**不碰** `streammind_gate.safetensors` 那 64 个旁支 key。

用法：
  python scripts/compress/verify_pruned_ckpt.py \
      --ckpt E:\\MageVL\\compress\\pruneL30_wanda75 --drop-layers 5,10,15,21,26,31
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402


def main_shards(d: Path) -> list[Path]:
    """主干分片：有 index.json 就按它的 weight_map；否则取目录里的 safetensors。"""
    idx = Path(d) / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
        return sorted({Path(d) / s for s in wm.values()})
    return sorted(Path(d).glob("*.safetensors"))


class Reader:
    def __init__(self, files: list[Path]):
        from safetensors import safe_open

        self.handles = [(f, safe_open(str(f), framework="pt")) for f in files]
        self.keys = set()
        for _, h in self.handles:
            self.keys |= set(h.keys())

    def get(self, key: str):
        for _, h in self.handles:
            if key in h.keys():
                return h.get_tensor(key)
        raise KeyError(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(C.MODEL))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--drop-layers", default="",
                    help="被丢掉的原始 LLM 层号，逗号分隔；给了才能正确比对层内权重")
    ap.add_argument("--expect-vision-drop", type=int, default=0)
    args = ap.parse_args()

    base_dir, ckpt_dir = Path(args.base), Path(args.ckpt)
    base, ckpt = Reader(main_shards(base_dir)), Reader(main_shards(ckpt_dir))

    drop = sorted(int(x) for x in args.drop_layers.split(",") if x.strip())
    n_base_layers = 36
    keep = [i for i in range(n_base_layers) if i not in set(drop)]
    pos = {orig: new for new, orig in enumerate(keep)}

    def remap(key: str) -> str:
        if ".language_model.layers." not in key:
            return key
        head, rest = key.split(".language_model.layers.", 1)
        seg, tail = rest.split(".", 1)
        if int(seg) in pos:
            return f"{head}.language_model.layers.{pos[int(seg)]}.{tail}"
        return ""  # 该层已被丢弃

    print(f"基线主干 key {len(base.keys)} / 产物 key {len(ckpt.keys)}；丢弃层 {drop}（保留 {len(keep)} 层）")
    import torch

    missing_expected = sorted((base.keys - ckpt.keys) - set()) if not drop else []
    compared = identical = 0
    mismatches, reshaped, missing_in_ckpt, extra_in_ckpt = [], [], [], []
    for k in sorted(base.keys):
        if ".language_model.layers." in k:
            seg = k.split(".language_model.layers.")[1].split(".")[0]
            if int(seg) in set(drop):
                continue
        tgt = remap(k)
        if not tgt or tgt not in ckpt.keys:
            missing_in_ckpt.append(k)
            continue
        compared += 1
        a, b = base.get(k), ckpt.get(tgt)
        if tuple(a.shape) == tuple(b.shape):
            if torch.equal(a, b):
                identical += 1
            else:
                mismatches.append((k, tgt, (a.float() - b.float()).abs().max().item()))
        else:
            reshaped.append((k, tgt, tuple(a.shape), tuple(b.shape)))
            if "mlp." not in k:
                mismatches.append((k, tgt, "形状改变且非 FFN"))
    for k in ckpt.keys - base.keys:
        extra_in_ckpt.append(k)

    print(f"已比对 {compared} 个 key：逐元素一致 {identical}，形状改变 {len(reshaped)}")
    for k, t, s0, s1 in reshaped[:4]:
        print(f"    形状改变 {k} {s0} -> {s1}")
        print(f"             (产物侧 {t})")
    if missing_in_ckpt:
        print(f"⚠️ 基线有、产物按映射找不到 {len(missing_in_ckpt)} 个：{missing_in_ckpt[:6]}")
    if extra_in_ckpt:
        print(f"⚠️ 产物多出 {len(extra_in_ckpt)} 个：{sorted(extra_in_ckpt)[:6]}")
    if mismatches:
        print(f"❌ 非预期差异 {len(mismatches)} 项：")
        for k, t, d in mismatches[:10]:
            print(f"    {k}  (对照 {t})  {d}")
    else:
        print("✅ 未发现非预期差异（缺的正好是丢掉的层；改形状的正好是 FFN；其余逐元素一致）")

    try:
        a = ckpt.get("lm_head.weight")
        b = ckpt.get("model.language_model.embed_tokens.weight")
        print(f"lm_head 与 embed_tokens 逐元素相同？{torch.equal(a, b)}（tie_word_embeddings=False，应为 False）")
    except KeyError as e:  # noqa: BLE001
        print(f"缺少关键 key: {e}")

    C.save_json(Path(ckpt_dir) / "verify_report.json", {
        "base": str(base_dir), "ckpt": str(ckpt_dir), "drop_layers": drop,
        "n_base_keys": len(base.keys), "n_ckpt_keys": len(ckpt.keys),
        "n_compared": compared, "n_identical": identical, "n_reshaped": len(reshaped),
        "n_mismatch": len(mismatches), "missing_in_ckpt": missing_in_ckpt[:50],
        "extra_in_ckpt": sorted(extra_in_ckpt)[:50],
        "mismatch_sample": [{"base_key": k, "ckpt_key": t, "detail": str(d)} for k, t, d in mismatches[:20]],
    })


if __name__ == "__main__":
    main()
