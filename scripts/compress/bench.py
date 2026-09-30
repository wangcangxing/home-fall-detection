#!/usr/bin/env python
"""统一基准测量：把一个模型变体跑固定评测集，产出体积/显存/延迟数据。

用法：
  python scripts/compress/bench.py --tag bf16
  python scripts/compress/bench.py --tag bnb4 --quant bnb4

输出：results/compress_bench_<tag>.json（入库）+ stdout 摘要。
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import common as C  # noqa: E402


def build_quant_config(name: str):
    import torch
    from transformers import BitsAndBytesConfig

    if name == "none":
        return None
    if name == "bnb4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if name == "bnb8":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise SystemExit(f"未知 quant: {name}")


def layer_census(model) -> dict:
    """统计模块类型分布；量化覆盖率必须显式看 Linear4bit（坑点 #20）。"""
    import torch
    from collections import Counter

    kinds = Counter(type(m).__name__ for m in model.modules())
    return {
        "module_types_top": kinds.most_common(10),
        "n_linear_plain": sum(1 for m in model.modules() if type(m) is torch.nn.Linear),
        "n_linear4bit": kinds.get("Linear4bit", 0),
        "n_linear8bit": kinds.get("Linear8bitLt", 0),
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="产物名，如 bf16 / bnb4")
    ap.add_argument("--quant", default="none", choices=["none", "bnb4", "bnb8"])
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="processor.image_processor.size['longest_edge']；默认出厂值")
    ap.add_argument("--only", default=None, help="只跑 id 含该子串的项（调试用）")
    ap.add_argument("--model", default=None, help="模型目录；默认 E:\\MageVL\\Mage-VL")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results_path = C.RESULTS / (args.out or f"compress_bench_{args.tag}.json")
    buf = []
    model_dir = Path(args.model) if args.model else C.MODEL
    rec = {"tag": args.tag, "quant": args.quant, "model": str(model_dir),
           "max_pixels": args.max_pixels,
           "max_new_tokens": args.max_new_tokens, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    processor = C.load_processor(model_dir)
    C.set_image_budget(processor, args.max_pixels)

    qc = build_quant_config(args.quant)
    import torch
    torch.cuda.reset_peak_memory_stats()
    model, load_sec = C.load_model(quantization_config=qc, model_dir=model_dir)
    mem_after_load = C.gpu_mem()
    rec.update({"load_sec": round(load_sec, 1), "mem_after_load": mem_after_load})
    C.log(f"[{args.tag}] load_sec={load_sec:.1f} dtype={model.dtype} "
          f"weights_resident={mem_after_load['allocated_gb']}GB", buf)
    rec["census"] = layer_census(model)
    C.log(f"[{args.tag}] census={rec['census']}", buf)

    # 磁盘体积：bf16 直接量模型目录里的 safetensors（量化变体由各自脚本记录）
    from safetensors_total import weights_bytes  # noqa: E402  (同目录)
    rec["disk"] = {"source_weights_bytes": weights_bytes(model_dir)}
    C.log(f"[{args.tag}] 源权重 {rec['disk']['source_weights_bytes']/1024**3:.2f} GiB", buf)

    items = C.eval_set()
    if args.only:
        items = [it for it in items if args.only in it["id"]]
    out_items = []
    for it in items:
        img = C.load_item_image(it) if it["kind"] == "image" else None
        r = C.generate_measured(model, processor, it["prompt"],
                                image=img, kind=it["kind"],
                                max_new_tokens=args.max_new_tokens)
        r["id"] = it["id"]
        r["source"] = it.get("source")
        out_items.append(r)
        C.log(f"[{args.tag}] {it['id']:>7} {r.get('prompt_tokens')}->{r.get('new_tokens')} "
              f"{r.get('gen_sec')}s {r.get('tok_per_s')}tok/s peak={r.get('peak_allocated_gb')}GB "
              f"{('ERR ' + r['error']) if r.get('error') else ''}", buf)

    ok = [r for r in out_items if not r.get("error")]
    rec["items"] = out_items
    rec["summary"] = {
        "n_items": len(out_items),
        "n_ok": len(ok),
        "n_err": len(out_items) - len(ok),
        "mean_tok_per_s": round(sum(r["tok_per_s"] for r in ok) / max(len(ok), 1), 3),
        "max_peak_allocated_gb": max([r["peak_allocated_gb"] for r in ok], default=None),
        "total_gen_sec": round(sum(r["gen_sec"] for r in ok), 1),
        "errors": [r["error"] for r in out_items if r.get("error")],
    }
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    C.save_json(results_path, rec)
    C.log(f"[{args.tag}] summary={rec['summary']}", buf)
    C.log(f"[{args.tag}] -> {results_path}", buf)
    (C.WORK / f"bench_{args.tag}.log").write_text("\n".join(buf) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
