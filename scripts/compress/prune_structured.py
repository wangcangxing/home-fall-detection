#!/usr/bin/env python
"""Mage-VL 端侧结构化剪枝：深度（丢层）+ FFN 宽度（丢神经元），产出真实更小的模型。

为什么是结构化而不是非结构化：非结构化稀疏（2:4/随机置零）不改变权重张量形状，
通用 GPU/NPU 上拿不到加速；端侧要的是**真的更小**的模型。所以这里物理删除结构。

已核实的模型结构（E:\\MageVL\\Mage-VL\\modeling_mage_vl.py）：
  - LLM 主干：model.model.language_model.layers  → 36 × Qwen3DecoderLayer
      · 每层 mlp：gate_proj[I,H] / up_proj[I,H] / down_proj[H,I]，I=9728
  - 视觉塔：model.model.visual.encoder.layers     → 24 × MageVLVisionEncoderLayer
  - 关键坑：transformers 的 `from_config` 会 **deepcopy** 配置，所以子模块持有的是
    自己的 config 副本；改层数必须同时改 model.config.text_config 与
    model.model.language_model.config，否则 Qwen3 建 mask 时会和层数对不上。

用法：
  python scripts/compress/prune_structured.py --tag pruneL30_ffn75 --llm-drop 6 --ffn-keep 0.75
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

COPY_SUFFIXES = {".py", ".json", ".jinja", ".txt"}


# --------------------------------------------------------------------------- #
# 配置同步
# --------------------------------------------------------------------------- #
def _uniq(cfgs):
    out, seen = [], set()
    for c in cfgs:
        if c is not None and id(c) not in seen:
            seen.add(id(c))
            out.append(c)
    return out


def text_cfgs(model):
    lm = model.model.language_model
    return _uniq([model.config.text_config, getattr(lm, "config", None),
                  getattr(model.config, "text_config", None)])


def vision_cfgs(model):
    vis = model.model.visual
    enc = getattr(vis, "encoder", None)
    return _uniq([model.config.vision_config, getattr(vis, "config", None),
                  getattr(enc, "config", None) if enc is not None else None])


# --------------------------------------------------------------------------- #
# 选层
# --------------------------------------------------------------------------- #
def pick_drop(n: int, k: int, strategy: str) -> list[int]:
    if k <= 0:
        return []
    if k >= n - 1:
        raise ValueError(f"最多只能丢 {n-2} 层（首尾必须保留），收到 k={k}")
    if strategy == "last":
        drop = list(range(n - k, n))
    elif strategy == "first":
        drop = list(range(1, k + 1))          # 保留第 0 层（embedding 直连的那层最敏感）
    elif strategy == "uniform":
        drop = sorted({int(round((i + 1) * n / (k + 1))) for i in range(k)})
        drop = [min(max(d, 1), n - 2) for d in drop]
        drop = sorted(set(drop))
        j = 1
        while len(drop) < k and j < n - 1:     # 去重后补齐
            if j not in drop:
                drop.append(j)
            j += 1
        drop = sorted(drop)[:k]
    else:
        raise ValueError(strategy)
    return drop


def drop_decoder_layers(model, k: int, strategy: str = "uniform") -> dict:
    lm = model.model.language_model
    layers = lm.layers
    n = len(layers)
    drop = pick_drop(n, k, strategy)
    keep = [i for i in range(n) if i not in set(drop)]
    new_layers = nn.ModuleList([layers[i] for i in keep])
    # 必须重编号 layer_idx：解码层构造时拿到的是**原始**下标，KV cache 只按新层数建槽，
    # 不重编号会在 cache_utils 的 self.layers[layer_idx] 处 IndexError（实测踩过）。
    for new_idx, layer in enumerate(new_layers):
        sa = getattr(layer, "self_attn", None)
        if sa is not None and hasattr(sa, "layer_idx"):
            sa.layer_idx = new_idx
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_idx
    lm.layers = new_layers
    for cfg in text_cfgs(model):
        try:
            cfg.num_hidden_layers = len(keep)
            lt = getattr(cfg, "layer_types", None)
            if isinstance(lt, list) and len(lt) == n:
                cfg.layer_types = [lt[i] for i in keep]
        except Exception as e:  # noqa: BLE001
            print(f"  ! 更新 text config 失败: {e}", flush=True)
    return {"n_before": n, "n_after": len(keep), "dropped": drop, "strategy": strategy}


def drop_vision_layers(model, k: int, strategy: str = "uniform") -> dict:
    enc = model.model.visual.encoder
    layers = enc.layers
    n = len(layers)
    drop = pick_drop(n, k, strategy)
    keep = [i for i in range(n) if i not in set(drop)]
    enc.layers = nn.ModuleList([layers[i] for i in keep])
    for cfg in vision_cfgs(model):
        try:
            cfg.num_hidden_layers = len(keep)
        except Exception as e:  # noqa: BLE001
            print(f"  ! 更新 vision config 失败: {e}", flush=True)
    return {"n_before": n, "n_after": len(keep), "dropped": drop, "strategy": strategy}


# --------------------------------------------------------------------------- #
# FFN 宽度剪枝
# --------------------------------------------------------------------------- #
def prune_ffn(model, keep_ratio: float, strategy: str = "wanda", seed: int = 0,
              act_imp: list | None = None) -> dict:
    """保留前 keep_ratio 比例的中间神经元，物理裁掉其余。

    strategy:
      wanda      = |激活_j| · ||down[:,j]||（校准集上统计，默认；见下面的实测教训）
      importance = ||gate[j]||·||down[:,j]||（纯权重，**实测把模型打崩**）
      first      = 直接取前 K 个（对照）
      random     = 随机取 K 个（对照）

    ⚠️ 实测教训（2026-09-30）：纯权重范数判据在本模型上**比随机还差**——保留权重最大的
    75% 会让输出彻底崩坏，而随机保留 75% 输出正常。原因推断：这类模型有"权重小但激活
    极大"的离群神经元，按权重排序恰好把它们整批丢掉。判据必须含激活。
    """
    rows, before_i, after_i = [], None, None
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for i, layer in enumerate(model.model.language_model.layers):
        mlp = layer.mlp
        gw, uw, dw = mlp.gate_proj.weight.data, mlp.up_proj.weight.data, mlp.down_proj.weight.data
        I = gw.shape[0]
        K = max(1, int(round(I * keep_ratio)))
        if strategy == "wanda":
            if act_imp is None:
                raise ValueError("wanda 策略需要先收集校准激活（act_imp）")
            imp = act_imp[i].to(gw.device).float() * dw.float().norm(dim=0)
            keep = torch.topk(imp, K, largest=True).indices.sort().values
        elif strategy == "importance":
            imp = gw.float().norm(dim=1) * dw.float().norm(dim=0)
            keep = torch.topk(imp, K, largest=True).indices.sort().values
        elif strategy == "first":
            keep = torch.arange(K, device=gw.device)
        elif strategy == "random":
            keep = torch.randperm(I, generator=gen)[:K].sort().values.to(gw.device)
        else:
            raise ValueError(strategy)
        dev, dt = gw.device, gw.dtype
        # gate/up: [I,H] 取行；down: [H,I] 取列 → 新 Linear 的 out_features 是 H=dw.shape[0]
        new_gate = nn.Linear(gw.shape[1], K, bias=False, device=dev, dtype=dt)
        new_up = nn.Linear(uw.shape[1], K, bias=False, device=dev, dtype=dt)
        new_down = nn.Linear(K, dw.shape[0], bias=False, device=dev, dtype=dt)
        with torch.no_grad():
            new_gate.weight.copy_(gw[keep])
            new_up.weight.copy_(uw[keep])
            new_down.weight.copy_(dw[:, keep])
        mlp.gate_proj, mlp.up_proj, mlp.down_proj = new_gate, new_up, new_down
        before_i, after_i = I, K
        rows.append({"layer": i, "I_before": I, "I_after": K})
    for cfg in text_cfgs(model):
        try:
            cfg.intermediate_size = after_i
        except Exception as e:  # noqa: BLE001
            print(f"  ! 更新 intermediate_size 失败: {e}", flush=True)
    return {"keep_ratio": keep_ratio, "strategy": strategy, "seed": seed,
            "I_before": before_i, "I_after": after_i, "n_layers_pruned": len(rows)}


# --------------------------------------------------------------------------- #
# 激活感知的重要性（Wanda 式）：在**与评测集不同的**固定校准文本上统计
# --------------------------------------------------------------------------- #
CALIB_PROMPTS = [
    "Summarize the following idea in one sentence: a small sensor wakes up a bigger model.",
    "Name two advantages of running a neural network on a low-power chip.",
    "What is the difference between a photograph and a video?",
    "Write one sentence about how a camera stores images.",
    "Give three examples of everyday objects made of metal.",
    "Explain what a battery does, in one sentence.",
    "List four colours in the order they appear in a rainbow.",
    "Describe the taste of a lemon in one short sentence.",
]


def collect_ffn_activation_importance(model, processor, prompts=None) -> list:
    """累计每个 FFN 中间神经元的平均 |激活|。

    钩在 `mlp.down_proj` 的**输入**上，取到的正是 act(gate(x)) * up(x)（形状 [B,T,I]），
    所以每个中间神经元的激活强度就是这一层的输入绝对值均值。
    """
    prompts = prompts or CALIB_PROMPTS
    layers = model.model.language_model.layers
    acc = [None] * len(layers)
    counts = [0] * len(layers)
    hooks = []

    def make(i):
        def h(module, args):
            x = args[0].detach()
            s = x.abs().float().sum(dim=(0, 1))
            acc[i] = s if acc[i] is None else acc[i] + s
            counts[i] += int(x.shape[0] * x.shape[1])
        return h

    for i, layer in enumerate(layers):
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make(i)))
    try:
        with torch.inference_mode():
            for p in prompts:
                inputs = C.build_inputs(processor, model, p, kind="text")
                model(**inputs)
    finally:
        for h in hooks:
            h.remove()
    return [(a / max(c, 1)).cpu() for a, c in zip(acc, counts)]


# --------------------------------------------------------------------------- #
# 保存 / 回读校验
# --------------------------------------------------------------------------- #
def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def save_standalone(model, out_dir: Path, src_dir: Path = C.MODEL) -> int:
    """先拷代码/分词器文件，再 save_pretrained（后者会写入剪枝后的 config.json）。

    排除权重与旧索引：原快照的 `model.safetensors.index.json` 指向的是**原分片名**，
    拷进产物目录会误导加载器去找不存在的 model-0000x-of-0000y.safetensors。
    ⚠️ 只按 `.safetensors` / 索引名排除，**不要**用 `startswith("model")`——
    那会把 `modeling_mage_vl.py` 一起排掉，产物就再也加载不起来了（实测踩过）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.iterdir():
        if not p.is_file() or p.suffix not in COPY_SUFFIXES:
            continue
        if p.suffix == ".safetensors" or p.name == "model.safetensors.index.json":
            continue
        shutil.copy2(p, out_dir / p.name)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    return C.dir_size_bytes(out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--llm-drop", type=int, default=0)
    ap.add_argument("--vision-drop", type=int, default=0)
    ap.add_argument("--strategy", default="uniform", choices=["uniform", "last", "first"])
    ap.add_argument("--ffn-keep", type=float, default=1.0, help="LLM FFN 中间维保留比例")
    ap.add_argument("--ffn-strategy", default="wanda",
                    choices=["wanda", "importance", "first", "random"])
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--items", default=None, help="只跑指定 id（逗号分隔），用于消融实验")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--reload-check", action="store_true", help="存盘后用 from_pretrained 回读并跑一张图")
    args = ap.parse_args()

    out_dir = C.WORK / args.tag
    results_path = C.RESULTS / f"compress_prune_{args.tag}.json"
    buf = []
    rec = {"tag": args.tag, "llm_drop": args.llm_drop, "vision_drop": args.vision_drop,
           "strategy": args.strategy, "ffn_keep": args.ffn_keep,
           "ffn_strategy": args.ffn_strategy,
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    processor = C.load_processor()
    model, load_sec = C.load_model()
    p0 = n_params(model)
    rec.update({"load_sec": round(load_sec, 1), "params_before": p0,
                "gpu_before_gb": C.gpu_mem()["allocated_gb"]})
    C.log(f"[{args.tag}] bf16 加载 {load_sec:.1f}s 参数 {p0/1e9:.4f}B "
          f"显存 {C.gpu_mem()['allocated_gb']}GB", buf)

    t0 = time.time()
    if args.llm_drop:
        rec["llm_prune"] = drop_decoder_layers(model, args.llm_drop, args.strategy)
        C.log(f"[{args.tag}] LLM 层: {rec['llm_prune']['n_before']} -> "
              f"{rec['llm_prune']['n_after']}（丢 {rec['llm_prune']['dropped']}）", buf)
    if args.vision_drop:
        rec["vision_prune"] = drop_vision_layers(model, args.vision_drop, args.strategy)
        C.log(f"[{args.tag}] 视觉层: {rec['vision_prune']['n_before']} -> "
              f"{rec['vision_prune']['n_after']}（丢 {rec['vision_prune']['dropped']}）", buf)
    if args.ffn_keep < 1.0:
        act_imp = None
        if args.ffn_strategy == "wanda":
            t1 = time.time()
            act_imp = collect_ffn_activation_importance(model, processor)
            rec["calib"] = {"n_prompts": len(CALIB_PROMPTS), "sec": round(time.time() - t1, 1),
                            "prompts": CALIB_PROMPTS}
            C.log(f"[{args.tag}] 校准激活收集完成（{len(CALIB_PROMPTS)} 条文本 / "
                  f"{rec['calib']['sec']}s）", buf)
        rec["ffn"] = prune_ffn(model, args.ffn_keep, args.ffn_strategy, act_imp=act_imp)
        C.log(f"[{args.tag}] FFN 中间维 {rec['ffn']['I_before']} -> {rec['ffn']['I_after']} "
              f"× {rec['ffn']['n_layers_pruned']} 层（策略 {args.ffn_strategy}）", buf)
    rec["prune_sec"] = round(time.time() - t0, 1)

    torch.cuda.empty_cache()
    p1 = n_params(model)
    rec.update({"params_after": p1, "param_ratio": round(p1 / p0, 4),
                "gpu_after_gb": C.gpu_mem()["allocated_gb"]})
    C.log(f"[{args.tag}] 剪枝后参数 {p1/1e9:.4f}B（{rec['param_ratio']*100:.1f}%），"
          f"GPU 常驻 {C.gpu_mem()['allocated_gb']}GB，耗时 {rec['prune_sec']}s", buf)

    items = C.eval_set()
    if args.items:
        want = {s.strip() for s in args.items.split(",") if s.strip()}
        items = [it for it in items if it["id"] in want]
    out_items = []
    for it in items:
        img = C.load_item_image(it) if it["kind"] == "image" else None
        r = C.generate_measured(model, processor, it["prompt"], image=img,
                                kind=it["kind"], max_new_tokens=args.max_new_tokens)
        r["id"] = it["id"]
        out_items.append(r)
        C.log(f"[{args.tag}] {it['id']:>7} {r.get('prompt_tokens')}->{r.get('new_tokens')} "
              f"{r.get('gen_sec')}s {r.get('tok_per_s')}tok/s peak={r.get('peak_allocated_gb')}GB "
              f"{('ERR ' + r['error']) if r.get('error') else ''}", buf)
    ok = [r for r in out_items if not r.get("error")]
    rec["items"] = out_items
    rec["summary"] = {
        "n_ok": len(ok), "n_err": len(out_items) - len(ok),
        "mean_tok_per_s": round(sum(r["tok_per_s"] for r in ok) / max(len(ok), 1), 3),
        "max_peak_allocated_gb": max([r["peak_allocated_gb"] for r in ok], default=None),
    }
    C.log(f"[{args.tag}] summary={rec['summary']}", buf)

    if not args.no_save:
        bytes_ = save_standalone(model, out_dir)
        rec["saved"] = {"dir": str(out_dir), "bytes": bytes_,
                        "gib": round(bytes_ / 1024 ** 3, 3)}
        C.log(f"[{args.tag}] 存盘 {bytes_/1024**3:.2f}GiB -> {out_dir}", buf)

    if args.reload_check:
        del model
        torch.cuda.empty_cache()
        m2, sec = C.load_model(model_dir=out_dir)
        p2 = n_params(m2)
        C.log(f"[{args.tag}] 回读校验: 加载 {sec:.1f}s，参数 {p2/1e9:.4f}B "
              f"（与剪枝后一致={p2 == p1}）", buf)
        r = C.generate_measured(m2, processor, "Describe this image in detail.",
                                image=C.DOG, kind="image", max_new_tokens=48)
        rec["reload_check"] = {"load_sec": round(sec, 1), "params": p2, "params_match": p2 == p1,
                               "answer": r.get("answer"), "error": r.get("error"),
                               "gen_sec": r.get("gen_sec")}
        C.log(f"[{args.tag}] 回读后推理: {r.get('gen_sec')}s "
              f"{'ERR ' + r['error'] if r.get('error') else r.get('answer', '')[:200]}", buf)

    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    C.save_json(results_path, rec)
    C.log(f"[{args.tag}] -> {results_path}", buf)
    (C.WORK / f"prune_{args.tag}.log").write_text("\n".join(buf) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
