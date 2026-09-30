#!/usr/bin/env python
"""Mage-VL 端侧量化：group-wise 非对称 RTN weight-only 量化（纯 PyTorch，无额外依赖）。

为什么自研而不是直接用 GPTQ/AWQ：
  Mage-VL 是 trust_remote_code 的**自定义架构**（mage_vl），GPTQ/AWQ 需要模型类
  与其量化集成配合；本机实测 `llm-compressor` 在该 venv 下 "from versions: none"，
  其余量化库的 PyPI 访问被网络重置（见 results/compress_env_probe.txt）。
  自研 RTN 不依赖任何新库，且产出的 int 权重 + group scale 正是端侧运行时要吃的格式。

量化公式（非对称、按 group 求 min/max）：
    w ≈ q * scale + zero,  scale = (max-min)/(2^b - 1),  q ∈ [0, 2^b-1]
    4bit 时两个 nibble 打包进一个 uint8（低位在前）。

产物：
    E:\\MageVL\\compress\\<tag>\\model_int.safetensors  + quant_config.json
    <repo>\\results\\compress_quant_<tag>.json          （实测证据，入库）
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

COPY_SUFFIXES = {".py", ".json", ".jinja", ".txt"}


# --------------------------------------------------------------------------- #
# 打包 / 解包
# --------------------------------------------------------------------------- #
def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """uint8 [out, in] (0..15) -> uint8 [out, in//2]，低位 nibble 存偶数下标。"""
    if q.shape[1] % 2:
        raise ValueError(f"4bit 打包要求 in_features 为偶数，实际 {q.shape[1]}")
    return (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)


def unpack_int4(p: torch.Tensor) -> torch.Tensor:
    """把最后一个维度上的 nibble 展开：任意形状 [..., in//2] -> [..., in]。

    必须支持 N 维：`QuantEmbedding` 查表得到的是 [B, T, hidden//2]（QuantLinear 是 2 维）。
    """
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    out = torch.empty(*p.shape[:-1], p.shape[-1] * 2, dtype=torch.uint8, device=p.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


# --------------------------------------------------------------------------- #
# 单层量化
# --------------------------------------------------------------------------- #
def quantize_weight(w: torch.Tensor, bits: int, group_size: int):
    """返回 (packed_q, scales_bf16, zeros_bf16, rel_err)；w 为 [out, in]。"""
    w32 = w.detach().to(torch.float32)
    out_f, in_f = w32.shape
    g = in_f if group_size <= 0 else min(group_size, in_f)
    if in_f % g:
        raise ValueError(f"group_size={g} 不能整除 in_features={in_f}")
    ng = in_f // g
    wg = w32.view(out_f, ng, g)
    mn = wg.min(dim=-1, keepdim=True).values
    mx = wg.max(dim=-1, keepdim=True).values
    qmax = float(2 ** bits - 1)
    scale = (mx - mn).clamp_min(1e-8) / qmax
    q = torch.round((wg - mn) / scale).clamp_(0, qmax)
    deq = (q * scale + mn).view(out_f, in_f)
    rel = ((deq - w32).norm() / w32.norm().clamp_min(1e-9)).item()
    q = q.view(out_f, in_f).to(torch.uint8)
    packed = pack_int4(q) if bits == 4 else q
    return packed, scale.squeeze(-1).to(torch.bfloat16), mn.squeeze(-1).to(torch.bfloat16), rel


class QuantLinear(nn.Module):
    """存 int 权重 + group scale/zero，前向时还原成浮点做 matmul。

    端侧运行时会把还原+matmul 融合成整数核；这里在 GPU 上只能测到「权重占用下降」
    与「反量化开销」，两个数字都要如实记录。
    """

    def __init__(self, packed, scales, zeros, bias, bits, in_features, out_features,
                 group_size, rel_err):
        super().__init__()
        self.bits = bits
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.rel_err = float(rel_err)
        self.register_buffer("qweight", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone().to(torch.bfloat16), requires_grad=False)
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_linear(cls, lin: nn.Linear, bits: int, group_size: int):
        packed, scales, zeros, rel = quantize_weight(lin.weight.data, bits, group_size)
        dev = lin.weight.device
        return cls(packed.to(dev), scales.to(dev), zeros.to(dev), lin.bias, bits,
                   lin.in_features, lin.out_features, group_size, rel)

    def dequantize(self, dtype=torch.bfloat16) -> torch.Tensor:
        q = unpack_int4(self.qweight) if self.bits == 4 else self.qweight
        q = q.to(dtype)
        ng = self.scales.shape[1]
        q = q.view(self.out_features, ng, -1)
        w = q * self.scales.to(dtype).unsqueeze(-1) + self.zeros.to(dtype).unsqueeze(-1)
        return w.view(self.out_features, self.in_features)

    def forward(self, x):
        w = self.dequantize(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)

    def storage_bytes(self) -> int:
        return (self.qweight.numel() * self.qweight.element_size()
                + self.scales.numel() * self.scales.element_size()
                + self.zeros.numel() * self.zeros.element_size()
                + (self.bias.numel() * self.bias.element_size() if self.bias is not None else 0))


class QuantEmbedding(nn.Module):
    """按**行**量化词嵌入表：每个 token 向量一个 scale/zero。

    为什么要单独实现：`nn.Embedding` 的权重是 [vocab, hidden] = [151936, 2560]（bf16 0.72 GiB），
    若整表反量化再查表，每次前向都要展开 0.7 GiB；按行量化后只对**查到的那些行**反量化。
    """

    def __init__(self, qweight, scales, zeros, bits, num_embeddings, embedding_dim):
        super().__init__()
        self.bits = bits
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)

    @classmethod
    def from_embedding(cls, emb: nn.Embedding, bits: int):
        w = emb.weight.data
        # group_size=0 → 整行一组（每个 token 一个 scale/zero）
        packed, scales, zeros, rel = quantize_weight(w, bits, 0)
        # per-row 时 ng==1，quantize_weight 给的是 [V,1]；这里压成 [V]，
        # 否则 idx 查表后会多出一维、和 [*, hidden] 广播错位（实测踩过）。
        scales = scales.reshape(-1).contiguous()
        zeros = zeros.reshape(-1).contiguous()
        assert scales.dim() == 1, scales.shape
        dev = w.device
        obj = cls(packed.to(dev), scales.to(dev), zeros.to(dev), bits,
                  w.shape[0], w.shape[1])
        obj.rel_err = float(rel)
        return obj

    def forward(self, idx):
        if self.bits == 4:
            q = unpack_int4(self.qweight[idx])
        else:
            q = self.qweight[idx]
        s = self.scales[idx].unsqueeze(-1)
        z = self.zeros[idx].unsqueeze(-1)
        return q.to(s.dtype) * s + z

    def storage_bytes(self) -> int:
        return (self.qweight.numel() * self.qweight.element_size()
                + self.scales.numel() * self.scales.element_size()
                + self.zeros.numel() * self.zeros.element_size())


# --------------------------------------------------------------------------- #
# 整模型量化
# --------------------------------------------------------------------------- #
def quantize_model(model, bits: int, group_size: int, skip=("lm_head",), quant_embed=False):
    stats = []

    def walk(module: nn.Module, prefix: str):
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if any(s in full for s in skip):
                    continue
                ql = QuantLinear.from_linear(child, bits, group_size)
                setattr(module, name, ql)
                stats.append({"name": full, "kind": "linear", "shape": list(child.weight.shape),
                              "rel_err": round(ql.rel_err, 6),
                              "storage_bytes": ql.storage_bytes()})
            elif isinstance(child, nn.Embedding) and quant_embed:
                qe = QuantEmbedding.from_embedding(child, bits)
                setattr(module, name, qe)
                stats.append({"name": full, "kind": "embedding",
                              "shape": list(child.weight.shape),
                              "rel_err": round(qe.rel_err, 6),
                              "storage_bytes": qe.storage_bytes()})
            else:
                walk(child, full)

    walk(model, "")
    return stats


def plain_linear_bytes(model) -> int:
    """量化前所有普通 nn.Linear 的 bf16 体积（分母，必须在量化前调用）。"""
    return sum(m.weight.numel() * m.weight.element_size()
               for m in model.modules() if type(m) is nn.Linear)


def quant_report(model) -> dict:
    n_q = n_plain = n_emb = 0
    q_bytes = plain_bytes = emb_bytes = 0
    for m in model.modules():
        if isinstance(m, QuantLinear):
            n_q += 1
            q_bytes += m.storage_bytes()
        elif isinstance(m, QuantEmbedding):
            n_emb += 1
            q_bytes += m.storage_bytes()
        elif type(m) is nn.Linear:
            n_plain += 1
            plain_bytes += m.weight.numel() * m.weight.element_size()
        elif type(m) is nn.Embedding:
            emb_bytes += m.weight.numel() * m.weight.element_size()
    return {"n_quant_linear": n_q, "n_quant_embedding": n_emb, "n_plain_linear": n_plain,
            "quant_storage_bytes": q_bytes,
            "remaining_plain_linear_bytes": plain_bytes,
            "remaining_plain_embedding_bytes": emb_bytes}


def _partition(sd: dict, shard_bytes: int) -> list[dict]:
    """按张量字节数贪心分片（先大后小，放进当前最小的那片）。"""
    items = sorted(sd.items(), key=lambda kv: kv[1].numel() * kv[1].element_size(), reverse=True)
    shards: list[dict] = [{}]
    sizes = [0]
    for k, v in items:
        n = v.numel() * v.element_size()
        i = sizes.index(min(sizes))
        if sizes[i] and sizes[i] + n > shard_bytes:
            shards.append({})
            sizes.append(0)
            i = len(shards) - 1
        shards[i][k] = v
        sizes[i] += n
    return shards


def save_quantized(model, out_dir: Path, meta: dict, src_dir: Path = None,
                   shard_bytes: int | None = None) -> int:
    """存盘：完整 state_dict（量化层存 int 权重，其余保持 bf16）+ 自描述元数据 + 代码/配置。

    存整份 state_dict 而不是只存量化层，是为了让量化包**自带全部权重**；
    代码与 config 一并拷入，使目录本身就是一个完整的 HF 风格快照（量化不改变张量形状，
    所以用原始 config 建壳即可，见 load_quantized）。

    `shard_bytes` 给定时按大小切片（GitHub Release 单资产上限 2 GiB，超了就必须切），
    并写 `model_int.safetensors.index.json`；`load_quantized` 能吃这种布局。
    """
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}
    total = sum(v.numel() * v.element_size() for v in sd.values())

    if shard_bytes and total > shard_bytes:
        parts = _partition(sd, shard_bytes)
        n = len(parts)
        weight_map = {}
        for i, part in enumerate(parts, 1):
            name = f"model_int-{i:05d}-of-{n:05d}.safetensors"
            save_file(part, str(out_dir / name), metadata={"format": "pt"})
            for k in part:
                weight_map[k] = name
        C.save_json(out_dir / "model_int.safetensors.index.json",
                    {"metadata": {"total_size": total}, "weight_map": weight_map})
        (out_dir / "model_int.safetensors").unlink(missing_ok=True)  # 清掉上一次的单文件布局
        paths = [out_dir / f"model_int-{i:05d}-of-{n:05d}.safetensors" for i in range(1, n + 1)]
    else:
        paths = [out_dir / "model_int.safetensors"]
        save_file(sd, str(paths[0]), metadata={"format": "pt"})
        # 清掉上一次的分片布局，避免同一目录里两种布局并存
        (out_dir / "model_int.safetensors.index.json").unlink(missing_ok=True)
        for old in out_dir.glob("model_int-*-of-*.safetensors"):
            old.unlink(missing_ok=True)

    C.save_json(out_dir / "quant_config.json", meta)

    src = Path(src_dir) if src_dir else C.MODEL
    for p in src.iterdir():
        if not p.is_file() or p.suffix not in COPY_SUFFIXES:
            continue
        # 只排除权重与索引；**不能**用 startswith("model") —— 那会连
        # modeling_mage_vl.py / configuration 一起排掉（实测踩过）。
        if p.suffix == ".safetensors" or p.name in (
                "model.safetensors.index.json", "model_int.safetensors", "quant_config.json"):
            continue
        shutil.copy2(p, out_dir / p.name)
    return sum(p.stat().st_size for p in paths)


def _load_state_dict(ckpt: Path) -> dict:
    """读量化包的权重：支持单文件与分片（index.json）两种布局。"""
    from safetensors.torch import load_file

    idx = ckpt / "model_int.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
        sd = {}
        for shard in sorted(set(wm.values())):
            sd.update(load_file(str(ckpt / shard)))
        return sd
    return load_file(str(ckpt / "model_int.safetensors"))


def load_quantized(ckpt_dir, device: str = "cuda"):
    """从量化包**独立**构建可运行模型（不需要原始 bf16 权重）。

    思路：量化不改变任何张量形状 → 用包内 config 在 meta device 上建空壳（不占内存），
    再把量化层换成 QuantLinear（直接吃包里的 int 权重），其余权重用 assign=True 灌入。
    """
    import json as _json

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    ckpt = Path(ckpt_dir)
    meta = _json.loads((ckpt / "quant_config.json").read_text(encoding="utf-8"))
    bits, gs = int(meta["bits"]), int(meta["group_size"])
    cfg = AutoConfig.from_pretrained(str(ckpt), trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    sd = _load_state_dict(ckpt)

    n_swapped, consumed = 0, set()
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            key = f"{full}.qweight"
            if key not in sd:
                continue
            packed = sd[key]
            in_f = packed.shape[1] * (2 if bits == 4 else 1)
            scales, zeros = sd[f"{full}.scales"], sd[f"{full}.zeros"]
            if isinstance(child, nn.Embedding):
                new = QuantEmbedding(packed, scales, zeros, bits, packed.shape[0], in_f)
            else:
                # 视觉塔的 qkv/proj/fc1/fc2 是带 bias 的 Linear：bias 必须一起接过来，
                # 否则它会以"多余的 key"形式留在 rest 里，被 load_state_dict 判为 unexpected。
                bias = sd.get(f"{full}.bias")
                new = QuantLinear(packed, scales, zeros, bias, bits, in_f, packed.shape[0], gs, 0.0)
                if bias is not None:
                    consumed.add(f"{full}.bias")
            setattr(mod, cname, new)
            consumed.update({key, f"{full}.scales", f"{full}.zeros"})
            n_swapped += 1

    rest = {k: v for k, v in sd.items() if k not in consumed}
    missing, unexpected = model.load_state_dict(rest, strict=False, assign=True)
    missing = [k for k in missing if k not in consumed]
    if missing or unexpected:
        raise RuntimeError(f"量化包与模型结构不匹配：缺 {missing[:8]}，多 {unexpected[:8]}")
    model = model.to(device).eval()
    return model, {"n_swapped": n_swapped, "n_rest": len(rest), "bits": bits, "group_size": gs}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def verify_load(ckpt_dir: str, max_new_tokens: int = 48) -> dict:
    """独立加载量化包并跑两项推理，证明「产物能被重新打开」，不依赖原始 bf16 权重。"""
    from transformers import AutoProcessor

    buf = []
    ckpt = Path(ckpt_dir)
    processor = AutoProcessor.from_pretrained(str(ckpt), trust_remote_code=True)
    model, info = load_quantized(ckpt)
    mem = C.gpu_mem()
    C.log(f"[load] 独立加载成功 {info}；GPU 常驻 {mem['allocated_gb']}GB / "
          f"{mem['total_gb']}GB", buf)
    rec = {"ckpt": str(ckpt), "info": info, "mem_after_load": mem, "items": []}
    for it in [x for x in C.eval_set() if x["id"] in ("text00", "img00")]:
        img = C.load_item_image(it) if it["kind"] == "image" else None
        r = C.generate_measured(model, processor, it["prompt"], image=img,
                                kind=it["kind"], max_new_tokens=max_new_tokens)
        r["id"] = it["id"]
        rec["items"].append(r)
        C.log(f"[load] {it['id']} {r.get('gen_sec')}s {r.get('tok_per_s')}tok/s "
              f"peak={r.get('peak_allocated_gb')}GB {'ERR ' + r['error'] if r.get('error') else ''}", buf)
        C.log(f"       回答: {(r.get('answer') or '')[:200]}", buf)
    out = C.RESULTS / f"compress_loadcheck_{ckpt.name}.json"
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    C.save_json(out, rec)
    C.log(f"[load] -> {out}", buf)
    (C.WORK / f"loadcheck_{ckpt.name}.log").write_text("\n".join(buf) + "\n", encoding="utf-8")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--group-size", type=int, default=64, help="-1 = per-channel")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--include-lm-head", action="store_true",
                    help="默认跳过 lm_head（词表 151936，量化它省 0.58GB 但通常最伤精度）")
    ap.add_argument("--quant-embed", action="store_true",
                    help="同时量化 nn.Embedding（词嵌入表，bf16 0.72GiB；按行量化）")
    ap.add_argument("--shard-gib", type=float, default=None,
                    help="按该大小切片存盘（GitHub Release 单资产上限 2 GiB，建议 1.9）")
    ap.add_argument("--out", default=None,
                    help="证据文件名（默认 compress_quant_<tag>.json）。"
                         "⚠️ 用 --no-eval 重跑会覆盖同名文件、抹掉上一次的分项数据，要留档就换名")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--no-eval", action="store_true", help="只量化+存盘，不跑评测集")
    ap.add_argument("--model", default=None, help="模型目录；默认 E:\\MageVL\\Mage-VL")
    ap.add_argument("--load", default=None, help="只做独立加载验证：从该量化包构建模型并跑两项")
    args = ap.parse_args()

    if args.load:
        verify_load(args.load, args.max_new_tokens)
        return

    tag = args.tag or (f"int{args.bits}" + ("_pc" if args.group_size <= 0 else f"_g{args.group_size}"))
    skip = () if args.include_lm_head else ("lm_head",)
    out_dir = C.WORK / tag
    results_path = C.RESULTS / (args.out or f"compress_quant_{tag}.json")
    buf = []
    rec = {"tag": tag, "bits": args.bits, "group_size": args.group_size,
           "skip": list(skip), "quant_embed": bool(args.quant_embed),
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    from safetensors_total import weights_bytes
    model_dir = Path(args.model) if args.model else C.MODEL
    src_bytes = weights_bytes(model_dir)
    rec["source_weights_bytes"] = src_bytes
    rec["model"] = str(model_dir)

    processor = C.load_processor(model_dir)
    model, load_sec = C.load_model(model_dir=model_dir)
    C.log(f"[{tag}] bf16 加载 {load_sec:.1f}s, 权重常驻 {C.gpu_mem()['allocated_gb']}GB", buf)

    torch.cuda.reset_peak_memory_stats()
    orig_linear_bytes = plain_linear_bytes(model)
    t0 = time.time()
    stats = quantize_model(model, args.bits, args.group_size, skip=skip,
                           quant_embed=args.quant_embed)
    quant_sec = time.time() - t0
    torch.cuda.empty_cache()
    rep = quant_report(model)
    rec.update({"load_sec": round(load_sec, 1), "quant_sec": round(quant_sec, 1),
                "peak_during_quant_gb": C.gpu_mem()["peak_allocated_gb"],
                "orig_linear_bytes": orig_linear_bytes,
                "report": rep, "layers": stats})
    rel = [s["rel_err"] for s in stats]
    rec["rel_err"] = {"n": len(rel), "mean": round(sum(rel) / max(len(rel), 1), 6),
                      "max": round(max(rel), 6), "min": round(min(rel), 6)}
    C.log(f"[{tag}] 量化 {len(stats)} 层 / {quant_sec:.1f}s：线性层 "
          f"{orig_linear_bytes/1024**3:.2f}GiB -> {rep['quant_storage_bytes']/1024**3:.2f}GiB "
          f"(1/{orig_linear_bytes/max(rep['quant_storage_bytes'],1):.2f})；未量化的普通 Linear "
          f"{rep['n_plain_linear']} 个（{rep['remaining_plain_linear_bytes']/1024**3:.2f}GiB）", buf)
    C.log(f"[{tag}] 层相对误差 mean={rec['rel_err']['mean']} max={rec['rel_err']['max']}", buf)

    mem_after = C.gpu_mem()
    rec["mem_after_quant"] = mem_after
    C.log(f"[{tag}] 量化后 GPU 常驻 {mem_after['allocated_gb']}GB "
          f"(总显存 {mem_after['total_gb']}GB)", buf)

    saved = save_quantized(model, out_dir, {
        "bits": args.bits, "group_size": args.group_size, "skip": list(skip),
        "quant_embed": bool(args.quant_embed),
        "source": str(model_dir), "rel_err": rec["rel_err"],
    }, src_dir=model_dir,
        shard_bytes=int(args.shard_gib * 1024 ** 3) if args.shard_gib else None)
    rec["saved"] = {"dir": str(out_dir), "bytes": saved,
                    "gib": round(saved / 1024 ** 3, 3),
                    "ratio_vs_bf16": round(saved / src_bytes, 4)}
    C.log(f"[{tag}] 存盘 {saved/1024**3:.2f}GiB（bf16 权重 {src_bytes/1024**3:.2f}GiB，"
          f"比值 {rec['saved']['ratio_vs_bf16']}）-> {out_dir}", buf)

    if not args.no_eval:
        items = C.eval_set()
        out_items = []
        for it in items:
            img = C.load_item_image(it) if it["kind"] == "image" else None
            r = C.generate_measured(model, processor, it["prompt"], image=img,
                                    kind=it["kind"], max_new_tokens=args.max_new_tokens)
            r["id"] = it["id"]
            out_items.append(r)
            C.log(f"[{tag}] {it['id']:>7} {r.get('prompt_tokens')}->{r.get('new_tokens')} "
                  f"{r.get('gen_sec')}s {r.get('tok_per_s')}tok/s peak={r.get('peak_allocated_gb')}GB "
                  f"{('ERR ' + r['error']) if r.get('error') else ''}", buf)
        ok = [r for r in out_items if not r.get("error")]
        rec["items"] = out_items
        rec["summary"] = {
            "n_ok": len(ok), "n_err": len(out_items) - len(ok),
            "mean_tok_per_s": round(sum(r["tok_per_s"] for r in ok) / max(len(ok), 1), 3),
            "max_peak_allocated_gb": max([r["peak_allocated_gb"] for r in ok], default=None),
        }
        C.log(f"[{tag}] summary={rec['summary']}", buf)

    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    C.save_json(results_path, rec)
    C.log(f"[{tag}] -> {results_path}", buf)
    (C.WORK / f"quant_{tag}.log").write_text("\n".join(buf) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
