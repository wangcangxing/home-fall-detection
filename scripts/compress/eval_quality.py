#!/usr/bin/env python
"""量化前后质量对比：固定评测集上的 KL 散度 / top-1 一致率 / 生成答案一致率。

做法（单进程两阶段，避免同时驻留两份 8.8GB 权重）：
  1. 加载 bf16 → 在每项的 **固定续写** 上做一次 teacher-forcing 前向，收集 logits（CPU, fp16）
     并贪心生成一次答案；
  2. 释放模型 → 重新加载 bf16 并施加与线上一致的量化 → 重复同一过程；
  3. 逐位置算 KL(bf16 || 量化)、top-1 是否相同，以及生成答案的完全一致率/相似度。

固定续写 = 同一段参考文本，所有变体都吃同样的 token，因此 KL 反映的是"量化把模型改了
多少"，不是"模型聪不聪明"。这条要在报告里写清楚，不能当能力分数用。

用法：
  python scripts/compress/eval_quality.py --tag int4g64 --bits 4 --group-size 64
"""
from __future__ import annotations

import argparse
import difflib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import quant_rtn as Q  # noqa: E402

CONTINUATION = (" This description is written in plain English. It lists the main objects that can "
                "be seen in the picture, together with their colours, their approximate positions, "
                "and any obvious action taking place. The tone is factual, no numbers are invented, "
                "and uncertain details are left out rather than guessed.")
N_CONT = 48


def collect(model, processor, items, max_new_tokens: int):
    """返回 (logits_list, answers)；logits 为 [n_cont, V] fp16 CPU。"""
    tok = processor.tokenizer
    cont_ids = tok(CONTINUATION, return_tensors="pt", add_special_tokens=False).input_ids
    cont_ids = cont_ids[:, :N_CONT].to(model.device)
    logits_list, answers = [], []
    for it in items:
        img = C.load_item_image(it) if it["kind"] == "image" else None
        inputs = C.build_inputs(processor, model, it["prompt"], img, it["kind"])
        n_prompt = int(inputs["input_ids"].shape[1])
        input_ids = torch.cat([inputs["input_ids"], cont_ids], dim=1)
        kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}
        # 必须同步延长 attention_mask：自定义建模代码用
        # position_ids = attention_mask.cumsum(-1) - 1（modeling_mage_vl.py L1269-1274），
        # mask 短于 input_ids 时 rotary 的 cos/sin 会对不上（实测报过 58 vs 33）。
        am = inputs.get("attention_mask")
        if am is not None:
            kwargs["attention_mask"] = torch.cat(
                [am, torch.ones((am.shape[0], cont_ids.shape[1]), dtype=am.dtype, device=am.device)],
                dim=1)
        assert kwargs.get("attention_mask", input_ids).shape[1] == input_ids.shape[1]
        with torch.inference_mode():
            out = model(input_ids=input_ids, **kwargs)
        n = int(cont_ids.shape[1])
        lg = out.logits[0, n_prompt - 1: n_prompt - 1 + n, :].to(torch.float16).cpu()
        logits_list.append(lg)
        del out
        torch.cuda.empty_cache()
        r = C.generate_measured(model, processor, it["prompt"], image=img,
                                kind=it["kind"], max_new_tokens=max_new_tokens)
        answers.append(r.get("answer") or "")
        print(f"    collected {it['id']} prompt={n_prompt} cont={n} "
              f"ans={len(answers[-1])}ch", flush=True)
    return logits_list, answers


def compare(base_logits, q_logits, base_ans, q_ans):
    import torch.nn.functional as F

    rows, kls, hits = [], [], []
    for i, (b, q) in enumerate(zip(base_logits, q_logits)):
        b64, q64 = b.float(), q.float()
        kl = F.kl_div(F.log_softmax(q64, -1), F.log_softmax(b64, -1),
                      reduction="batchmean", log_target=True).item()
        top1 = (b64.argmax(-1) == q64.argmax(-1)).float().mean().item()
        ans_ratio = difflib.SequenceMatcher(None, base_ans[i], q_ans[i]).ratio()
        rows.append({"i": i, "kl": round(kl, 5), "top1_agree": round(top1, 4),
                     "answer_exact": base_ans[i] == q_ans[i],
                     "answer_ratio": round(ans_ratio, 4)})
        kls.append(kl)
        hits.append(top1)
    return {
        "n": len(rows),
        "mean_kl": round(sum(kls) / max(len(kls), 1), 5),
        "max_kl": round(max(kls), 5) if kls else None,
        "mean_top1_agree": round(sum(hits) / max(len(hits), 1), 4),
        "answer_exact_rate": round(sum(r["answer_exact"] for r in rows) / max(len(rows), 1), 4),
        "mean_answer_ratio": round(sum(r["answer_ratio"] for r in rows) / max(len(rows), 1), 4),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--include-lm-head", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--model", default=None, help="基线模型目录；默认 E:\\MageVL\\Mage-VL")
    ap.add_argument("--quant-model", default=None,
                    help="被量化的模型目录（可与基线不同：例如「原模型 vs 剪枝后再量化」做端到端对比）")
    ap.add_argument("--quant-embed", action="store_true", help="同时量化 nn.Embedding")
    args = ap.parse_args()

    quant_model_dir = args.quant_model or args.model
    results_path = C.RESULTS / f"compress_quality_{args.tag}.json"
    buf = []
    items = C.eval_set()
    processor = C.load_processor(args.model)
    C.set_image_budget(processor, args.max_pixels)

    C.log(f"[{args.tag}] 阶段 1/2：基线 {args.model or C.MODEL}", buf)
    model, sec = C.load_model(model_dir=args.model)
    C.log(f"[{args.tag}] bf16 加载 {sec:.1f}s", buf)
    base_logits, base_ans = collect(model, processor, items, args.max_new_tokens)
    del model
    torch.cuda.empty_cache()

    C.log(f"[{args.tag}] 阶段 2/2：int{args.bits} 量化（目标 {quant_model_dir or C.MODEL}）", buf)
    skip = () if args.include_lm_head else ("lm_head",)
    model, sec = C.load_model(model_dir=quant_model_dir)
    t0 = time.time()
    stats = Q.quantize_model(model, args.bits, args.group_size, skip=skip,
                             quant_embed=args.quant_embed)
    torch.cuda.empty_cache()
    C.log(f"[{args.tag}] 量化 {len(stats)} 层 / {time.time()-t0:.1f}s，"
          f"GPU 常驻 {C.gpu_mem()['allocated_gb']}GB", buf)
    storage = sum(s["storage_bytes"] for s in stats)
    n_embed = sum(1 for s in stats if s.get("kind") == "embedding")
    C.log(f"[{args.tag}] 量化后权重 {storage/1024**3:.3f} GiB"
          f"（{len(stats)} 层，其中词嵌入 {n_embed} 个；未量化的 lm_head/其余按 bf16 另计）", buf)
    q_logits, q_ans = collect(model, processor, items, args.max_new_tokens)

    cmp_ = compare(base_logits, q_logits, base_ans, q_ans)
    rec = {"tag": args.tag, "bits": args.bits, "group_size": args.group_size,
           "skip": list(skip), "n_cont": N_CONT, "continuation": CONTINUATION,
           "base_model": str(args.model or C.MODEL), "quant_model": str(quant_model_dir or C.MODEL),
           "quant_embed": bool(args.quant_embed),
           "quant_layers": len(stats),
           "quant_storage_bytes": storage,
           "items": [it["id"] for it in items],
           "comparison": cmp_,
           "base_answers": base_ans, "quant_answers": q_ans,
           "rel_err_mean": round(sum(s["rel_err"] for s in stats) / max(len(stats), 1), 6),
           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    C.save_json(results_path, rec)
    C.log(f"[{args.tag}] mean_KL={cmp_['mean_kl']} top1_agree={cmp_['mean_top1_agree']} "
          f"答案完全一致率={cmp_['answer_exact_rate']} 相似度={cmp_['mean_answer_ratio']}", buf)
    for r, it in zip(cmp_["rows"], items):
        C.log(f"    {it['id']:>7} KL={r['kl']:.5f} top1={r['top1_agree']:.3f} "
              f"ans_exact={r['answer_exact']} ratio={r['answer_ratio']:.3f}", buf)
    C.log(f"[{args.tag}] -> {results_path}", buf)
    (C.WORK / f"quality_{args.tag}.log").write_text("\n".join(buf) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
