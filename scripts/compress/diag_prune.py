#!/usr/bin/env python
"""诊断：为什么剪枝后（内存内 / 回读包）推理会 IndexError / 输出乱码。

只做只读诊断，不改文件。输出 = 真实 traceback + 各层 config 副本的实际取值 +
lm_head 与 embed_tokens 是否被绑成同一张量。

用法：
  python scripts/compress/diag_prune.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import prune_structured as P  # noqa: E402


def dump_cfg(tag, model):
    lm = model.model.language_model
    vis = model.model.visual
    print(f"  [{tag}] MageVLConfig.text_config.num_hidden_layers = {model.config.text_config.num_hidden_layers}")
    print(f"  [{tag}] MageVLConfig.text_config.intermediate_size  = {model.config.text_config.intermediate_size}")
    print(f"  [{tag}] Qwen3Model.config.num_hidden_layers         = {lm.config.num_hidden_layers}")
    print(f"  [{tag}] Qwen3Model.config.intermediate_size          = {lm.config.intermediate_size}")
    print(f"  [{tag}] Qwen3Model.layers 长度                       = {len(lm.layers)}")
    print(f"  [{tag}] Qwen3Model.config.layer_types 长度           = {len(getattr(lm.config, 'layer_types', []) or [])}")
    print(f"  [{tag}] vision.config.num_hidden_layers              = {vis.config.num_hidden_layers}")
    print(f"  [{tag}] vision.encoder.layers 长度                   = {len(vis.encoder.layers)}")
    print(f"  [{tag}] rope_scaling/max_pos                         = "
          f"{getattr(lm.config, 'rope_scaling', None)} / {lm.config.max_position_embeddings}")


def try_gen(tag, model, processor, prompt, image=None, kind="text", n=16):
    inp = C.build_inputs(processor, model, prompt, image, kind)
    n_prompt = int(inp["input_ids"].shape[1])
    try:
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=n, do_sample=False)
        ans = processor.tokenizer.decode(out[0, n_prompt:], skip_special_tokens=True)
        print(f"  [{tag}] 生成成功: {ans[:200]!r}")
        return ans
    except Exception:
        print(f"  [{tag}] 生成失败，真实堆栈：")
        traceback.print_exc()
        return None


def main():
    processor = C.load_processor()

    print("=== A. 内存内剪枝后 ===")
    m, _ = C.load_model()
    P.drop_decoder_layers(m, 6, "uniform")
    P.prune_ffn(m, 0.75)
    dump_cfg("A", m)
    try_gen("A-text", m, processor, "What is the capital of France?")
    try_gen("A-image", m, processor, "Describe this image in detail.", image=C.DOG, kind="image")
    del m
    torch.cuda.empty_cache()

    print()
    print("=== B. 回读剪枝包 ===")
    m2, _ = C.load_model(model_dir=r"E:\MageVL\compress\pruneL30_ffn75")
    dump_cfg("B", m2)
    print(f"  [B] lm_head 与 embed_tokens 同一张量? "
          f"{m2.lm_head.weight.data_ptr() == m2.model.language_model.embed_tokens.weight.data_ptr()}")
    try_gen("B-text", m2, processor, "What is the capital of France?")
    try_gen("B-image", m2, processor, "Describe this image in detail.", image=C.DOG, kind="image")


if __name__ == "__main__":
    main()
