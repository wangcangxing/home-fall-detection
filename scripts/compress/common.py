#!/usr/bin/env python
"""Mage-VL 端侧压缩工具链：公共加载 / 输入构造 / 测量组件。

严格沿用仓库既有探针的模型调用路径（见 scripts/baseline_run.py）：
    AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                         torch_dtype="auto", device_map="auto")

约定：
  - 模型快照      E:\\MageVL\\Mage-VL
  - 重资产输出    E:\\MageVL\\compress\\            （量化/剪枝产物，不入库）
  - 测量证据      <repo>\\results\\compress_*.json  （入库）
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = Path(os.environ.get("MAGEVL_MODEL", r"E:\MageVL\Mage-VL"))
WORK = Path(os.environ.get("MAGEVL_WORK", r"E:\MageVL\compress"))
RESULTS = REPO / "results"
WORK.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 测量工具
# --------------------------------------------------------------------------- #
def gb(n: float) -> float:
    return round(n / 1024 ** 3, 3)


def gpu_mem() -> dict:
    import torch

    return {
        "allocated_gb": gb(torch.cuda.memory_allocated()),
        "reserved_gb": gb(torch.cuda.memory_reserved()),
        "peak_allocated_gb": gb(torch.cuda.max_memory_allocated()),
        "peak_reserved_gb": gb(torch.cuda.max_memory_reserved()),
        "total_gb": gb(torch.cuda.get_device_properties(0).total_memory),
    }


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def log(msg: str, buf: list | None = None) -> None:
    print(msg, flush=True)
    if buf is not None:
        buf.append(msg)


# --------------------------------------------------------------------------- #
# 模型 / 处理器
# --------------------------------------------------------------------------- #
def load_processor(model_dir=None):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(model_dir or MODEL), trust_remote_code=True)


def load_model(quantization_config=None, dtype=None, device_map="auto", model_dir=None):
    """返回 (model, load_sec)。quantization_config 为 None 时按 bf16 加载。"""
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()

    kwargs = {"trust_remote_code": True, "device_map": device_map}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["torch_dtype"] = dtype or torch.bfloat16
    else:
        kwargs["torch_dtype"] = dtype or "auto"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(str(model_dir or MODEL), **kwargs).eval()
    return model, time.time() - t0


def set_image_budget(processor, max_pixels: int | None):
    """唯一的有效像素旋钮：processor.image_processor.size['longest_edge']。

    已核实（见留痕坑点 #18/#23）：Qwen2VLImageProcessor 没有 max_pixels 属性，
    预算存在 size 这个 SizeDict 里；video_preprocessor_config.json 的 max_pixels
    在 frames 路径上不会被读。
    """
    if max_pixels is None:
        return
    size = processor.image_processor.size
    if hasattr(size, "longest_edge"):
        size.longest_edge = int(max_pixels)
    else:
        processor.image_processor.size = {"longest_edge": int(max_pixels)}


# --------------------------------------------------------------------------- #
# 输入构造（文本 / 图像）
# --------------------------------------------------------------------------- #
def build_inputs(processor, model, prompt: str, image=None, kind: str = "text"):
    from PIL import Image as PILImage

    if kind == "image":
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
        # image 既可能是路径，也可能已经是 PIL.Image（frame_at / load_item_image 返回的）
        img = image if isinstance(image, PILImage.Image) else PILImage.open(image)
        media = {"images": [img.convert("RGB")]}
    else:
        content = [{"type": "text", "text": prompt}]
        media = {}
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", **media)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    return inputs


def generate_measured(model, processor, prompt: str, image=None, kind: str = "text",
                      max_new_tokens: int = 64):
    """贪心生成一次并测量；返回结果字典（异常时 answer=None + error）。"""
    import torch

    rec = {"kind": kind, "prompt": prompt, "image": str(image) if image else None,
           "max_new_tokens": max_new_tokens}
    try:
        inputs = build_inputs(processor, model, prompt, image, kind)
        n_prompt = int(inputs["input_ids"].shape[1])
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        dt = time.time() - t0
        n_new = int(out.shape[1] - n_prompt)
        answer = processor.tokenizer.decode(out[0, n_prompt:], skip_special_tokens=True).strip()
        rec.update({
            "prompt_tokens": n_prompt,
            "new_tokens": n_new,
            "gen_sec": round(dt, 3),
            "tok_per_s": round(n_new / max(dt, 1e-9), 3),
            "peak_allocated_gb": gpu_mem()["peak_allocated_gb"],
            "answer": answer,
        })
    except torch.OutOfMemoryError as e:
        rec.update({"error": f"OOM: {str(e).splitlines()[0][:200]}", "answer": None})
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001 - 探针需要把失败也记成数据
        rec.update({"error": f"{type(e).__name__}: {str(e)[:300]}", "answer": None})
        torch.cuda.empty_cache()
    return rec


# --------------------------------------------------------------------------- #
# 固定评测集（写死在代码里 = 可复现；不依赖任何外部数据集）
# --------------------------------------------------------------------------- #
TEXT_PROMPTS = [
    "What is the capital of France? Answer with the city name only.",
    "List five common household objects, one per line.",
    "Explain in three sentences why the sky is blue.",
    "What is 17 * 23? Answer with the number only.",
    "Write a short paragraph describing a rainy afternoon.",
    "Name three differences between a cat and a dog.",
    "In one sentence, explain what model quantization means in machine learning.",
    "Translate 'good morning' into Chinese, Japanese, and French.",
]

# 图像项：dog.jpg + 从本地示例视频按时间点抽帧（不下载任何数据）
IMAGE_PROMPTS = [
    ("dog.jpg", 0.0, "Describe this image in detail."),
    ("video", 0.5, "Describe what is happening in this image in one sentence."),
    ("video", 1.5, "Describe what is happening in this image in one sentence."),
    ("video", 3.0, "Describe what is happening in this image in one sentence."),
    ("video", 5.0, "Describe what is happening in this image in one sentence."),
    ("video", 7.0, "Describe what is happening in this image in one sentence."),
    ("video", 9.0, "Describe what is happening in this image in one sentence."),
    ("video", 11.0, "Describe what is happening in this image in one sentence."),
]

VIDEO = MODEL / "examples" / "soccer-broadcast.mp4"
DOG = MODEL / "examples" / "dog.jpg"


def frame_at(seconds: float):
    """按时间点抽一帧（对示例视频足够；不追求与既有探针的抽帧口径一致）。"""
    import cv2
    from PIL import Image as PILImage

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    return PILImage.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))


def eval_set():
    """返回固定评测项列表：{id, kind, prompt, image_path|seconds}。"""
    items = []
    for i, p in enumerate(TEXT_PROMPTS):
        items.append({"id": f"text{i:02d}", "kind": "text", "prompt": p})
    for i, (src, sec, p) in enumerate(IMAGE_PROMPTS):
        if src == "dog.jpg":
            items.append({"id": f"img{i:02d}", "kind": "image", "prompt": p,
                          "image_path": str(DOG), "source": "dog.jpg"})
        else:
            items.append({"id": f"img{i:02d}", "kind": "image", "prompt": p,
                          "seconds": sec, "source": "soccer-broadcast.mp4"})
    return items


def load_item_image(item):
    from PIL import Image as PILImage

    p = item.get("image_path")
    if p:
        return PILImage.open(p).convert("RGB")
    return frame_at(item["seconds"])
