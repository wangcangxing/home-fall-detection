#!/usr/bin/env python
"""probe A（修正版）：降低像素预算后，frames 后端在 16GB 上能跑多少帧、精度损失多少。

接口依据（全部核实自源码，非猜测）：
  - processor.image_processor 是 Qwen2VLImageProcessor，**没有** min_pixels/max_pixels 属性，
    预算在 `size`（SizeDict）里：shortest_edge=min_pixels、longest_edge=max_pixels。
    实测 `ip.size["longest_edge"] = X` 可写且生效。
  - video_processing_mage_vl.py L507-512：传 list[PIL] 时 `_coerce_video_input` 直接返回，
    **不做** smart_resize；第一道 resize（extract_video_frames_to_pil）只在传路径时发生。
  - video_processing_mage_vl.py L626-627：第二道 `ip(images=frames_pil)` 对所有输入都生效
    → 对 PIL 列表输入，**只有 image_processor.size 是有效旋钮**。
  - 因此同时改 ip.size 与 vp.min/max_pixels，覆盖 PIL 列表与路径两种输入。
"""
import difflib
import time
from pathlib import Path

import torch

MODEL = r"E:\MageVL\Mage-VL"
VIDEO = rf"{MODEL}\examples\soccer-broadcast.mp4"
OUT = Path(r"E:\MageVL\baseline")

CASES = [
    (8, None, "8帧-出厂4000000"),
    (8, 150000, "8帧-150k"),
    (16, 150000, "16帧-150k"),
    (32, 150000, "32帧-150k"),
    (32, 64000, "32帧-64k"),
    (64, 64000, "64帧-64k"),
    (64, 150000, "64帧-150k"),
]


def sample_video(path, n):
    import cv2
    import numpy as np
    from PIL import Image
    cap = cv2.VideoCapture(path)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, fc - 1, min(n, fc), dtype=int)
    fs = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            fs.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return fs


def main():
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()

    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    ).eval()

    ip = processor.image_processor
    vp = processor.video_processor
    assert vp._image_processor is ip, "共享实例假设不成立，probe 需要改"
    orig_longest = ip.size["longest_edge"]
    orig_shortest = ip.size["shortest_edge"]
    orig_vp = (vp.min_pixels, vp.max_pixels)
    print(f"出厂: image_processor.size.longest_edge={orig_longest} shortest_edge={orig_shortest} | "
          f"video_processor.min/max={orig_vp}", flush=True)

    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Describe this video."}]}],
        tokenize=False, add_generation_prompt=True,
    )

    lines = []
    ref = None
    for n, budget, tag in CASES:
        if budget is None:
            ip.size["longest_edge"] = orig_longest
            vp.min_pixels, vp.max_pixels = orig_vp
        else:
            ip.size["longest_edge"] = budget
            vp.min_pixels, vp.max_pixels = orig_shortest, budget
        frames = sample_video(VIDEO, n)
        try:
            inputs = processor(text=[text], videos=[frames], return_tensors="pt", padding=True)
            inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
            L = int(inputs["input_ids"].shape[1])
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            dt = time.time() - t0
            ans = processor.tokenizer.decode(out[0, L:], skip_special_tokens=True).strip()
            pk = torch.cuda.max_memory_allocated() / 1024**3
            rsv = torch.cuda.max_memory_reserved() / 1024**3
            if ref is None:
                ref = ans
            sim = difflib.SequenceMatcher(None, ref, ans).ratio()
            line = (f"[{tag}] OK  实际帧数={len(frames)} prompt={L} ({L/len(frames):.0f}/帧) "
                    f"peak_alloc={pk:.2f}GB peak_res={rsv:.2f}GB gen={dt:.1f}s 相似度={sim:.3f}")
            line += f"\n    回答: {ans[:260]}"
        except torch.OutOfMemoryError as e:
            line = f"[{tag}] OOM  {str(e).splitlines()[0][:100]}"
        except Exception as e:
            line = f"[{tag}] ERR  {type(e).__name__}: {str(e)[:150]}"
        print(line, flush=True)
        lines.append(line)
        del inputs
        torch.cuda.empty_cache()

    ip.size["longest_edge"] = orig_longest
    vp.min_pixels, vp.max_pixels = orig_vp
    (OUT / "probe_maxpixels.txt").write_text(
        f"出厂 longest_edge={orig_longest} shortest_edge={orig_shortest} vp={orig_vp}\n\n"
        + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
