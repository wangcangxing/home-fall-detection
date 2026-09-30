#!/usr/bin/env python
"""Mage-VL 可行性基线：图像 + frames 视频，并测峰值显存。

严格复刻官方 inference.py 的调用路径（E:\\MageVL\\Mage-VL\\inference.py）：
  AutoProcessor.from_pretrained(path, trust_remote_code=True)
  AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                       torch_dtype="auto", device_map="auto").eval()
  model.generate(**inputs, max_new_tokens=..., do_sample=False)
"""
import sys
import time
from pathlib import Path

import torch

MODEL = r"E:\MageVL\Mage-VL"
OUT = Path(r"E:\MageVL\baseline")
OUT.mkdir(parents=True, exist_ok=True)


def peak(tag):
    alloc = torch.cuda.max_memory_allocated() / 1024**3
    reserv = torch.cuda.max_memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    line = f"[{tag}] peak_allocated={alloc:.2f}GB peak_reserved={reserv:.2f}GB total={total:.2f}GB"
    print(line, flush=True)
    return line


def run_case(model, processor, messages, media_kwargs, question, max_new_tokens, tag):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", **media_kwargs)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    dt = time.time() - t0
    answer = processor.tokenizer.decode(
        output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    n_new = output.shape[1] - inputs["input_ids"].shape[1]
    n_tok = int(inputs["input_ids"].shape[1])
    print(f"\n===== {tag} =====")
    print(f"prompt_tokens={n_tok} new_tokens={n_new} gen_sec={dt:.1f} tok/s={n_new/max(dt,1e-9):.2f}")
    print(answer)
    print(peak(tag))
    return answer


def main():
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    ).eval()
    print(f"load_sec={time.time()-t0:.1f}  dtype={model.dtype}  device={model.device}")
    print("weights_on_gpu_GB=", round(torch.cuda.memory_allocated() / 1024**3, 2))

    # 1) 图像
    q1 = "Describe this image in detail."
    run_case(model, processor,
             [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q1}]}],
             {"images": [Image.open(rf"{MODEL}\examples\dog.jpg").convert("RGB")]},
             q1, 256, "image")

    # 2) 视频 frames 后端（复刻官方 sample_video）
    import cv2
    import numpy as np

    def sample_video(path, num_frames):
        cap = cv2.VideoCapture(path)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = np.linspace(0, fc - 1, min(num_frames, fc), dtype=int)
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames

    vid = rf"{MODEL}\examples\soccer-broadcast.mp4"
    frames = sample_video(vid, 32)
    print(f"\nsampled {len(frames)} frames from soccer-broadcast.mp4")
    q2 = "Describe this video."
    run_case(model, processor,
             [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": q2}]}],
             {"videos": [frames]},
             q2, 256, "video-frames-32")

    with open(OUT / "vram_summary.txt", "w", encoding="utf-8") as f:
        f.write(peak("after-all") + "\n")


if __name__ == "__main__":
    sys.exit(main())
