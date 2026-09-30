#!/usr/bin/env python
"""Mage-VL 在 16GB 卡上 frames 视频后端能跑多少帧。

只加载一次模型，依次测试不同 num_frames；每次 OOM 后清空缓存继续。
"""
import time
from pathlib import Path

import torch

MODEL = r"E:\MageVL\Mage-VL"
VIDEO = rf"{MODEL}\examples\soccer-broadcast.mp4"
OUT = Path(r"E:\MageVL\baseline")
OUT.mkdir(parents=True, exist_ok=True)


def sample_video(path, num_frames):
    import cv2
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(path)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, fc - 1, min(num_frames, fc), dtype=int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def main():
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    ).eval()
    print("loaded. weights_on_gpu_GB=", round(torch.cuda.memory_allocated() / 1024**3, 2), flush=True)

    lines = []
    for n in (2, 4, 8, 16, 32):
        frames = sample_video(VIDEO, n)
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Describe this video."}]}],
            tokenize=False, add_generation_prompt=True,
        )
        try:
            inputs = processor(text=[text], videos=[frames], return_tensors="pt", padding=True)
            inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
            L = int(inputs["input_ids"].shape[1])
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            dt = time.time() - t0
            ans = processor.tokenizer.decode(out[0, L:], skip_special_tokens=True).strip()
            pk = torch.cuda.max_memory_allocated() / 1024**3
            rsv = torch.cuda.max_memory_reserved() / 1024**3
            line = f"frames={n:>2} OK   prompt_tokens={L:>6} peak_alloc={pk:5.2f}GB peak_reserved={rsv:5.2f}GB gen={dt:5.1f}s | {ans[:70]}"
        except torch.OutOfMemoryError as e:
            msg = str(e).split("\n")[0]
            line = f"frames={n:>2} OOM  prompt_tokens=n/a  {msg[:110]}"
        except Exception as e:
            line = f"frames={n:>2} ERR  {type(e).__name__}: {str(e)[:110]}"
        print(line, flush=True)
        lines.append(line)
        del inputs
        torch.cuda.empty_cache()

    (OUT / "video_frame_limits.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
