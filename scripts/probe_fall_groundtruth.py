#!/usr/bin/env python
"""probe E：带真值的时序测试。

真值来自 pat2echo/fall-detection-posture-classification 的人工标注
（train/labels/hr_fall_detection_1.csv，48 秒片段）：
  0-44s  = None / Stand / Sit（含 34-36s 的 Sit 难负样本）
  45-46s = Stand-Lie (Fall)   ← 唯一真摔倒
  47-48s = 爬起来
问法：逐帧问「Is there a person lying on the ground in this image?」
"""
import re
import time
from pathlib import Path

import torch

MODEL = r"E:\MageVL\Mage-VL"
VIDEO = Path(r"E:\MageVL\eval\posture\train\hr_fall_detection_1.mp4")
OUT = Path(r"E:\MageVL\baseline")
Q_LIE = "Is there a person lying on the ground in this image? Answer yes or no."
Q_PER = "Is there a person in this image? Answer yes or no."

# (秒, 真值说明, 是否预期躺倒)
GRID = [
    (10.0, "Stand", False), (34.5, "Sit(难负样本)", False), (38.0, "Stand", False),
    (43.0, "None", False), (44.5, "None→fall前一刻", False),
    (45.2, "**Fall 开始**", True), (45.7, "**Fall 中**", True), (46.2, "**Fall 末/躺地**", True),
    (46.8, "起身", True), (47.5, "Stand", False),
]


def yes(a):
    a = a.strip().lower()
    if re.match(r"^(yes|y)\b", a): return True
    if re.match(r"^(no|n)\b", a): return False
    return None


def main():
    import cv2
    import numpy as np
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar(); hf_logging.set_verbosity_error()

    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype="auto", device_map="auto").eval()
    print("模型已加载", flush=True)

    def ask(img, q):
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
            tokenize=False, add_generation_prompt=True)
        ins = processor(text=[text], images=[img], return_tensors="pt")
        ins = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in ins.items()}
        with torch.inference_mode():
            out = model.generate(**ins, max_new_tokens=8, do_sample=False)
        return processor.tokenizer.decode(out[0, ins["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lines = [f"视频 {VIDEO.name}: fps={fps:.2f} frames={n} dur={n/fps:.2f}s"]
    lines.append(f"{'时间':>6} {'真值':<20} {'预期':<5} {'躺倒?':<6} {'有人?':<6} 理由")
    ok_cnt = tot = 0
    raw = []
    for t, desc, expect in GRID:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            lines.append(f"{t:>6.1f} {desc:<20} 读帧失败"); continue
        img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        al = yes(ask(img, Q_LIE)); ap = yes(ask(img, Q_PER))
        raw.append(f"  t={t:>4.1f}s lie={al} per={ap}")
        if al is not None:
            tot += 1; ok_cnt += (al == expect)
        lines.append(f"{t:>6.1f} {desc:<20} {str(expect):<5} {str(al):<6} {str(ap):<6}")
    cap.release()

    lines.append(f"\n逐帧时序一致率（10 个时间点）= {ok_cnt}/{tot}")
    lines.append("注：45-48s 是 1 秒级的短暂事件，逐帧抽查只能近似；真值本身也标注为 1 秒粒度。")

    # 视频级：含摔倒的窗口 vs 正常窗口
    ip, vp = processor.image_processor, processor.video_processor
    ole, omin = ip.size["longest_edge"], vp.min_pixels
    ip.size["longest_edge"] = 150000; vp.min_pixels, vp.max_pixels = 3136, 150000

    def vid_window(t0, t1, nf=16):
        c = cv2.VideoCapture(str(VIDEO))
        f0, f1 = int(t0 * fps), int(t1 * fps)
        idx = np.linspace(f0, f1, nf, dtype=int)
        fr = []
        for i in idx:
            c.set(cv2.CAP_PROP_POS_FRAMES, int(i)); o, f = c.read()
            if o: fr.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        c.release(); return fr

    for t0, t1, tag in ((0, 8, "0-8s 无摔倒"), (37, 44, "37-44s 无摔倒"), (41, 48, "41-48s 含45-46s摔倒")):
        frames = vid_window(t0, t1)
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text",
              "text": "Is there a person falling or lying on the ground in this video? Answer yes or no, then describe briefly."}]}],
            tokenize=False, add_generation_prompt=True)
        ins = processor(text=[text], videos=[frames], return_tensors="pt", padding=True)
        ins = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in ins.items()}
        if "pixel_values" in ins: ins["pixel_values"] = ins["pixel_values"].to(model.dtype)
        t0s = time.time()
        with torch.inference_mode():
            out = model.generate(**ins, max_new_tokens=70, do_sample=False)
        ans = processor.tokenizer.decode(out[0, ins["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        lines.append(f"[VIDEO {tag}] {len(frames)}帧 {time.time()-t0s:.1f}s -> {ans[:280]}")
    ip.size["longest_edge"] = ole

    (OUT / "probe_fall_groundtruth.txt").write_text("\n".join(lines) + "\n" + "\n".join(raw) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
