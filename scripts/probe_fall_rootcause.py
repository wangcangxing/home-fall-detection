#!/usr/bin/env python
"""probe F：漏检根因定位——是「帧数不够」还是「问法问的是状态」？

真值：hr_fall_detection_1.mp4 的 45-46s 是唯一摔倒事件（人工标注）。
对照：
  Q_state = "Is there a person lying on the ground in this video?"（问状态）
  Q_event = "Did a person fall down in this video?"（问事件）
窗口与帧率：
  41-48s / 16帧 ≈ 2.3fps（已实测漏检）
  44-47s / 16帧 ≈ 5.3fps
  44-47s / 32帧 ≈ 10.7fps
  45-47s / 16帧 ≈ 8fps
  0-8s   / 16帧（负对照）
"""
import time
from pathlib import Path

import torch

MODEL = r"E:\MageVL\Mage-VL"
VIDEO = Path(r"E:\MageVL\eval\posture\train\hr_fall_detection_1.mp4")
OUT = Path(r"E:\MageVL\baseline")
Q_STATE = "Is there a person lying on the ground in this video? Answer yes or no."
Q_EVENT = "Did a person fall down in this video? Answer yes or no."
CASES = [
    (41, 48, 16, Q_STATE, "已知漏检的基线"),
    (41, 48, 16, Q_EVENT, "同窗口换成事件问法"),
    (44, 47, 16, Q_EVENT, "5.3fps + 事件问法"),
    (44, 47, 32, Q_EVENT, "10.7fps + 事件问法"),
    (45, 47, 16, Q_EVENT, "8fps + 事件问法"),
    (44.5, 46.5, 16, Q_EVENT, "8fps 紧贴事件"),
    (0, 8, 16, Q_EVENT, "负对照（应为 no）"),
]


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
    ip, vp = processor.image_processor, processor.video_processor
    ip.size["longest_edge"] = 150000
    vp.min_pixels, vp.max_pixels = 3136, 150000
    print("模型已加载", flush=True)

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    lines = []
    for t0, t1, nf, q, tag in CASES:
        idx = np.linspace(int(t0 * fps), int(t1 * fps), nf, dtype=int)
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
            if ok: frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": q}]}],
            tokenize=False, add_generation_prompt=True)
        ins = processor(text=[text], videos=[frames], return_tensors="pt", padding=True)
        ins = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in ins.items()}
        if "pixel_values" in ins: ins["pixel_values"] = ins["pixel_values"].to(model.dtype)
        t = time.time()
        with torch.inference_mode():
            out = model.generate(**ins, max_new_tokens=40, do_sample=False)
        ans = processor.tokenizer.decode(out[0, ins["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        rate = nf / (t1 - t0)
        line = f"[{t0}-{t1}s {nf}帧 {rate:.1f}fps] {'事件' if q == Q_EVENT else '状态'}问法 | {tag:<22} -> {ans[:120]}"
        print(line, flush=True)
        lines.append(line)
    cap.release()
    lines.append("\n真值：45-46s 有一次摔倒；其余窗口无摔倒。含事件的窗口若答 no 即为漏检。")
    (OUT / "probe_fall_rootcause.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
