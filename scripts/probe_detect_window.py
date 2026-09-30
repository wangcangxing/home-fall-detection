#!/usr/bin/env python
"""probe J：扫出「摔倒后可被检出的时间窗口」——用来定触发事件后该延迟多久抽帧。

真值：
  hr_fall_detection_1.mp4  摔倒 45-46s
  hr_fall_detection_3.mp4  摔倒 205-211s（连续 7 秒）
每 1 秒扫一帧，问「Is there a person lying on the ground in this image?」
输出 yes/no 序列，标出相对触发时刻（=标注起点）的延迟。
"""
import time
from pathlib import Path

import torch

MODEL = r"E:\MageVL\Mage-VL"
D = Path(r"E:\MageVL\eval\posture\train")
OUT = Path(r"E:\MageVL\baseline")
Q = "Is there a person lying on the ground in this image? Answer yes or no."
SCANS = [("hr_fall_detection_1.mp4", 42, 50, 45.0), ("hr_fall_detection_3.mp4", 200, 214, 205.0)]


def main():
    import cv2
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar(); hf_logging.set_verbosity_error()

    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype="auto", device_map="auto").eval()

    def ask(img):
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q}]}],
            tokenize=False, add_generation_prompt=True)
        ins = processor(text=[text], images=[img], return_tensors="pt")
        ins = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in ins.items()}
        with torch.inference_mode():
            out = model.generate(**ins, max_new_tokens=8, do_sample=False)
        a = processor.tokenizer.decode(out[0, ins["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return a.lower().startswith("yes")

    lines = ["=== probe J：摔倒后可检出窗口（逐秒扫描）==="]
    for name, t0, t1, onset in SCANS:
        vid = D / name
        cap = cv2.VideoCapture(str(vid))
        seq = []
        for t in range(t0, t1 + 1):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            if not ok:
                continue
            got = ask(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            seq.append((t, got, t - onset))
        cap.release()
        lines.append(f"\n--- {name}  标注摔倒起点={onset:.0f}s ---")
        lines.append("  " + " ".join(f"{t}s:{'Y' if g else 'n'}" for t, g, _ in seq))
        lines.append("  相对触发延迟: " + " ".join(f"+{d:.0f}s:{'Y' if g else 'n'}" for _, g, d in seq))
        hits = [(t, d) for t, g, d in seq if g]
        if hits:
            ds = [d for _, d in hits]
            lines.append(f"  可检出点 {len(hits)} 个，延迟范围 {min(ds):.0f}~{max(ds):.0f}s"
                         f"（中位 {sorted(ds)[len(ds)//2]:.0f}s）")
        else:
            lines.append("  逐秒扫描未检出任何可检出点")
    (OUT / "probe_detect_window.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
