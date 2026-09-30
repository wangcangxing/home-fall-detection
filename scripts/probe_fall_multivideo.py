#!/usr/bin/env python
"""probe G：多视频交叉验证——摔倒漏检是「片段特例（画质/距离）」还是「普遍问题」？

真值（数据集人工标注）：
  hr_fall_detection_2.mp4 (206.8s): 摔倒 @ ~28-29s, ~136s, ~163-165s
  hr_fall_detection_3.mp4 (394.3s): 摔倒 @ ~204-211s, ~219-220s, ~383-384s
同一视频内取"无摔倒"窗口作配对对照，排除画面风格差异。

另有：用 OpenCV HOG 行人检测器定量估计「人占画面多大」（客观代理，不依赖 VLM）。
"""
import time
from pathlib import Path

import torch

MODEL = r"E:\MageVL\Mage-VL"
D = Path(r"E:\MageVL\eval\posture\train")
OUT = Path(r"E:\MageVL\baseline")
Q = "Is there a person falling or lying on the ground in this video? Answer yes or no."

CASES = [
    ("hr_fall_detection_2.mp4", 25, 31, True, "摔倒@28-29s"),
    ("hr_fall_detection_2.mp4", 133, 139, True, "摔倒@136s"),
    ("hr_fall_detection_2.mp4", 160, 166, True, "摔倒@163-165s"),
    ("hr_fall_detection_2.mp4", 5, 11, False, "对照 无摔倒"),
    ("hr_fall_detection_2.mp4", 60, 66, False, "对照 无摔倒"),
    ("hr_fall_detection_2.mp4", 100, 106, False, "对照 无摔倒"),
    ("hr_fall_detection_3.mp4", 201, 212, True, "摔倒@204-211s(长达7s)"),
    ("hr_fall_detection_3.mp4", 216, 222, True, "摔倒@219-220s"),
    ("hr_fall_detection_3.mp4", 380, 386, True, "摔倒@383-384s"),
    ("hr_fall_detection_3.mp4", 20, 26, False, "对照 无摔倒"),
    ("hr_fall_detection_3.mp4", 300, 306, False, "对照 无摔倒"),
]
NF = 16


def hog_size(path, ts):
    """用 HOG 行人检测器估计人在画面里占的宽度比例；返回 (检测帧数, 最大宽度占比)"""
    import cv2
    cap = cv2.VideoCapture(str(path))
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    hits, best = 0, 0.0
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.resize(fr, (640, 360))
        rects, _ = hog.detectMultiScale(fr, winStride=(8, 8))
        for (x, y, w, h) in rects:
            if h / fr.shape[0] > 0.25:
                hits += 1
                best = max(best, w / fr.shape[1])
    cap.release()
    return hits, best


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

    lines = ["=== 人在画面里多大 ===",
             "  未能客观测量：opencv-python 5.0.0.93 已无 cv2.HOGDescriptor，环境内也没有人体检测器。",
             "  已知客观事实：三段视频均为 1280x720 @30fps。"]

    lines.append("\n=== 视频级判定（16帧，max_pixels=150000）===")
    correct = tot = 0
    for name, t0, t1, is_fall, tag in CASES:
        cap = cv2.VideoCapture(str(D / name))
        fps = cap.get(cv2.CAP_PROP_FPS)
        idx = np.linspace(int(t0 * fps), int(t1 * fps), NF, dtype=int)
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
            if ok: frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        cap.release()
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": Q}]}],
            tokenize=False, add_generation_prompt=True)
        ins = processor(text=[text], videos=[frames], return_tensors="pt", padding=True)
        ins = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in ins.items()}
        if "pixel_values" in ins: ins["pixel_values"] = ins["pixel_values"].to(model.dtype)
        t = time.time()
        with torch.inference_mode():
            out = model.generate(**ins, max_new_tokens=12, do_sample=False)
        ans = processor.tokenizer.decode(out[0, ins["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        got = ans.strip().lower().startswith("yes")
        ok = (got == is_fall)
        tot += 1; correct += ok
        verdict = "命中" if (is_fall and got) else ("漏检" if (is_fall and not got) else ("误报" if got else "正确否定"))
        line = f"  {name[-6:-4]} 真值={'有摔倒' if is_fall else '无摔倒'} @{t0}-{t1}s {tag:<20} -> {ans[:20]:<20} {verdict} {'OK' if ok else 'X'}"
        print(line, flush=True); lines.append(line)
    lines.append(f"\n汇总：{correct}/{tot} 判对；摔倒窗口命中数见上（漏检=摔倒窗口答 no）")
    (OUT / "probe_fall_multivideo.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
