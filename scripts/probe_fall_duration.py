#!/usr/bin/env python
"""probe H：按「摔倒事件时长」分档测命中率——直接回答「5 秒内就行」这个要求。

真值：pat2echo 数据集 valid/ 各片段的 CSV（start_time,end_time,action,is_fall）。
测法：对每个真值摔倒事件，取 [开始-1s, 开始+4s] 这 5 秒窗口、16 帧（≈3.2fps），
     问「Is there a person falling or lying on the ground in this video?」
     另取同视频内无摔倒窗口作对照，估误报。
"""
import csv
import time
from pathlib import Path

import torch

MODEL = r"E:\MageVL\Mage-VL"
V = Path(r"E:\MageVL\eval\posture\valid")
OUT = Path(r"E:\MageVL\baseline")
Q = "Is there a person falling or lying on the ground in this video? Answer yes or no."
NF = 16
MAX_PER_BUCKET = 4
BUCKETS = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 99)]


def load_events():
    ev = []
    for c in sorted((V / "labels").glob("*.csv")):
        vid = V / (c.stem + ".mp4")
        if not vid.is_file():
            continue
        with open(c, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if str(r.get("is_fall", "")).strip().lower() == "true":
                    ev.append((vid, float(r["start_time"]), float(r["end_time"]), c.stem))
    return ev


def main():
    import cv2
    import numpy as np
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar(); hf_logging.set_verbosity_error()

    events = load_events()
    print(f"valid/ 共 {len(events)} 个真值摔倒事件", flush=True)

    picks = []
    for lo, hi in BUCKETS:
        b = [e for e in events if lo <= (e[2] - e[1]) < hi]
        picks += [(e, f"{lo}-{hi}s") for e in b[:MAX_PER_BUCKET]]
    print(f"分档抽样 {len(picks)} 个事件待测", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype="auto", device_map="auto").eval()
    ip, vp = processor.image_processor, processor.video_processor
    ip.size["longest_edge"] = 150000
    vp.min_pixels, vp.max_pixels = 3136, 150000
    print("模型已加载", flush=True)

    def ask_video(vid, t0, t1):
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS)
        idx = np.linspace(int(t0 * fps), max(int(t1 * fps), int(t0 * fps) + 1), NF, dtype=int)
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
            if ok: frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        cap.release()
        if not frames:
            return None
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": Q}]}],
            tokenize=False, add_generation_prompt=True)
        ins = processor(text=[text], videos=[frames], return_tensors="pt", padding=True)
        ins = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in ins.items()}
        if "pixel_values" in ins: ins["pixel_values"] = ins["pixel_values"].to(model.dtype)
        with torch.inference_mode():
            out = model.generate(**ins, max_new_tokens=12, do_sample=False)
        a = processor.tokenizer.decode(out[0, ins["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return a.lower().startswith("yes")

    lines = ["=== probe H：按摔倒时长分档的命中率（5 秒窗口 / 16 帧 ≈3.2fps）==="]
    stat = {}
    for (vid, s, e, stem), bucket in picks:
        dur = e - s
        t0, t1 = max(0.0, s - 1.0), s + 4.0
        t = time.time()
        got = ask_video(vid, t0, t1)
        stat.setdefault(bucket, [0, 0])
        stat[bucket][1] += 1
        stat[bucket][0] += bool(got)
        line = f"  {stem:<18} 事件 {s:6.1f}-{e:6.1f}s (时长 {dur:4.1f}s) 窗口{t0:.0f}-{t1:.0f}s -> {'yes 命中' if got else 'no  漏检'}  {time.time()-t:.1f}s"
        print(line, flush=True); lines.append(line)

    lines.append("\n按事件时长分档的命中率：")
    for lo, hi in BUCKETS:
        b = f"{lo}-{hi}s"
        if b in stat:
            hit, tot = stat[b]
            lines.append(f"  时长 {b:<8} 命中 {hit}/{tot}")
    lines.append("注：窗口固定 5 秒（满足「5 秒内报出来」的需求形态）。")

    # 对照：同视频无摔倒窗口
    lines.append("\n对照组（同视频、无摔倒窗口，应为 no）：")
    ctrl = [("fall_detection_4", 20, 25), ("fall_detection_5", 30, 35),
            ("fall_detection_8", 50, 55), ("fall_detection_10", 20, 25), ("fall_detection_9", 20, 25)]
    cok = 0
    for stem, t0, t1 in ctrl:
        vid = V / f"{stem}.mp4"
        if not vid.is_file(): continue
        got = ask_video(vid, t0, t1)
        cok += (got is False)
        lines.append(f"  {stem} @{t0}-{t1}s -> {'yes 误报' if got else 'no  正确否定'}")
    lines.append(f"  对照正确率 {cok}/{len(ctrl)}")

    (OUT / "probe_fall_duration.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
