#!/usr/bin/env python
"""probe I：验证用户方案——「每 5 秒拍一张，直接读图判状态」。

对整段真实室内视频按 5 秒间隔逐帧提问，输出完整时间线：
  - 真值摔倒区间内是否有 yes（命中）
  - 真值区间外的 yes 数量（误报），并给出误报率
  - 单图延迟（决定这个方案在边缘设备上的可行性）
"""
import csv
import time
from pathlib import Path

import torch

MODEL = r"E:\MageVL\Mage-VL"
V = Path(r"E:\MageVL\eval\posture")
OUT = Path(r"E:\MageVL\baseline")
Q = "Is there a person lying on the ground in this image? Answer yes or no."
INTERVAL = 5.0


def load_falls(video_dir, labels_dir, stem):
    p = Path(labels_dir) / f"{stem}.csv"
    if not p.is_file():
        return []
    out = []
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("is_fall", "")).strip().lower() == "true":
                out.append((float(r["start_time"]), float(r["end_time"])))
    return out


def main():
    import cv2
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar(); hf_logging.set_verbosity_error()

    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype="auto", device_map="auto").eval()
    print("模型已加载", flush=True)

    def ask(img):
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q}]}],
            tokenize=False, add_generation_prompt=True)
        ins = processor(text=[text], images=[img], return_tensors="pt")
        ins = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in ins.items()}
        t = time.time()
        with torch.inference_mode():
            out = model.generate(**ins, max_new_tokens=8, do_sample=False)
        a = processor.tokenizer.decode(out[0, ins["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        ntok = int(ins["input_ids"].shape[1])
        return a.lower().startswith("yes"), time.time() - t, ntok

    lines = [f"=== probe I：每 {INTERVAL:.0f} 秒一张图，直接判「有人躺在地上」==="]
    for sub, stem in (("train", "hr_fall_detection_3"), ("valid", "fall_detection_10"),
                      ("train", "hr_fall_detection_1")):
        vid = V / sub / f"{stem}.mp4"
        lab = V / sub / "labels"
        if not vid.is_file():
            continue
        falls = load_falls(None, lab, stem)
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = n / fps
        lines.append(f"\n--- {stem} ({dur:.0f}s, {len(falls)} 个真值摔倒区间: {falls}) ---")
        t = 0.0; tl = []; lat = []; toks = []
        while t < dur:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            if not ok:
                break
            img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            got, dt, nt = ask(img)
            lat.append(dt); toks.append(nt)
            tl.append((t, got))
            t += INTERVAL
        cap.release()
        # 统计
        in_win = [g for tt, g in tl if any(s <= tt <= e + 0.5 for s, e in falls)]
        out_win = [(tt, g) for tt, g in tl if not any(s <= tt <= e + 0.5 for s, e in falls)]
        hit = sum(1 for g in in_win if g)
        fp = sum(1 for _, g in out_win if g)
        lines.append(f"  采样 {len(tl)} 张；真值区间内 {len(in_win)} 张 → 命中 {hit}")
        lines.append(f"  区间外 {len(out_win)} 张 → 误报 {fp}（误报率 {fp/max(len(out_win),1):.3f}）")
        lines.append(f"  单图延迟 平均 {sum(lat)/len(lat):.2f}s (min {min(lat):.2f} / max {max(lat):.2f})，prompt token 均值 {sum(toks)//len(toks)}")
        lines.append("  时间线(秒:yes/no，* 标记真值区间): " + " ".join(
            f"{tt:.0f}:{'Y' if g else 'n'}{'*' if any(s <= tt <= e + 0.5 for s, e in falls) else ''}" for tt, g in tl))
        fpdetail = [f"{tt:.0f}s" for tt, g in out_win if g]
        if fpdetail:
            lines.append(f"  误报发生在: {', '.join(fpdetail)}")

    (OUT / "probe_snapshot5s.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
