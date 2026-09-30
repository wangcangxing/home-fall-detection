#!/usr/bin/env python
"""probe C：Mage-VL 到底能不能识别「有人摔倒/躺在地上」——这是整个方案的前提。

正类（20）：公开 CCTV 跌倒数据集的 laying 图（YOLO-pose 标注 class 0 = laying）
负类（20）：examples/dog.jpg + soccer-broadcast.mp4 采样帧（画面里有人，但无人躺倒）
问法：Is there a person lying on the ground in this image? Answer yes or no.
输出：混淆矩阵 + 准确率 + 全部原始回答（便于人工复核）
"""
import re
import time
from pathlib import Path

import torch

MODEL = r"E:\MageVL\Mage-VL"
CCTV = Path(r"E:\MageVL\eval\fall-CCTV_Incident_Dataset_Fall_Lying_Down_Detection\laying_dataset")
OUT = Path(r"E:\MageVL\baseline")
N_PER_CLASS = 20
QUESTION_IMG = "Is there a person lying on the ground in this image? Answer yes or no."
QUESTION_VID = "Is there a person falling or lying on the ground in this video? Answer yes or no, then describe briefly."


def pick_positives():
    pos = []
    for t in sorted((CCTV / "labels").glob("*.txt")):
        cls = {int(l.split()[0]) for l in t.read_text().splitlines() if l.strip()}
        if cls == {0}:
            img = CCTV / "images" / (t.stem + ".png")
            if img.is_file():
                pos.append(img)
        if len(pos) >= N_PER_CLASS:
            break
    return pos


def pick_negatives():
    out = [Path(MODEL) / "examples" / "dog.jpg"]
    import cv2
    import numpy as np
    from PIL import Image
    cap = cv2.VideoCapture(str(Path(MODEL) / "examples" / "soccer-broadcast.mp4"))
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, fc - 1, N_PER_CLASS - 1, dtype=int)
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            p = OUT / f"_negframe_{int(i):05d}.jpg"
            Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).save(p, quality=92)
            out.append(p)
    cap.release()
    return out


def main():
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()

    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    ).eval()
    print("模型已加载", flush=True)

    def ask_image(path):
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION_IMG}]}],
            tokenize=False, add_generation_prompt=True)
        ins = processor(text=[text], images=[Image.open(path).convert("RGB")], return_tensors="pt")
        ins = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in ins.items()}
        with torch.inference_mode():
            out = model.generate(**ins, max_new_tokens=12, do_sample=False)
        return processor.tokenizer.decode(out[0, ins["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def yes(ans):
        a = ans.strip().lower()
        if re.match(r"^(yes|y)\b", a):
            return True
        if re.match(r"^(no|n)\b", a):
            return False
        return None

    pos, neg = pick_positives(), pick_negatives()
    print(f"正类 {len(pos)} 张 / 负类 {len(neg)} 张\n", flush=True)

    rows = []
    tp = fn = tn = fp = unk = 0
    for label, items in (("POS(laying)", pos), ("NEG(not lying)", neg)):
        for p in items:
            t0 = time.time()
            ans = ask_image(p)
            v = yes(ans)
            dt = time.time() - t0
            if label.startswith("POS"):
                if v is True: tp += 1
                elif v is False: fn += 1
                else: unk += 1
            else:
                if v is False: tn += 1
                elif v is True: fp += 1
                else: unk += 1
            line = f"[{label}] {p.name:28s} {dt:5.1f}s -> {ans[:70]!r}"
            print(line, flush=True)
            rows.append(line)

    total = tp + fn + tn + fp
    acc = (tp + tn) / total if total else float("nan")
    summary = (f"\n混淆矩阵: 命中(真跌倒答yes)={tp} 漏报(真跌倒答no)={fn} "
               f"正确否定(正常答no)={tn} 误报(正常答yes)={fp} 无法判定={unk}\n"
               f"准确率 = {acc:.3f} (n={total})  漏报率={fn/(tp+fn) if tp+fn else float('nan'):.3f}  "
               f"误报率={fp/(tn+fp) if tn+fp else float('nan'):.3f}")
    print(summary, flush=True)
    rows.append(summary)

    # 视频
    frames_needed = 16
    vp = processor.video_processor
    ip = processor.image_processor
    orig_le = ip.size["longest_edge"]
    ip.size["longest_edge"] = 150000
    vp.min_pixels, vp.max_pixels = 3136, 150000
    import cv2
    import numpy as np
    for v in (Path(r"E:\MageVL\eval\fall-fall-detection\sample_1.mp4"),
              Path(r"E:\MageVL\eval\fall-fall-detection\sample_2.mp4")):
        cap = cv2.VideoCapture(str(v)); fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = np.linspace(0, fc - 1, min(frames_needed, fc), dtype=int)
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
            if ok: frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        cap.release()
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": QUESTION_VID}]}],
            tokenize=False, add_generation_prompt=True)
        ins = processor(text=[text], videos=[frames], return_tensors="pt", padding=True)
        ins = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in ins.items()}
        if "pixel_values" in ins:
            ins["pixel_values"] = ins["pixel_values"].to(model.dtype)
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**ins, max_new_tokens=90, do_sample=False)
        ans = processor.tokenizer.decode(out[0, ins["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        line = f"[VIDEO] {v.name} ({len(frames)}帧, {time.time()-t0:.1f}s) -> {ans[:400]}"
        print(line, flush=True)
        rows.append(line)
    ip.size["longest_edge"] = orig_le

    (OUT / "probe_fall.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
