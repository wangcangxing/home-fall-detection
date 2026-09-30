#!/usr/bin/env python
"""probe D：扩充版跌倒识别评测 + 时序/对照测试。

设计要点（因为无法独立确认视频真值，改用可判定的对照设计）：
  - 对照问题「Is there a person in this image?」与主问题「...lying on the ground?」同时问，
    用来区分「真的在判躺倒」还是「只是检测到有人」——若两问答案完全一致，说明主问题没在起作用。
  - 负样本尽量取同域：足球视频帧（有人、无人躺倒）+ dog.jpg + 跌倒视频开头帧（同场景同人）。
    注意：跌倒视频的「第几秒倒下」我无法独立确认，故开头帧只作弱负样本并在报告里标注。
  - 无绝对真值的时段，报告「答案时间线」而不是准确率。
"""
import re
import time
from pathlib import Path

import torch

MODEL = r"E:\MageVL\Mage-VL"
CCTV = Path(r"E:\MageVL\eval\fall-CCTV_Incident_Dataset_Fall_Lying_Down_Detection\laying_dataset")
FRAMES = Path(r"E:\MageVL\baseline\fallframes")
OUT = Path(r"E:\MageVL\baseline")
Q_LIE = "Is there a person lying on the ground in this image? Answer yes or no."
Q_PERSON = "Is there a person in this image? Answer yes or no."


def yes(ans):
    a = ans.strip().lower()
    if re.match(r"^(yes|y)\b", a):
        return True
    if re.match(r"^(no|n)\b", a):
        return False
    return None


def main():
    import cv2
    import numpy as np
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True,
        torch_dtype="auto", device_map="auto").eval()
    print("模型已加载", flush=True)

    def ask(img_path, question):
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}],
            tokenize=False, add_generation_prompt=True)
        ins = processor(text=[text], images=[Image.open(img_path).convert("RGB")], return_tensors="pt")
        ins = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in ins.items()}
        with torch.inference_mode():
            out = model.generate(**ins, max_new_tokens=8, do_sample=False)
        return processor.tokenizer.decode(out[0, ins["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    lines = []

    # ---------------- Test A：状态识别，大样本 ----------------
    pos = []
    for t in sorted((CCTV / "labels").glob("*.txt")):
        cls = {int(l.split()[0]) for l in t.read_text().splitlines() if l.strip()}
        if 0 in cls:
            p = CCTV / "images" / (t.stem + ".png")
            if p.is_file():
                pos.append(p)

    neg = [Path(MODEL) / "examples" / "dog.jpg"]
    cap = cv2.VideoCapture(str(Path(MODEL) / "examples" / "soccer-broadcast.mp4"))
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in np.linspace(0, fc - 1, 60, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
        if ok:
            p = OUT / f"_soc_{int(i):05d}.jpg"
            Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).save(p, quality=90)
            neg.append(p)
    cap.release()
    for name in ("sample_1", "sample_2"):
        for t in (0, 1, 2, 3):
            p = FRAMES / f"{name}_t{t:02d}s.jpg"
            if p.is_file():
                neg.append(p)

    lines.append(f"=== Test A 状态识别：正类 {len(pos)} / 负类 {len(neg)} ===")
    tp = fn = tn = fp = unk = 0
    t0 = time.time()
    detail = []
    for label, items in (("POS", pos), ("NEG", neg)):
        for p in items:
            ans = ask(p, Q_LIE); v = yes(ans)
            if label == "POS":
                tp += v is True; fn += v is False; unk += v is None
            else:
                tn += v is False; fp += v is True; unk += v is None
            detail.append(f"  [{label}] {p.name:34s} -> {ans[:40]!r}")
    n = tp + fn + tn + fp
    lines.append(f"混淆矩阵: TP={tp} FN={fn} TN={tn} FP={fp} 无法判定={unk}")
    lines.append(f"准确率={(tp+tn)/n:.4f} (n={n})  漏报率={fn/max(tp+fn,1):.4f}  误报率={fp/max(tn+fp,1):.4f}")
    lines.append(f"耗时 {time.time()-t0:.1f}s（{(time.time()-t0)/max(len(pos)+len(neg),1):.2f}s/张）")
    lines += detail

    # ---------------- Test B：时序 + 对照 ----------------
    lines.append("\n=== Test B 时序时间线 + 「有没有人」对照 ===")
    for name in ("sample_1", "sample_2"):
        row_lie, row_per = [], []
        for t in range(0, 14):
            p = FRAMES / f"{name}_t{t:02d}s.jpg"
            if not p.is_file():
                continue
            vl, vp = yes(ask(p, Q_LIE)), yes(ask(p, Q_PERSON))
            row_lie.append("Y" if vl else ("n" if vl is False else "?"))
            row_per.append("Y" if vp else ("n" if vp is False else "?"))
        s = "".join(row_lie)
        lines.append(f"{name} t=0..13s  躺倒? [{s}]  有人? [{''.join(row_per)}]")
        lines.append(f"    判为躺倒秒数={s.count('Y')}  判为有人秒数={''.join(row_per).count('Y')}"
                     f"  → {'两问高度一致(疑似只检测到有人)' if s == ''.join(row_per) else '两问不一致(主问题在起作用)'}")

    # 负控制：足球视频应当全程 no（用本脚本刚抽出的 _soc_ 帧，等距取 14 张）
    soc = sorted(OUT.glob("_soc_*.jpg"))
    step = max(1, len(soc) // 14)
    lie, per = [], []
    for p in soc[::step][:14]:
        vl, vp = yes(ask(p, Q_LIE)), yes(ask(p, Q_PERSON))
        lie.append("Y" if vl else ("n" if vl is False else "?"))
        per.append("Y" if vp else ("n" if vp is False else "?"))
    lines.append(f"soccer(负控制, {len(soc[::step][:14])}帧) 躺倒? [{''.join(lie)}]  有人? [{''.join(per)}]")

    (OUT / "probe_fall_expanded.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6]), flush=True)


if __name__ == "__main__":
    main()
