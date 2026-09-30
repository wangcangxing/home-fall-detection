#!/usr/bin/env python
"""probe K：验证「YOLO 人体框几何」能否替代/补足 VLM 做姿态与位置判断。

不需要新装依赖：用已安装的 torchvision 检测模型（COCO 预训练，类别 1 = person）。
对每个图取最大的人体框，计算：
  aspect = w / h    （>1 横置=可能躺卧；<1 竖立=站立/坐）
  bottom = box下沿 y / 图高  （接近 1 = 贴近画面底部=地面；明显小于 1 = 家具高度）
对「真躺倒图」与「站立/正常图」比较这两个量的可分性。
"""
import json
import time
from pathlib import Path

import torch

CCTV = Path(r"E:\MageVL\eval\fall-CCTV_Incident_Dataset_Fall_Lying_Down_Detection\laying_dataset")
OUT = Path(r"E:\MageVL\baseline")
MODEL = r"E:\MageVL\Mage-VL"
N = 40


def main():
    import cv2
    import numpy as np
    from PIL import Image
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
    from torchvision.transforms.functional import pil_to_tensor

    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).eval().cuda()
    print("检测模型已加载", flush=True)

    # 正类：CCTV 含 laying 的图
    pos = []
    for t in sorted((CCTV / "labels").glob("*.txt")):
        cls = {int(l.split()[0]) for l in t.read_text().splitlines() if l.strip()}
        if cls == {0}:
            p = CCTV / "images" / (t.stem + ".png")
            if p.is_file():
                pos.append(p)
        if len(pos) >= N:
            break
    # 负类：足球视频帧（人站立/跑动）
    neg = []
    cap = cv2.VideoCapture(str(Path(MODEL) / "examples" / "soccer-broadcast.mp4"))
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in np.linspace(0, fc - 1, N, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
        if ok:
            p = OUT / f"_yolo_{int(i):05d}.jpg"
            Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).save(p, quality=90)
            neg.append(p)
    cap.release()

    def measure(path):
        img = Image.open(path).convert("RGB")
        t = pil_to_tensor(img).float().cuda() / 255.0
        with torch.inference_mode():
            pred = model([t])[0]
        keep = (pred["labels"] == 1) & (pred["scores"] > 0.6)
        if keep.sum() == 0:
            return None
        boxes = pred["boxes"][keep]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        b = boxes[areas.argmax()]
        w, h = float(b[2] - b[0]), float(b[3] - b[1])
        H = img.size[1]
        return {"aspect": w / max(h, 1e-6), "bottom": float(b[3]) / H,
                "area_frac": (w * h) / (img.size[0] * H)}

    res = {"pos": [], "neg": []}
    for label, items in (("pos", pos), ("neg", neg)):
        for p in items:
            m = measure(p)
            if m:
                res[label].append(m)

    def stat(rs, key):
        v = sorted(r[key] for r in rs)
        if not v:
            return None
        return (len(v), v[0], v[len(v) // 2], v[-1])

    lines = ["=== probe K：YOLO 人体框几何能否区分「躺倒」vs「站立」===",
             f"正类（CCTV 真躺倒）{len(pos)} 张 → 检出人 {len(res['pos'])}",
             f"负类（足球站立/跑动）{len(neg)} 张 → 检出人 {len(res['neg'])}", ""]
    for key in ("aspect", "bottom"):
        lines.append(f"{key}: (n, min, 中位, max)")
        for lab in ("pos", "neg"):
            s = stat(res[lab], key)
            lines.append(f"   {'躺倒' if lab=='pos' else '站立'}: {s}")

    # 用 aspect 做阈值分类，找最佳阈值
    best = None
    for thr in [x / 100 for x in range(50, 260, 5)]:
        tp = sum(1 for r in res["pos"] if r["aspect"] > thr)
        tn = sum(1 for r in res["neg"] if r["aspect"] <= thr)
        acc = (tp + tn) / max(len(res["pos"]) + len(res["neg"]), 1)
        if best is None or acc > best[1]:
            best = (thr, acc, tp, len(res["pos"]), tn, len(res["neg"]))
    lines.append(f"\n单用 aspect 的最佳阈值 = {best[0]:.2f}，准确率 {best[1]:.3f}"
                 f"（躺倒判对 {best[2]}/{best[3]}，站立判对 {best[4]}/{best[5]}）")

    # 组合 aspect + bottom
    best2 = None
    for ta in [x / 100 for x in range(60, 200, 10)]:
        for tb in [x / 100 for x in range(60, 101, 5)]:
            tp = sum(1 for r in res["pos"] if r["aspect"] > ta and r["bottom"] > tb)
            tn = sum(1 for r in res["neg"] if not (r["aspect"] > ta and r["bottom"] > tb))
            acc = (tp + tn) / max(len(res["pos"]) + len(res["neg"]), 1)
            if best2 is None or acc > best2[1]:
                best2 = (ta, tb, acc, tp, len(res["pos"]), tn, len(res["neg"]))
    lines.append(f"aspect>{best2[0]:.2f} 且 bottom>{best2[1]:.2f} 判为躺倒：准确率 {best2[2]:.3f}"
                 f"（躺倒判对 {best2[3]}/{best2[4]}，站立判对 {best2[5]}/{best2[6]}）")

    (OUT / "probe_yolo_geometry.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
