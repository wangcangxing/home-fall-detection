#!/usr/bin/env python
"""构建 URFD 的 YOLO 数据集（姿态三分类，方案 A）。

做四件事：
  1. 从关键点判姿态类别（stand / sit / lie），**每帧一个标注文件**
  2. **按官方 video_splits.pkl 划分** train/val/test —— 按视频划分，避免相邻帧泄漏
  3. 用**硬链接**把图片接进 images/（同一磁盘卷，不复制、不占额外空间）
  4. 写 data.yaml

用法：可重复运行；图片尚未下载完时只处理已存在的帧并报告缺口。
"""
import json
import math
import os
import pickle
from collections import Counter
from pathlib import Path

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
L_AN, R_AN = 15, 16

CONF_TAU = 0.30
MIN_JOINTS = 6
LIE_ANGLE = 60.0
HIP_FLOOR = 0.80
BOX_PAD = 0.08
IMG_W, IMG_H = 640, 480

CLASSES = ["stand", "sit", "lie"]

ROOT = Path(r"E:\MageVL\eval\urfd")
OUT = Path(r"E:\MageVL\dataset\urfd_yolo")


def mid(k, a, b):
    if k[a, 2] > CONF_TAU and k[b, 2] > CONF_TAU:
        return (k[a, 0] + k[b, 0]) / 2.0, (k[a, 1] + k[b, 1]) / 2.0
    return None


def classify(k):
    valid = k[:, 2] > CONF_TAU
    if valid.sum() < MIN_JOINTS:
        return None, "关节不足"
    sh, hip = mid(k, L_SH, R_SH), mid(k, L_HIP, R_HIP)
    if sh is None or hip is None:
        return None, "缺肩或髋"
    dx, dy = hip[0] - sh[0], hip[1] - sh[1]
    torso = math.hypot(dx, dy)
    if torso < 1e-3:
        return None, "躯干退化"
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    if angle > LIE_ANGLE:
        return "lie", f"角{angle:.0f}"
    if hip[1] / IMG_H > HIP_FLOOR:
        return "lie", f"髋高{hip[1]/IMG_H:.2f}"
    ankles = [k[i] for i in (L_AN, R_AN) if k[i, 2] > CONF_TAU]
    if not ankles:
        return ("stand" if angle < 20 else "sit"), f"角{angle:.0f}无踝"
    ank_y = sum(a[1] for a in ankles) / len(ankles)
    r = (ank_y - hip[1]) / torso
    return ("sit" if r < 0.9 else "stand"), f"髋踝{r:.2f}"


def main():
    for s in ("train", "val", "test"):
        (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)

    data = pickle.load(open(ROOT / "poses" / "URFD_keypoints.pkl", "rb"))
    splits = pickle.load(open(ROOT / "processed" / "URFD" / "video_splits.pkl", "rb"))
    vsplit = {}
    for s, vids in splits.items():
        for v in vids:
            vsplit[v] = s

    cidx = {c: i for i, c in enumerate(CLASSES)}
    stats = Counter()
    per_split = Counter()
    missing_img = 0
    no_pose = 0
    linked = 0
    vid_not_in_split = []

    for rec in data:
        vid = rec["video_id"]
        split = vsplit.get(vid)
        if split is None:
            vid_not_in_split.append(vid)
            continue
        leaf = vid.split("/")[-1]
        kp = rec["keypoints"]
        for t in range(kp.shape[0]):
            k = kp[t]
            cls, _ = classify(k)
            if cls is None:
                no_pose += 1
                continue
            valid = k[:, 2] > CONF_TAU
            xs, ys = k[valid, 0], k[valid, 1]
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            pad = BOX_PAD * max(x1 - x0, y1 - y0)
            x0, y0 = max(0.0, x0 - pad), max(0.0, y0 - pad)
            x1, y1 = min(float(IMG_W), x1 + pad), min(float(IMG_H), y1 + pad)
            if x1 - x0 < 4 or y1 - y0 < 4:
                no_pose += 1
                continue

            src = ROOT / "raw" / "URFD" / vid / f"{leaf}-{t+1:03d}.png"
            if not src.is_file():
                missing_img += 1
                continue

            stem = f"{leaf}-{t+1:03d}"
            dst_img = OUT / "images" / split / f"{stem}.png"
            dst_lab = OUT / "labels" / split / f"{stem}.txt"
            if not dst_img.exists():
                try:
                    os.link(src, dst_img)          # 硬链接：不占额外空间
                except OSError:
                    import shutil
                    shutil.copy2(src, dst_img)     # 跨卷时退回复制
            cx, cy = (x0 + x1) / 2 / IMG_W, (y0 + y1) / 2 / IMG_H
            bw, bh = (x1 - x0) / IMG_W, (y1 - y0) / IMG_H
            dst_lab.write_text(f"{cidx[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n",
                               encoding="utf-8")
            stats[cls] += 1
            per_split[split] += 1
            linked += 1

    yaml_txt = (
        f"# URFD 姿态三分类数据集（由 scripts/build_urfd_yolo.py 生成）\n"
        f"# 划分来自数据集自带的 video_splits.pkl —— **按视频划分，无帧级泄漏**\n"
        f"path: {OUT.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    (OUT / "data.yaml").write_text(yaml_txt, encoding="utf-8")

    report = {
        "已生成实例总数": linked,
        "按类别": dict(stats),
        "按划分": dict(per_split),
        "跳过_无姿态": no_pose,
        "跳过_图片未下载": missing_img,
        "不在官方划分里的视频": vid_not_in_split,
        "data.yaml": yaml_txt,
    }
    (OUT / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")

    print(f"生成实例: {linked}")
    print("  按类别:", dict(stats))
    print("  按划分:", dict(per_split))
    print(f"  跳过(无姿态): {no_pose}   跳过(图未下载): {missing_img}")
    if vid_not_in_split:
        print(f"  ⚠ 不在官方划分里的视频 {len(vid_not_in_split)} 个: {vid_not_in_split[:5]}")
    print(f"\ndata.yaml 已写入 {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
