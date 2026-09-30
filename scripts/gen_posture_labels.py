#!/usr/bin/env python
"""从 URFD 姿态关键点生成 YOLO 检测标注（方案 A：姿态三分类）。

输入：poses/URFD_keypoints.pkl
      70 个 dict，每个 {video_id, path, label, keypoints}
      keypoints 形状 (T, 17, 3) = (帧, COCO-17 关节, x/y/置信度)，**像素坐标**
      label: 1=fall 视频, 0=nonfall 视频（视频级）

输出：YOLO 格式
      labels/<video_id>.txt   每行 `cls cx cy w h`（归一化）
      images.txt              图片路径清单（供后续拷贝/软链）
      label_stats.json        类别分布 + 按视频级标签的交叉验证

姿态判据（COCO-17 索引）：
  0 nose, 5/6 shoulders, 11/12 hips, 13/14 knees, 15/16 ankles
  ① 躯干（肩中点→髋中点）与竖直方向夹角 > 60° → lie
  ② 否则用「髋-踝垂直距离 / 躯干长」判 sit / stand
"""
import json
import math
import pickle
from collections import Counter
from pathlib import Path

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
L_KN, R_KN, L_AN, R_AN = 13, 14, 15, 16

CONF_TAU = 0.30      # 关节置信度阈值
MIN_JOINTS = 6       # 有效关节少于这个数就跳过该帧
LIE_ANGLE = 60.0     # 躯干偏离竖直超过该角度判为躺
HIP_FLOOR = 0.80     # 髋部归一化 y 超过此值 → 人在接近地面的高度（诊断得出：
                     #   站立髋高中位 0.46~0.58，躺下 0.85~0.97；加这条是为了兜住
                     #   躯干角没到 60° 但人已落地的情况，如 fall-25 最大角度仅 50° 而髋高 0.91）
BOX_PAD = 0.08       # 框外扩比例
IMG_W, IMG_H = 640, 480   # URFD 标准分辨率（会由脚本用真实图片复核）


def mid(k, a, b):
    """两个关节的中点，两者都有效才返回。"""
    if k[a, 2] > CONF_TAU and k[b, 2] > CONF_TAU:
        return (k[a, 0] + k[b, 0]) / 2.0, (k[a, 1] + k[b, 1]) / 2.0
    return None


def classify(k):
    """返回 (cls_name, 诊断信息)；无法判定返回 (None, ...)。"""
    valid = k[:, 2] > CONF_TAU
    if valid.sum() < MIN_JOINTS:
        return None, "关节不足"

    sh = mid(k, L_SH, R_SH)
    hip = mid(k, L_HIP, R_HIP)
    if sh is None or hip is None:
        return None, "缺肩或髋"

    dx, dy = hip[0] - sh[0], hip[1] - sh[1]
    torso = math.hypot(dx, dy)
    if torso < 1e-3:
        return None, "躯干退化"
    # 与竖直方向夹角：0°=竖直站立，90°=水平躺倒
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))

    if angle > LIE_ANGLE:
        return "lie", f"躯干角 {angle:.0f}°"
    # 兜底：躯干角不够但髋部已经接近地面（人已落地）
    hip_y = hip[1] / IMG_H
    if hip_y > HIP_FLOOR:
        return "lie", f"髋高 {hip_y:.2f}（角 {angle:.0f}°）"

    # 髋到踝的垂直距离（图像坐标 y 向下为正）
    ankles = [k[i] for i in (L_AN, R_AN) if k[i, 2] > CONF_TAU]
    if not ankles:
        return ("stand" if angle < 20 else "sit"), f"躯干角 {angle:.0f}° 无踝"
    ank_y = sum(a[1] for a in ankles) / len(ankles)
    hip_to_ankle = (ank_y - hip[1]) / torso   # 站立时远大于 1

    if hip_to_ankle < 0.9:
        return "sit", f"髋踝比 {hip_to_ankle:.2f}"
    return "stand", f"髋踝比 {hip_to_ankle:.2f}"


def main():
    root = Path(r"E:\MageVL\eval\urfd")
    out = Path(r"E:\MageVL\dataset\urfd_yolo")
    (out / "labels").mkdir(parents=True, exist_ok=True)

    data = pickle.load(open(root / "poses" / "URFD_keypoints.pkl", "rb"))
    print(f"视频数: {len(data)}")

    CLASSES = ["stand", "sit", "lie"]
    cidx = {c: i for i, c in enumerate(CLASSES)}

    img_list, stats, per_video = [], Counter(), []
    for rec in data:
        vid = rec["video_id"]                       # 如 fall/fall-01-cam0-rgb/fall-01-cam0-rgb
        vlabel = int(rec["label"])
        kp = rec["keypoints"]
        leaf = vid.split("/")[-1]
        lines = []
        vc = Counter()
        for t in range(kp.shape[0]):
            k = kp[t]
            cls, why = classify(k)
            if cls is None:
                continue
            valid = k[:, 2] > CONF_TAU
            xs, ys = k[valid, 0], k[valid, 1]
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            w, h = x1 - x0, y1 - y0
            pad = BOX_PAD * max(w, h)
            x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
            # 裁到图像范围
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(IMG_W), x1), min(float(IMG_H), y1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            cx, cy = (x0 + x1) / 2 / IMG_W, (y0 + y1) / 2 / IMG_H
            bw, bh = (x1 - x0) / IMG_W, (y1 - y0) / IMG_H
            lines.append(f"{cidx[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            vc[cls] += 1
            stats[cls] += 1
            img_list.append(f"raw/URFD/{vid}/{leaf}-{t+1:03d}.png")
        if lines:
            safe = vid.replace("/", "__")
            (out / "labels" / f"{safe}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            per_video.append({"video_id": vid, "label": vlabel, "frames": len(lines),
                              "dist": dict(vc), "lie_frac": vc["lie"] / max(len(lines), 1)})

    (out / "images.txt").write_text("\n".join(img_list) + "\n", encoding="utf-8")

    # ==== 交叉验证：fall 视频应当出现 lie，nonfall 视频应当几乎没有 lie ====
    fall = [v for v in per_video if v["label"] == 1]
    non = [v for v in per_video if v["label"] == 0]
    def summ(g):
        if not g:
            return "无"
        lf = [v["lie_frac"] for v in g]
        return (f"视频 {len(g)} 个 | 有 lie 的视频 {sum(1 for x in lf if x > 0)} 个 "
                f"| lie 占比 中位 {sorted(lf)[len(lf)//2]:.3f} 最大 {max(lf):.3f}")
    report = {
        "类别总数": dict(stats),
        "总框数": sum(stats.values()),
        "标注文件数": len(per_video),
        "fall视频(真值=1)": summ(fall),
        "nonfall视频(真值=0)": summ(non),
        "逐视频": per_video,
    }
    (out / "label_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 类别分布 =====")
    for c in CLASSES:
        print(f"  {c:6} {stats[c]:>7}  ({stats[c]/max(sum(stats.values()),1)*100:.1f}%)")
    print(f"  合计 {sum(stats.values())} 个框，{len(per_video)} 个标注文件")
    print("\n===== 与视频级标签交叉验证（关键）=====")
    print("  fall   :", summ(fall))
    print("  nonfall:", summ(non))
    print("\n  期望：fall 视频大量出现 lie；nonfall 视频 lie 占比接近 0")
    print("  若 nonfall 也大量出现 lie → 姿态判据有误，需调参")


if __name__ == "__main__":
    main()
