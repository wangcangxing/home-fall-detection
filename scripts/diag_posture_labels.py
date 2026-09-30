#!/usr/bin/env python
"""诊断：为什么部分 fall 视频一个 lie 都判不出来？

对每个视频沿时间轴提取三个量：
  angle     躯干与竖直的夹角（当前判据 >60° 判 lie）
  hip_y     髋部中心的归一化 y（越大越靠近画面底部=越接近地面）
  hip_ankle 髋-踝垂直距离 / 躯干长（站立时大，坐/躺时小）
用来判断：是判据不对（角度没到 60°），还是数据本身没人躺下。
"""
import pickle
import math
from pathlib import Path

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
L_AN, R_AN = 15, 16
TAU = 0.30
IMG_H = 480


def mid(k, a, b):
    if k[a, 2] > TAU and k[b, 2] > TAU:
        return (k[a, 0] + k[b, 0]) / 2, (k[a, 1] + k[b, 1]) / 2
    return None


def series(kp):
    ang, hy, ha = [], [], []
    for t in range(kp.shape[0]):
        k = kp[t]
        sh, hip = mid(k, L_SH, R_SH), mid(k, L_HIP, R_HIP)
        if sh is None or hip is None:
            ang.append(None); hy.append(None); ha.append(None); continue
        dx, dy = hip[0] - sh[0], hip[1] - sh[1]
        torso = math.hypot(dx, dy)
        ang.append(math.degrees(math.atan2(abs(dx), abs(dy))) if torso > 1e-3 else None)
        hy.append(hip[1] / IMG_H)
        anks = [k[i] for i in (L_AN, R_AN) if k[i, 2] > TAU]
        ha.append(((sum(a[1] for a in anks) / len(anks)) - hip[1]) / torso if anks and torso > 1e-3 else None)
    return ang, hy, ha


def summ(v):
    v = [x for x in v if x is not None]
    if not v:
        return "无"
    v2 = sorted(v)
    return f"中位 {v2[len(v2)//2]:.2f} 最大 {max(v2):.2f}"


def main():
    data = pickle.load(open(Path(r"E:\MageVL\eval\urfd\poses\URFD_keypoints.pkl"), "rb"))
    print("=== 问题视频（fall 但 lie≈0） vs 正常视频 的三项指标 ===")
    print(f"{'视频':<22}{'角度 中位/最大':<24}{'髋高 中位/最大':<24}{'髋踝比 中位/最小'}")
    show = ["fall-25-cam0-rgb", "fall-27-cam0-rgb", "fall-21-cam0-rgb",
            "fall-01-cam0-rgb", "fall-10-cam0-rgb", "fall-30-cam0-rgb"]
    for rec in data:
        leaf = rec["video_id"].split("/")[-1]
        if leaf not in show:
            continue
        ang, hy, ha = series(rec["keypoints"])
        def s(v, big=True):
            v = [x for x in v if x is not None]
            if not v: return "无"
            v2 = sorted(v)
            m = v2[len(v2)//2]
            return f"{m:.2f} / {(max(v2) if big else min(v2)):.2f}"
        print(f"{leaf:<22}{s(ang):<24}{s(hy):<24}{s(ha, False)}")

    print("\n=== 全量统计：fall 视频里「角度从未超过 60°」的有几个 ===")
    never60, never_hip = [], []
    for rec in data:
        if int(rec["label"]) != 1:
            continue
        ang, hy, _ = series(rec["keypoints"])
        a = [x for x in ang if x is not None]
        h = [x for x in hy if x is not None]
        leaf = rec["video_id"].split("/")[-1]
        if not a or max(a) <= 60:
            never60.append((leaf, round(max(a), 1) if a else None, round(max(h), 2) if h else None))
        if h and max(h) < 0.72:
            never_hip.append((leaf, round(max(h), 2)))
    print(f"  角度最大值 ≤60° 的 fall 视频: {len(never60)} 个")
    for x in never60:
        print(f"    {x[0]:<22} 最大角度={x[1]}  最大髋高={x[2]}")
    print(f"  髋高最大值 <0.72（从未接近地面）的 fall 视频: {len(never_hip)} 个")
    for x in never_hip[:10]:
        print(f"    {x[0]:<22} 最大髋高={x[1]}")


if __name__ == "__main__":
    main()
