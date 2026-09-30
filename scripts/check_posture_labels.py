#!/usr/bin/env python
"""校验生成的姿态标注是否合理（无法看图，改用几何自洽性）。

检查三件事：
  1. 各姿态类别的**框纵横比**分布——lie 的框应当明显偏宽（w>h），stand 明显偏窄
  2. nonfall 视频里出现 lie 的视频，逐视频看占比（区分"ADL 里正常躺下"与"判据误判"）
  3. lie 框的**中心高度**——躺地上的框中心应偏低
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

out = Path(r"E:\MageVL\dataset\urfd_yolo")
CLASSES = ["stand", "sit", "lie"]


def main():
    stats = json.loads((out / "label_stats.json").read_text(encoding="utf-8"))
    print("=== 1) 逐视频：nonfall 里 lie 占比最高的 12 个 ===")
    non = [v for v in stats["逐视频"] if v["label"] == 0]
    for v in sorted(non, key=lambda x: -x["lie_frac"])[:12]:
        d = v["dist"]
        print(f"  lie={v['lie_frac']:.3f} frames={v['frames']:>4} "
              f"stand={d.get('stand',0):>4} sit={d.get('sit',0):>3} lie={d.get('lie',0):>4}  {v['video_id'].split('/')[-1]}")

    print("\n=== 2) 逐视频：fall 里 lie 占比最低的 8 个 ===")
    fall = [v for v in stats["逐视频"] if v["label"] == 1]
    for v in sorted(fall, key=lambda x: x["lie_frac"])[:8]:
        d = v["dist"]
        print(f"  lie={v['lie_frac']:.3f} frames={v['frames']:>4} "
              f"stand={d.get('stand',0):>4} sit={d.get('sit',0):>3} lie={d.get('lie',0):>4}  {v['video_id'].split('/')[-1]}")

    print("\n=== 3) 框几何自洽性：各类别的纵横比(w/h)与中心高度 ===")
    asp = defaultdict(list)
    cy = defaultdict(list)
    for f in (out / "labels").glob("*.txt"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            c, cx, cyy, w, h = line.split()
            c = int(c)
            asp[CLASSES[c]].append(float(w) / max(float(h), 1e-9))
            cy[CLASSES[c]].append(float(cyy))

    print(f"  {'类别':<7} {'框数':>6} {'纵横比 中位':>11} {'纵横比 25%':>10} {'纵横比 75%':>10} {'中心高度 中位':>13}")
    for c in CLASSES:
        a, y = asp[c], cy[c]
        if not a:
            continue
        a.sort()
        print(f"  {c:<7} {len(a):>6} {statistics.median(a):>11.2f} "
              f"{a[len(a)//4]:>10.2f} {a[3*len(a)//4]:>10.2f} {statistics.median(y):>13.3f}")
    print("\n  判据：lie 的纵横比中位应 > 1（横躺），stand 应 < 1（竖立）；")
    print("        若 lie 与 stand 的纵横比分布大幅重叠，说明姿态判据不可靠。")


if __name__ == "__main__":
    main()
