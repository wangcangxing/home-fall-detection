#!/usr/bin/env python
"""评估姿态检测器：不看 mAP，看「该报 lie 的时候报没报」。

对 test 划分逐帧推理，取置信度最高的人体框类别，然后**按视频聚合**：
  - fall 视频：摔倒之后应当出现 lie
  - nonfall 视频：URFD 的 ADL 序列本身就含「正常躺下」，所以出现 lie 是**正确的**
    （这些是难负样本——人躺在地上但没摔）

输出：
  - 每视频的类别命中统计
  - 逐类 AP（来自 ultralytics val）
  - 与朴素几何基线的对比（框纵横比 82.7% / 躺倒召回 66%）

用法：
  python eval_posture_detector.py --weights E:\\MageVL\\runs\\urfd_posture_v1\\weights\\best.pt
"""
import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"E:\MageVL\eval\urfd")
DATA = Path(r"E:\MageVL\dataset\urfd_yolo\data.yaml")
IMG_ROOT = Path(r"E:\MageVL\eval\urfd\raw\URFD")
REPO_RESULTS = Path(r"D:\program\模型优化\results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 帧（调试）")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)

    # 先跑一次标准 val 拿逐类 AP
    print("=== ultralytics val（标准指标）===", flush=True)
    m = model.val(data=str(DATA), split=args.split, imgsz=args.imgsz, verbose=False)
    try:
        names = m.names
        ap50 = m.box.ap50
        per_class = {names[i]: float(ap50[i]) for i in range(len(names))}
    except Exception as e:
        per_class = {"err": str(e)}
    print("  mAP50  :", round(float(m.box.map50), 4))
    print("  mAP50-95:", round(float(m.box.map), 4))
    for k, v in per_class.items():
        print(f"    {k:<8} AP50={v}")

    # 逐帧推理，按视频聚合
    split_vids = pickle.load(open(ROOT / "processed" / "URFD" / "video_splits.pkl", "rb"))[args.split]
    vlabel = {}
    for rec in pickle.load(open(ROOT / "poses" / "URFD_keypoints.pkl", "rb")):
        vlabel[rec["video_id"]] = int(rec["label"])

    imgs = sorted((Path(r"E:\MageVL\dataset\urfd_yolo\images") / args.split).glob("*.png"))
    if args.limit:
        imgs = imgs[:args.limit]
    print(f"\n=== 逐帧推理 {len(imgs)} 帧（conf>{args.conf}）===", flush=True)

    per_video = defaultdict(Counter)
    for i in range(0, len(imgs), 64):
        batch = [str(p) for p in imgs[i:i + 64]]
        res = model.predict(batch, imgsz=args.imgsz, conf=args.conf, verbose=False)
        for p, r in zip(imgs[i:i + 64], res):
            leaf = p.stem.rsplit("-", 1)[0]          # 如 fall-01-cam0-rgb
            if r.boxes is None or len(r.boxes) == 0:
                per_video[leaf]["none"] += 1
                continue
            cls = r.boxes.cls.int().tolist()
            conf = r.boxes.conf.tolist()
            best = max(zip(conf, cls))[1]            # 置信度最高的那个框
            per_video[leaf][model.names[best]] += 1
        if (i // 64) % 10 == 0:
            print(f"  {min(i+64,len(imgs))}/{len(imgs)}", flush=True)

    # 聚合
    rows = []
    for leaf, c in sorted(per_video.items()):
        vid = next((v for v in split_vids if v.endswith(leaf)), None)
        lab = vlabel.get(vid, -1)
        tot = sum(c.values())
        rows.append({"video": leaf, "true_fall": lab, "frames": tot,
                     "stand": c["stand"], "sit": c["sit"], "lie": c["lie"], "none": c["none"],
                     "lie_frac": c["lie"] / max(tot, 1)})

    fall_v = [r for r in rows if r["true_fall"] == 1]
    non_v = [r for r in rows if r["true_fall"] == 0]

    def summ(g, tag):
        if not g:
            return f"{tag}: 无视频"
        hit = sum(1 for r in g if r["lie"] > 0)
        frac = sorted(r["lie_frac"] for r in g)
        return (f"{tag}: {len(g)} 个视频，其中检出 lie 的 {hit} 个 "
                f"({hit/len(g)*100:.0f}%)；lie 占比 中位 {frac[len(frac)//2]:.3f} "
                f"最大 {max(frac):.3f}")

    print("\n=== 按视频聚合（关键结果）===")
    print(" ", summ(fall_v, "fall 视频  "))
    print(" ", summ(non_v, "nonfall 视频"))
    print("\n  说明：nonfall 视频出现 lie 是**正常的**——URFD 的 ADL 序列含正常人主动躺下。")
    print("        这些正是系统必须不报警的难负样本。")
    print("\n  逐视频明细：")
    for r in sorted(rows, key=lambda x: (-x["true_fall"], -x["lie_frac"])):
        print(f"    [{'FALL' if r['true_fall']==1 else 'adl '}] {r['video']:<22} "
              f"帧{r['frames']:>4} stand={r['stand']:>4} sit={r['sit']:>3} "
              f"lie={r['lie']:>4} none={r['none']:>3} lie占比={r['lie_frac']:.3f}")

    REPO_RESULTS.mkdir(parents=True, exist_ok=True)
    out = REPO_RESULTS / "eval_posture_detector.json"
    out.write_text(json.dumps({
        "weights": args.weights, "split": args.split, "conf": args.conf,
        "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
        "per_class_AP50": per_class,
        "按视频": rows,
        "汇总": {"fall": summ(fall_v, "fall"), "nonfall": summ(non_v, "nonfall")},
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n汇总已写入 {out}")


if __name__ == "__main__":
    main()
