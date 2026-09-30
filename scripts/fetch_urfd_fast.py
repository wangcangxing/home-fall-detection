#!/usr/bin/env python
"""下载 URFD 图片（可断可续、适合「慢慢下载」）。

特点：
  - **可重复运行**：已存在的文件自动跳过，中断后重跑不浪费带宽
  - `--status`     只报告进度，不下载
  - `--max-minutes` 本次最多跑 N 分钟就收工（适合分多次慢慢下）
  - `--workers`    并发数（HF 对单文件较小场景，24~32 已接近上限）
  - 只下「关键点 pkl 里真正用到的帧」，不拉无关文件

用法：
    # 看进度
    python fetch_urfd_fast.py --status
    # 分次下载：每次最多 20 分钟
    python fetch_urfd_fast.py --workers 24 --max-minutes 20
    # 一次下完
    python fetch_urfd_fast.py --workers 24
"""
import argparse
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "minhy112/fall-detection-data"
LOCAL = Path(r"E:\MageVL\eval\urfd")


def list_targets(limit=None):
    """从已下好的关键点 pkl 里推出需要的帧，只下这些（避免下无关文件）。"""
    data = pickle.load(open(LOCAL / "poses" / "URFD_keypoints.pkl", "rb"))
    targets = []
    for rec in data:
        vid = rec["video_id"]
        leaf = vid.split("/")[-1]
        for t in range(rec["keypoints"].shape[0]):
            targets.append(f"raw/URFD/{vid}/{leaf}-{t+1:03d}.png")
    if limit:
        targets = targets[:limit]
    return targets


def local_path(rel):
    return LOCAL / rel.replace("/", "\\")


def report(targets):
    have = sum(1 for p in targets if local_path(p).is_file())
    print(f"目标 {len(targets)} 张 | 已有 {have} ({have/len(targets)*100:.1f}%) | "
          f"待下 {len(targets)-have}")
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 张（调试用）")
    ap.add_argument("--report-every", type=int, default=500)
    ap.add_argument("--status", action="store_true", help="只报进度")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="本次最多跑 N 分钟就收工（可重复运行，续着下）")
    args = ap.parse_args()

    targets = list_targets(args.limit)
    print(f"仓库 {REPO}", flush=True)
    have = report(targets)
    if args.status:
        print("\n（--status 模式，未下载任何文件）")
        return 0

    todo = [p for p in targets if not local_path(p).is_file()]
    if not todo:
        print("全部已存在，无需下载")
        return 0
    print(f"并发 {args.workers}，开始下载 {len(todo)} 张"
          + (f"，本次限时 {args.max_minutes} 分钟" if args.max_minutes else ""), flush=True)

    t0 = time.time()
    deadline = t0 + args.max_minutes * 60 if args.max_minutes else None
    done = fail = 0

    def fetch(rel):
        dest = local_path(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=REPO, filename=rel, repo_type="dataset", local_dir=str(LOCAL))
        return rel

    stopped_early = False
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, r): r for r in todo}
        for fu in as_completed(futs):
            done += 1
            try:
                fu.result()
            except Exception as e:
                fail += 1
                if fail <= 5:
                    print(f"  失败 {futs[fu]}: {type(e).__name__} {str(e)[:80]}", flush=True)
            if done % args.report_every == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                print(f"  {done}/{len(todo)}  失败{fail}  {rate:.1f} 张/秒  "
                      f"ETA {((len(todo)-done)/max(rate,1e-9))/60:.1f} 分钟", flush=True)
            if deadline and time.time() > deadline:
                stopped_early = True
                print(f"  达到限时 {args.max_minutes} 分钟，收工（已下 {done-fail} 张）", flush=True)
                for f in futs:
                    f.cancel()
                break

    el = (time.time() - t0) / 60
    print(f"\n本次：成功 {done-fail} / 失败 {fail} / 用时 {el:.1f} 分钟"
          + ("（限时中断，可重跑继续）" if stopped_early else ""), flush=True)
    report(targets)
    return 0


if __name__ == "__main__":
    sys.exit(main())

