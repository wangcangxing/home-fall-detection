#!/usr/bin/env python
"""训练 YOLOv10n 做姿态三分类（stand / sit / lie）。

用法：
    python train_yolov10n_posture.py --epochs 100 --imgsz 640
    python train_yolov10n_posture.py --smoke          # 冒烟测试：1 epoch + 极小样本

要点：
  - 数据集由 scripts/build_urfd_yolo.py 生成，**已按官方 video_splits 划分**，无帧级泄漏
  - 从 COCO 预训练的 yolov10n.pt 起步（迁移学习）
  - 训练完在 test 划分上评估，并把结果写到 runs/ 与 results/
"""
import argparse
import json
import os
import shutil
from pathlib import Path

DATA = Path(r"E:\MageVL\dataset\urfd_yolo\data.yaml")
PROJ = Path(r"E:\MageVL\runs")
REPO_RESULTS = Path(r"D:\program\模型优化\results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov10n.pt", help="预训练权重或 yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32,
                    help="冒烟测试显示 batch16 只用 2.8G/15.9G 显存，加大批次可提升吞吐")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="urfd_posture_v1")
    ap.add_argument("--data", default=str(DATA),
                    help="数据集 data.yaml；域适配微调时指向 E:\\MageVL\\dataset\\domain_yolo\\data.yaml")
    ap.add_argument("--smoke", action="store_true", help="冒烟测试：1 epoch 小样本")
    ap.add_argument("--resume", action="store_true",
                    help="**严格续训**：从 --model 指向的 last.pt 接着跑（epoch 与 LR 调度连续）。"
                         "ultralytics 会忽略本次传入的 epochs/name 等，改用该 run 目录里的 args.yaml")
    args = ap.parse_args()

    data = Path(args.data)
    if not data.is_file():
        raise SystemExit(f"找不到数据集配置：{data}\n请先运行 scripts/build_urfd_yolo.py 或 build_domain_yolo.py")

    from ultralytics import YOLO

    epochs = 1 if args.smoke else args.epochs
    model = YOLO(args.model)

    if args.resume:
        # 严格续训：ultralytics 从 last.pt 里读出原 args.yaml（epochs/name/数据集都在其中），
        # epoch 计数与 LR 调度接着上次走，**不会**重跑已完成的 epoch。
        print(f"严格续训：从 {args.model} 继续（epoch 与 LR 调度连续）", flush=True)
        res = model.train(resume=True)
    else:
        print(f"开始训练：model={args.model} epochs={epochs} imgsz={args.imgsz} "
              f"batch={args.batch}", flush=True)
        res = model.train(
            data=str(data),
            epochs=epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            project=str(PROJ),
            name=args.name,
            exist_ok=True,
            pretrained=True,
            seed=0,
            deterministic=True,
            patience=20,
            val=True,
            plots=True,
            verbose=True,
        )

    save_dir = Path(getattr(res, "save_dir", PROJ / args.name))
    print(f"\n训练完成，产物目录：{save_dir}", flush=True)

    # 在 test 划分上评估（数据集没有 test 划分时跳过，例如域适配数据集）
    best = save_dir / "weights" / "best.pt"
    metrics = None
    has_test = (data.parent / "images" / "test").is_dir()
    if best.is_file() and not args.smoke and has_test:
        m = YOLO(str(best))
        # workers 显式传小值：ultralytics 默认 8 个 dataloader 子进程会耗尽提交内存（坑点 #73）
        metrics = m.val(data=str(data), split="test", imgsz=args.imgsz, workers=args.workers)
        print("\n=== test 划分评估 ===", flush=True)
        try:
            print(metrics.results_dict, flush=True)
            # 逐类别指标
            names = m.names
            for i, c in names.items():
                print(f"  {c}: {metrics.box.ap50[i] if hasattr(metrics.box,'ap50') else '?'}", flush=True)
        except Exception as e:
            print("指标读取失败:", e, flush=True)
    elif not has_test:
        print(f"\n（数据集没有 test 划分，跳过内置评估：{data.parent / 'images' / 'test'}）", flush=True)

    # 汇总到仓库 results/（便于留痕与入库）
    # ⚠️ resume 时 CLI 的 data/epochs 只是脚本默认值，**会误导**；以 checkpoint 里的实际 args 为准
    act = getattr(res, "args", None)
    data_used = str(getattr(act, "data", data)) if act is not None else str(data)
    epochs_used = getattr(act, "epochs", epochs) if act is not None else epochs
    REPO_RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "data": data_used,
        "model": args.model,
        "epochs": epochs_used,
        "resume": args.resume,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "save_dir": str(save_dir),
        "best_weights": str(best) if best.is_file() else None,
        "smoke": args.smoke,
        "results_dict": getattr(metrics, "results_dict", None) if metrics else None,
    }
    out = REPO_RESULTS / f"train_{args.name}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n汇总已写入 {out}", flush=True)
    print(f"最佳权重：{best}", flush=True)


if __name__ == "__main__":
    main()
