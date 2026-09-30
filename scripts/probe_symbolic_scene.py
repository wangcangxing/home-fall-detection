#!/usr/bin/env python
"""probe L：演示「YOLO → 符号化场景」这一层能给出多少语义。

用 COCO 预训练检测器（80 类）对若干图输出「物体类别 + 框几何」，
看它能不能构成一份符号化的场景描述（供下游小 LLM 汇总成自然语言）。
"""
from pathlib import Path

import torch

CCTV = Path(r"E:\MageVL\eval\fall-CCTV_Incident_Dataset_Fall_Lying_Down_Detection\laying_dataset\images")
MODEL = r"E:\MageVL\Mage-VL"
OUT = Path(r"E:\MageVL\baseline")

# COCO 80 类里与家庭场景相关的
FAMILY = {1: "person 人", 18: "dog 狗", 17: "cat 猫", 57: "couch 沙发", 59: "bed 床",
          56: "chair 椅子", 60: "dining table 桌子", 62: "tv 电视", 58: "potted plant 盆栽",
          61: "toilet 马桶", 15: "bench 长凳", 13: "stop sign", 3: "car 车"}


def main():
    import cv2
    from PIL import Image
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
    from torchvision.transforms.functional import pil_to_tensor

    w = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=w).eval().cuda()
    cats = w.meta["categories"]

    cases = []
    for f in sorted(CCTV.glob("laying*.png"))[:2]:
        cases.append(("CCTV 躺倒图 " + f.name, f))
    cases.append(("examples/dog.jpg", Path(MODEL) / "examples" / "dog.jpg"))
    cap = cv2.VideoCapture(str(Path(MODEL) / "examples" / "soccer-broadcast.mp4"))
    for i in (100, 400):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = cap.read()
        if ok:
            p = OUT / f"_sym_{i}.jpg"
            Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).save(p, quality=90)
            cases.append((f"soccer 帧 {i}", p))
    cap.release()

    lines = ["=== probe L：YOLO 输出的「符号化场景」示例（COCO 80 类）==="]
    for tag, path in cases:
        img = Image.open(path).convert("RGB")
        t = pil_to_tensor(img).float().cuda() / 255.0
        with torch.inference_mode():
            pred = model([t])[0]
        keep = pred["scores"] > 0.6
        H, W = img.size[1], img.size[0]
        objs = []
        for box, lab, sc in zip(pred["boxes"][keep], pred["labels"][keep], pred["scores"][keep]):
            x0, y0, x1, y1 = [float(v) for v in box]
            bw, bh = x1 - x0, y1 - y0
            name = FAMILY.get(int(lab), cats[int(lab)])
            objs.append(f"{name}(score={sc:.2f}, aspect={bw/max(bh,1e-6):.2f}, "
                        f"bottom={y1/H:.2f}, area={bw*bh/(W*H)*100:.1f}%)")
        lines.append(f"\n--- {tag} ---")
        lines.append("  检出 " + str(len(objs)) + " 个物体：")
        for o in sorted(objs)[:8]:
            lines.append("    " + o)
        if not objs:
            lines.append("    （无高置信检出）")
    lines.append("\n说明：这一堆「类别+几何」就是送给小 LLM 的符号化输入。"
                 "语义来自检测器的类别词表，不来自任何视觉语言模型。")
    (OUT / "probe_symbolic_scene.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
