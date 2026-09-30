#!/usr/bin/env python
"""probe B：Mage-VL 的自定义代码能否接受 bitsandbytes 4bit 量化加载。

只回答三个问题：
  1. `BitsAndBytesConfig(load_in_4bit=True)` + trust_remote_code 能不能加载起来？
  2. 加载后权重显存降到多少？有没有真的替换成 4bit 层？
  3. 量化后图像推理结果是否仍与 bf16 基线一致？16 帧视频能不能过？
"""
import difflib
import time
from pathlib import Path

import torch

MODEL = r"E:\MageVL\Mage-VL"
VIDEO = rf"{MODEL}\examples\soccer-broadcast.mp4"
DOG = rf"{MODEL}\examples\dog.jpg"
OUT = Path(r"E:\MageVL\baseline")
OUT.mkdir(parents=True, exist_ok=True)

BF16_IMAGE_ANSWER = (
    "The image depicts a dog sitting on a patterned rug. The dog appears to be a Border Collie, "
    "characterized by its thick, fluffy coat with a mix of white, black, and brown fur."
)


def log(msg, buf):
    print(msg, flush=True)
    buf.append(msg)


def sample_video(path, num_frames):
    import cv2
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(path)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, fc - 1, min(num_frames, fc), dtype=int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def main():
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()

    buf = []
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, trust_remote_code=True, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.bfloat16,
        ).eval()
    except Exception as e:
        log(f"加载失败: {type(e).__name__}: {str(e)[:600]}", buf)
        (OUT / "probe_bnb4bit.txt").write_text("\n".join(buf) + "\n", encoding="utf-8")
        return
    log(f"加载成功，耗时 {time.time()-t0:.1f}s；dtype={model.dtype}；device={model.device}", buf)
    log(f"权重显存 alloc={torch.cuda.memory_allocated()/1024**3:.2f}GB "
        f"reserved={torch.cuda.memory_reserved()/1024**3:.2f}GB", buf)

    # 统计被替换的 4bit 层
    from collections import Counter
    kinds = Counter(type(m).__name__ for m in model.modules())
    log("模块类型计数（Top 8）: " + str(kinds.most_common(8)), buf)
    n_lin = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    log(f"剩余普通 nn.Linear: {n_lin}", buf)

    def run(media_kwargs, question, media_type, tag, max_new_tokens=128):
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": media_type}, {"type": "text", "text": question}]}],
            tokenize=False, add_generation_prompt=True,
        )
        try:
            inputs = processor(text=[text], return_tensors="pt", padding=True, **media_kwargs)
            inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            L = int(inputs["input_ids"].shape[1])
            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            dt = time.time() - t1
            ans = processor.tokenizer.decode(out[0, L:], skip_special_tokens=True).strip()
            pk = torch.cuda.max_memory_allocated() / 1024**3
            log(f"[{tag}] OK prompt={L} peak_alloc={pk:.2f}GB gen={dt:.1f}s", buf)
            log(f"    回答: {ans[:300]}", buf)
            return ans
        except torch.OutOfMemoryError as e:
            log(f"[{tag}] OOM {str(e).splitlines()[0][:100]}", buf)
        except Exception as e:
            log(f"[{tag}] ERR {type(e).__name__}: {str(e)[:200]}", buf)
        torch.cuda.empty_cache()
        return None

    from PIL import Image as PILImage
    ans = run({"images": [PILImage.open(DOG).convert("RGB")]},
              "Describe this image in detail.", "image", "image-4bit")
    if ans:
        log(f"    与 bf16 基线首段的相似度 = {difflib.SequenceMatcher(None, BF16_IMAGE_ANSWER, ans[:len(BF16_IMAGE_ANSWER)]).ratio():.3f}", buf)

    for n in (8, 16, 32):
        torch.cuda.empty_cache()
        run({"videos": [sample_video(VIDEO, n)]}, "Describe this video.", "video", f"video-{n}帧-4bit")

    (OUT / "probe_bnb4bit.txt").write_text("\n".join(buf) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
