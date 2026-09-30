#!/usr/bin/env python
"""把实测数据里的**本机绝对路径**替换成占位符（脱敏），数值一概不动。

为什么先解析 JSON 再处理字符串：JSON 里 Windows 路径是双重反斜杠转义
（`"E:\\\\MageVL\\\\Mage-VL"`），直接在原文上做正则会既容易漏、又可能改坏转义。
`json.load` 之后拿到的是真实的单反斜杠路径，替换干净且不会破坏文件。

替换规则（保持可读、可复现）：
    E:\\MageVL\\...        -> <ASSETS>/...
    D:\\program\\模型优化  -> <REPO>
    C:\\pagefile.sys       -> <PAGEFILE>
    C:\\Users\\<name>\\... -> <HOME>/...
    其余 X:\\...           -> <PATH>/...（兜底）

用法：
  python scripts/compress/desensitize_results.py            # 默认处理 results/compress_*
  python scripts/compress/desensitize_results.py --check    # 只报告，不改文件
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

RULES = [
    (re.compile(r"[A-Za-z]:[\\/]+MageVL[\\/]?", re.I), "<ASSETS>/"),
    (re.compile(r"[A-Za-z]:[\\/]+program[\\/]+模型优化[\\/]?", re.I), "<REPO>/"),
    (re.compile(r"[A-Za-z]:[\\/]+pagefile\.sys", re.I), "<PAGEFILE>"),
    (re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\"]+", re.I), "<HOME>"),
    # 兜底：其余本机盘符绝对路径（至少两级，避免误伤 "C:\\" 这类孤立片段）
    (re.compile(r"[A-Za-z]:[\\/](?:[^\\/\"\s]+[\\/])+[^\\/\"\s]*"), "<PATH>/"),
]


def scrub(text: str) -> tuple[str, int]:
    n = 0
    for rx, rep in RULES:
        text, k = rx.subn(rep, text)
        n += k
    return text, n


def scrub_obj(o):
    if isinstance(o, str):
        return scrub(o)
    if isinstance(o, list):
        out, total = [], 0
        for v in o:
            nv, k = scrub_obj(v)
            out.append(nv)
            total += k
        return out, total
    if isinstance(o, dict):
        out, total = {}, 0
        for k, v in o.items():
            nv, n = scrub_obj(v)
            out[k] = nv
            total += n
        return out, total
    return o, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只报告不修改")
    ap.add_argument("--glob", default="compress_*", help="处理哪些文件（默认 compress_*）")
    args = ap.parse_args()

    files = sorted(p for p in RESULTS.glob(args.glob) if p.is_file())
    if not files:
        print(f"{RESULTS} 下没有匹配 {args.glob} 的文件")
        return

    grand = 0
    for p in files:
        raw = p.read_text(encoding="utf-8")
        if p.suffix == ".json":
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"  ! 跳过（JSON 解析失败）：{p.name} -> {e}")
                continue
            new_obj, n = scrub_obj(obj)
            new = json.dumps(new_obj, ensure_ascii=False, indent=2) + "\n"
        else:
            new, n = scrub(raw)
        grand += n
        if n == 0:
            print(f"  --  {p.name}：无需替换")
            continue
        print(f"  {'??' if args.check else 'OK'}  {p.name}：替换 {n} 处")
        if not args.check and new != raw:
            p.write_text(new, encoding="utf-8")
    print(f"\n合计替换 {grand} 处（{'仅检查，未写文件' if args.check else '已写回'}）")

    # 复查：是否还有残留的本机绝对路径
    left = []
    for p in files:
        raw = p.read_text(encoding="utf-8")
        for m in re.finditer(r"[A-Za-z]:[\\/]{1,2}[^\\/\",\s]+", raw):
            left.append((p.name, m.group(0)))
    if left:
        print(f"⚠️ 仍有 {len(left)} 处疑似绝对路径（前 10）：")
        for name, s in left[:10]:
            print(f"    {name}: {s}")
    else:
        print("✅ 复查：未发现残留的本机绝对路径")


if __name__ == "__main__":
    main()
