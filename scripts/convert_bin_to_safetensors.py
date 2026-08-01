#!/usr/bin/env python3
"""
将 Janus-Pro / checkpoint 目录里的 pytorch_model.bin 转换为 model.safetensors。

背景:
  transformers >= 4.51 因 CVE-2025-32434 强制要求 torch >= 2.6 才能用 torch.load
  加载 .bin，但当前评测环境 torch=2.5.1 会直接报 ValueError。转成 safetensors
  后 transformers 会优先加载它，绕过 CVE 校验。

用法:
  # 单个目录
  python scripts/convert_bin_to_safetensors.py <ckpt_dir>

  # 批量（glob 支持通配符）
  python scripts/convert_bin_to_safetensors.py \
      src/t2i-r1/reward_weight/Janus-Pro-1B \
      src/t2i-r1/src/outputs/*/checkpoint-*

  # 转完删掉原始 .bin（省磁盘）
  python scripts/convert_bin_to_safetensors.py --delete-bin <ckpt_dir>
"""

import argparse
import glob
import os
import sys
from pathlib import Path


def convert_sharded(ckpt_dir: Path, delete_bin: bool, force: bool) -> bool:
    import json
    import re

    import torch
    from safetensors.torch import save_file

    idx_src = ckpt_dir / "pytorch_model.bin.index.json"
    idx_dst = ckpt_dir / "model.safetensors.index.json"

    with open(idx_src) as f:
        idx = json.load(f)

    weight_map = idx["weight_map"]
    shards = sorted(set(weight_map.values()))

    def bin_to_st(name: str) -> str:
        m = re.match(r"pytorch_model-(\d+)-of-(\d+)\.bin$", name)
        if m:
            return f"model-{m.group(1)}-of-{m.group(2)}.safetensors"
        return name.replace("pytorch_model", "model").replace(".bin", ".safetensors")

    if idx_dst.exists() and not force:
        print(f"[SKIP] {ckpt_dir} : model.safetensors.index.json 已存在（--force 覆盖）")
        return False

    print(f"[SHARDED] {ckpt_dir}  shards={len(shards)}")

    for shard in shards:
        src = ckpt_dir / shard
        dst = ckpt_dir / bin_to_st(shard)
        if dst.exists() and not force:
            print(f"  [SKIP] {dst.name} 已存在")
            continue
        if not src.exists():
            print(f"  [WARN] {src.name} 不存在")
            continue
        print(f"  [LOAD] {src.name}  ({src.stat().st_size/1e9:.2f} GB)")
        sd = torch.load(str(src), map_location="cpu", weights_only=True)
        sd = {k: v.contiguous() for k, v in sd.items()}
        save_file(sd, str(dst), metadata={"format": "pt"})
        print(f"  [SAVE] {dst.name}  ({dst.stat().st_size/1e9:.2f} GB)")
        del sd

    new_weight_map = {k: bin_to_st(v) for k, v in weight_map.items()}
    new_idx = {"metadata": idx.get("metadata", {}), "weight_map": new_weight_map}
    with open(idx_dst, "w") as f:
        json.dump(new_idx, f, indent=2)
    print(f"[SAVE] {idx_dst.name}")

    if delete_bin:
        for shard in shards:
            p = ckpt_dir / shard
            if p.exists():
                p.unlink()
                print(f"[DEL ] {p.name}")
        idx_src.unlink()
        print(f"[DEL ] {idx_src.name}")

    return True


def convert_one(ckpt_dir: Path, delete_bin: bool = False, force: bool = False) -> bool:
    """Return True if conversion happened, False if skipped."""
    src = ckpt_dir / "pytorch_model.bin"
    dst = ckpt_dir / "model.safetensors"
    idx = ckpt_dir / "pytorch_model.bin.index.json"

    if idx.exists():
        return convert_sharded(ckpt_dir, delete_bin=delete_bin, force=force)
    if not src.exists():
        print(f"[SKIP] {ckpt_dir} : no pytorch_model.bin")
        return False
    if dst.exists() and not force:
        print(f"[SKIP] {ckpt_dir} : model.safetensors 已存在（--force 覆盖）")
        return False

    import torch
    from safetensors.torch import save_file

    print(f"[LOAD] {src}  ({src.stat().st_size/1e9:.2f} GB)")
    sd = torch.load(str(src), map_location="cpu", weights_only=True)
    print(f"[TENS] {len(sd)} tensors")
    sd = {k: v.contiguous() for k, v in sd.items()}

    save_file(sd, str(dst), metadata={"format": "pt"})
    print(f"[SAVE] {dst}  ({dst.stat().st_size/1e9:.2f} GB)")

    if delete_bin:
        src.unlink()
        print(f"[DEL ] {src}")

    return True


def expand_targets(patterns) -> list:
    dirs = []
    for pat in patterns:
        matches = glob.glob(pat)
        if not matches:
            print(f"[WARN] no match for pattern: {pat}", file=sys.stderr)
            continue
        for m in matches:
            p = Path(m)
            if p.is_dir():
                dirs.append(p)
            elif p.is_file() and p.name == "pytorch_model.bin":
                dirs.append(p.parent)
            else:
                print(f"[WARN] not a dir or bin: {m}", file=sys.stderr)
    return sorted(set(dirs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="checkpoint 目录或 glob 模式（可多个）")
    ap.add_argument("--delete-bin", action="store_true", help="转换成功后删除原始 pytorch_model.bin")
    ap.add_argument("--force", action="store_true", help="即使 model.safetensors 已存在也重新转换")
    args = ap.parse_args()

    dirs = expand_targets(args.targets)
    if not dirs:
        print("no valid checkpoint directories found", file=sys.stderr)
        sys.exit(1)

    print(f"Targets ({len(dirs)}):")
    for d in dirs:
        print(f"  - {d}")
    print()

    ok, skipped, failed = 0, 0, 0
    for d in dirs:
        try:
            if convert_one(d, delete_bin=args.delete_bin, force=args.force):
                ok += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"[FAIL] {d} : {e}")
            failed += 1

    print()
    print(f"Summary: converted={ok}  skipped={skipped}  failed={failed}")
    sys.exit(0 if failed == 0 else 2)


if __name__ == "__main__":
    main()
