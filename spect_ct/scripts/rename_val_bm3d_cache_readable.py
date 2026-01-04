#!/usr/bin/env python3
"""
Rename existing validation BM3D/thin caches from md5-key names to human-readable names,
WITHOUT recomputation.

We keep caches in:
  datasets/SPECT229/<patient>/bm3d_cache/

This script creates (or uses) a seed-scoped subdir:
  datasets/SPECT229/<patient>/bm3d_cache/seed{seed}/
and moves:
  <md5>_bm3d.npy  -> seed{seed}/{label}_bm3d.npy
  <md5>_thin.npy  -> seed{seed}/{label}_thin.npy

where md5 is computed exactly as used during validation:
  md5(f"{proj_file_path}__{label}")[:16]

Labels: 20s, x2, x3, x4, x5 (configurable).

Usage:
  python spect_ct/scripts/rename_val_bm3d_cache_readable.py --seed 123 --patients AnYufeng BaTunasong BaYasu BaiYukun
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple


def md5_16(s: str) -> str:
    return hashlib.md5(str(s).encode()).hexdigest()[:16]


def rename_one_patient(patient_dir: Path, seed: int, labels: List[str], overwrite: bool) -> Tuple[int, int, int]:
    """
    Returns: (moved, skipped, missing)
    """
    moved = skipped = missing = 0
    cache_root = patient_dir / "bm3d_cache"
    if not cache_root.exists():
        raise FileNotFoundError(cache_root)

    proj_files = list(patient_dir.glob("*_Proj4Filter.dat"))
    if not proj_files:
        raise FileNotFoundError(f"No *_Proj4Filter.dat under {patient_dir}")
    proj_file = proj_files[0]

    out_dir = cache_root / f"seed{int(seed)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for lab in labels:
        key = md5_16(f"{proj_file}__{lab}")
        src_bm3d = cache_root / f"{key}_bm3d.npy"
        src_thin = cache_root / f"{key}_thin.npy"
        dst_bm3d = out_dir / f"{lab}_bm3d.npy"
        dst_thin = out_dir / f"{lab}_thin.npy"

        # bm3d
        if dst_bm3d.exists() and not overwrite:
            skipped += 1
        else:
            if not src_bm3d.exists():
                missing += 1
            else:
                if dst_bm3d.exists():
                    dst_bm3d.unlink()
                src_bm3d.rename(dst_bm3d)
                moved += 1

        # thin (optional but expected)
        if dst_thin.exists() and not overwrite:
            skipped += 1
        else:
            if not src_thin.exists():
                missing += 1
            else:
                if dst_thin.exists():
                    dst_thin.unlink()
                src_thin.rename(dst_thin)
                moved += 1

    return moved, skipped, missing


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", type=str, default="datasets/SPECT229")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--patients", type=str, nargs="+", default=["AnYufeng", "BaTunasong", "BaYasu", "BaiYukun"])
    p.add_argument("--labels", type=str, nargs="+", default=["20s", "x2", "x3", "x4", "x5"])
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    root = Path(args.dataroot)
    moved = skipped = missing = 0
    for name in args.patients:
        d = root / name
        m, s, ms = rename_one_patient(d, seed=int(args.seed), labels=list(args.labels), overwrite=bool(args.overwrite))
        moved += m
        skipped += s
        missing += ms
        print(f"[{name}] moved={m} skipped={s} missing={ms}")

    print(f"TOTAL moved={moved} skipped={skipped} missing={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


