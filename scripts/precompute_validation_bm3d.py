#!/usr/bin/env python3
"""
Precompute BM3D denoising for validation set with Poisson-thinned low-dose projections.

This script:
1. Loads the 20s original projections for validation patients (indices 0-3)
2. Performs Poisson thinning to generate low-dose projections (x2, x3, x4, x5)
3. Applies BM3D denoising (via Anscombe transform) to each low-dose projection
4. Saves the results to datasets/SPECT229/<patient>/bm3d_cache/

All experiments can then reuse these cached BM3D results during validation,
avoiding redundant computation (BM3D is deterministic and very slow).

Usage:
    python scripts/precompute_validation_bm3d.py [--seed 123] [--overwrite]
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
import multiprocessing as mp
import zlib

# Add BasicSR to path

from prime.pipeline.io import load_projection_u16
from prime.pipeline.val_cache import bm3d_anscombe_denoise, poisson_thin_binomial


def _seed_dir(cache_root: Path, seed: int) -> Path:
    # Keep different seeds separated without hashes.
    return cache_root / f"seed{int(seed)}"


def _thin_seed(*, base_seed: int, patient_name: str, label: str) -> int:
    token = f"{patient_name}::{label}"
    return (int(base_seed) + int(zlib.adler32(token.encode("utf-8")))) % (2**32)


def main():
    parser = argparse.ArgumentParser(description="Precompute validation BM3D cache")
    parser.add_argument("--dataroot", type=str, default="datasets/SPECT229", help="Path to SPECT229 dataset root")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for Poisson thinning (default: 123)")
    parser.add_argument(
        "--thin-factors", type=int, nargs="+", default=[2, 3, 4, 5], help="Thinning factors to precompute (default: 2 3 4 5)"
    )
    parser.add_argument(
        "--patients",
        type=str,
        nargs="+",
        default=None,
        help="Optional: patient folder names under dataroot (overrides --val-indices).",
    )
    parser.add_argument("--val-indices", type=int, nargs="+", default=[0, 1, 2, 3], help="Validation patient indices")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    parser.add_argument(
        "--thin-only",
        action="store_true",
        help="Only generate and save {label}_thin.npy (skip BM3D entirely).",
    )
    parser.add_argument("--sigma-psd", type=float, default=1.0, help="BM3D noise std in Anscombe domain")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers for BM3D")
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    if not dataroot.exists():
        print(f"❌ Dataroot not found: {dataroot}")
        return 1

    all_patients = sorted([p for p in dataroot.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if not all_patients:
        print(f"❌ No patient directories found in {dataroot}")
        return 1

    num_workers = int(args.num_workers) if args.num_workers is not None else mp.cpu_count()

    print(f"📂 Found {len(all_patients)} patient directories in {dataroot}")
    print(f"🎲 Random seed: {args.seed}")
    print(f"📊 Thinning factors: {args.thin_factors}")
    if args.patients:
        print(f"✅ Patients: {args.patients}")
    else:
        print(f"✅ Validation indices: {args.val_indices}")
    print(f"🔧 BM3D workers: {num_workers} (CPU count: {mp.cpu_count()})")
    if bool(args.thin_only):
        print("⚡ thin-only: enabled (BM3D will be skipped)")
    print()

    total_computed = 0
    total_skipped = 0

    if args.patients:
        chosen_patients: list[Path] = []
        missing: list[str] = []
        all_by_name = {p.name: p for p in all_patients}
        for name in args.patients:
            p = all_by_name.get(str(name))
            if p is None:
                missing.append(str(name))
            else:
                chosen_patients.append(p)
        if missing:
            print(f"❌ Some patients not found under {dataroot}: {missing}")
            return 1
        patient_plan = [("name", p) for p in chosen_patients]
    else:
        patient_plan = [("idx", int(i)) for i in args.val_indices]

    k_list = [1] + list(args.thin_factors)
    total_tasks = len(patient_plan) * len(k_list)

    with tqdm(total=total_tasks, desc="Overall", unit="task", position=0, leave=True) as pbar_outer:
        for kind, token in patient_plan:
            if kind == "idx":
                patient_idx = int(token)
                if patient_idx >= len(all_patients):
                    print(f"⚠️  Skipping index {patient_idx} (only {len(all_patients)} patients available)")
                    pbar_outer.update(len(k_list))
                    continue
                patient_dir = all_patients[patient_idx]
            else:
                patient_dir = Path(token)
                patient_idx = -1

            patient_name = patient_dir.name

            proj_files = list(patient_dir.glob("*_Proj4Filter.dat"))
            if not proj_files:
                print(f"⚠️  No *_Proj4Filter.dat found in {patient_dir}")
                pbar_outer.update(len(k_list))
                continue

            proj_file = proj_files[0]
            if patient_idx >= 0:
                pbar_outer.set_description(f"Patient {patient_idx:03d}: {patient_name[:20]}")
                print(f"\n👤 Patient [{patient_idx:03d}]: {patient_name}")
            else:
                pbar_outer.set_description(f"Patient: {patient_name[:24]}")
                print(f"\n👤 Patient: {patient_name}")

            try:
                proj_20s = load_projection_u16(proj_file, views=60, h=128, w=128)
            except Exception as e:
                print(f"   ❌ Failed to load {proj_file}: {e}")
                pbar_outer.update(len(k_list))
                continue

            print(f"   📥 Loaded 20s projection: shape={proj_20s.shape}, sum={proj_20s.sum()/1e4:.1f}W")

            cache_root = patient_dir / "bm3d_cache"
            cache_dir = _seed_dir(cache_root, int(args.seed))
            cache_dir.mkdir(parents=True, exist_ok=True)

            for k in k_list:
                label = "20s" if int(k) == 1 else f"x{int(k)}"
                cache_file = cache_dir / f"{label}_bm3d.npy"
                thin_file = cache_dir / f"{label}_thin.npy"

                if bool(args.thin_only):
                    if thin_file.exists() and (not args.overwrite):
                        print(f"   ⏭️  {label}: thin cache exists, skipping (use --overwrite to recompute)")
                        total_skipped += 1
                        pbar_outer.update(1)
                        continue
                else:
                    if cache_file.exists() and thin_file.exists() and (not args.overwrite):
                        print(f"   ⏭️  {label}: cache exists, skipping (use --overwrite to recompute)")
                        total_skipped += 1
                        pbar_outer.update(1)
                        continue

                if int(k) == 1:
                    proj_low = proj_20s.astype(np.float32, copy=False)
                    print(f"   🔬 {label}: Using original 20s (no thinning)")
                else:
                    print(f"   🔬 {label}: Thinning...")
                    seed_k = _thin_seed(base_seed=int(args.seed), patient_name=patient_name, label=label)
                    proj_low = poisson_thin_binomial(proj_20s, int(k), seed=seed_k)
                    exp_sum = (proj_20s.sum() / float(k)) / 1e4
                    print(f"        Thinned: sum={proj_low.sum()/1e4:.1f}W (expected: {exp_sum:.1f}W)")

                try:
                    np.save(thin_file, proj_low.astype(np.float32, copy=False))
                except Exception as e:
                    print(f"   ⚠️  {label}: Failed to save thin cache: {e}")

                if bool(args.thin_only):
                    total_computed += 1
                    print(f"   ✅ {label}: thin saved ({thin_file.name})")
                    pbar_outer.update(1)
                    continue

                print(f"   🧹 {label}: BM3D denoising (parallel: {num_workers} workers)...")
                proj_bm3d = bm3d_anscombe_denoise(proj_low, sigma_psd=float(args.sigma_psd), num_workers=num_workers)

                print(f"   💾 {label}: Saving to {cache_file.name}")
                np.save(cache_file, proj_bm3d)
                total_computed += 1
                print(f"   ✅ {label}: Done (size: {cache_file.stat().st_size / 1e6:.1f} MB)")
                pbar_outer.update(1)

            print()

    print("\n🎉 Precomputation complete!")
    print(f"   Computed: {total_computed}")
    print(f"   Skipped:  {total_skipped}")
    print()
    print("💡 To use these caches during validation, add to your YAML:")
    print("   val:")
    print("     bm3d_cache_root: datasets/SPECT229/{patient}/bm3d_cache")
    print(f"     bm3d_cache_seed: {args.seed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
