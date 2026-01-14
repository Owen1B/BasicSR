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
    python spect_ct/scripts/precompute_validation_bm3d.py [--seed 123] [--overwrite]
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
import multiprocessing as mp

# Add BasicSR to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from basicsr.utils.anscombe import anscombe_forward, anscombe_inverse_unbiased


def poisson_thin(counts: np.ndarray, factor: int, rng: np.random.Generator) -> np.ndarray:
    """
    Poisson thinning: generate a low-dose sample from high-dose counts.

    For each observed count y, thinning one sub-exposure is:
        y_sub ~ Binomial(y, 1/factor)
    which matches the multinomial split marginal and is vectorizable.

    Args:
        counts: (60, H, W) count-domain projection
        factor: thinning factor (2, 3, 4, 5)
        rng: numpy random generator

    Returns:
        (60, H, W) thinned projection
    """
    counts_i = np.clip(counts, 0, None).astype(np.int64, copy=False)
    p = 1.0 / float(factor)
    thinned_i = rng.binomial(counts_i, p).astype(np.int64, copy=False)
    return thinned_i.astype(np.float32, copy=False)


def _bm3d_denoise_single_view(args):
    """
    BM3D denoising for a single view (worker function for multiprocessing).

    Args:
        args: tuple of (view_idx, view_data, sigma_psd)

    Returns:
        tuple of (view_idx, denoised_view)
    """
    try:
        import bm3d
    except ImportError:
        raise ImportError("bm3d package not found. Install: pip install bm3d")

    view_idx, view_data, sigma_psd = args
    view_data = np.clip(view_data.astype(np.float32, copy=False), 0.0, None)

    # Anscombe forward
    anscombe_img = anscombe_forward(view_data)
    # BM3D in Anscombe domain
    # In Anscombe domain, noise is approximately N(0, 1), so sigma_psd should be ~1.0
    denoised_anscombe = bm3d.bm3d(
        anscombe_img,
        sigma_psd=float(sigma_psd),
        stage_arg=bm3d.BM3DStages.ALL_STAGES,
    )
    # Anscombe inverse
    denoised = anscombe_inverse_unbiased(denoised_anscombe)

    return view_idx, denoised


def bm3d_denoise_projection(proj: np.ndarray, sigma_psd: float = 1.0, num_workers: int = None) -> np.ndarray:
    """
    BM3D denoising via Anscombe transform (CPU-based, parallelized across views).

    Args:
        proj: (60, H, W) count-domain projection
        sigma_psd: BM3D noise std in Anscombe domain (recommended: 1.0 for Anscombe)
        num_workers: Number of parallel workers (default: cpu_count)

    Returns:
        (60, H, W) denoised projection in count domain
    """
    proj = np.clip(proj.astype(np.float32, copy=False), 0.0, None)
    denoised = np.zeros_like(proj, dtype=np.float32)

    if num_workers is None:
        num_workers = mp.cpu_count()

    # Prepare arguments for each view
    args_list = [(v, proj[v], sigma_psd) for v in range(proj.shape[0])]

    # Parallel processing with progress bar
    with mp.Pool(processes=num_workers) as pool:
        for view_idx, denoised_view in tqdm(
            pool.imap_unordered(_bm3d_denoise_single_view, args_list),
            total=len(args_list),
            desc="  BM3D",
            leave=False,
            dynamic_ncols=True,
        ):
            denoised[view_idx] = denoised_view

    return denoised


def _seed_dir(cache_root: Path, seed: int) -> Path:
    # Keep different seeds separated without hashes.
    return cache_root / f"seed{int(seed)}"


def main():
    parser = argparse.ArgumentParser(description="Precompute validation BM3D cache")
    parser.add_argument("--dataroot", type=str, default="datasets/SPECT229",
                        help="Path to SPECT229 dataset root")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for Poisson thinning (default: 123)")
    parser.add_argument("--thin-factors", type=int, nargs="+", default=[2, 3, 4, 5],
                        help="Thinning factors to precompute (default: 2 3 4 5)")
    parser.add_argument(
        "--patients",
        type=str,
        nargs="+",
        default=None,
        help="Optional: patient folder names under dataroot (overrides --val-indices). Example: --patients BaYasu BaTunasong",
    )
    parser.add_argument("--val-indices", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="Validation patient indices (default: 0 1 2 3)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing cache files")
    parser.add_argument(
        "--thin-only",
        action="store_true",
        help="Only generate and save {label}_thin.npy (skip BM3D entirely). Useful for fast mp4 thinning expansion.",
    )
    parser.add_argument("--sigma-psd", type=float, default=1.0,
                        help="BM3D noise std in Anscombe domain (default: 1.0, recommended for Anscombe)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers for BM3D (default: cpu_count)")
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    if not dataroot.exists():
        print(f"❌ Dataroot not found: {dataroot}")
        return 1

    # Find all patient directories
    all_patients = sorted([p for p in dataroot.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if not all_patients:
        print(f"❌ No patient directories found in {dataroot}")
        return 1

    num_workers = args.num_workers if args.num_workers is not None else mp.cpu_count()

    print(f"📂 Found {len(all_patients)} patient directories in {dataroot}")
    print(f"🎲 Random seed: {args.seed}")
    print(f"📊 Thinning factors: {args.thin_factors}")
    if args.patients is not None and len(args.patients) > 0:
        print(f"✅ Patients: {args.patients}")
    else:
    print(f"✅ Validation indices: {args.val_indices}")
    print(f"🔧 BM3D workers: {num_workers} (CPU count: {mp.cpu_count()})")
    if bool(args.thin_only):
        print("⚡ thin-only: enabled (BM3D will be skipped)")
    print()

    total_computed = 0
    total_skipped = 0

    # Decide patient list
    if args.patients is not None and len(args.patients) > 0:
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

    # Calculate total tasks for progress bar.
    # Note: we always process k_list = [1] + thin_factors.
    total_tasks = len(patient_plan) * (len(args.thin_factors) + 1)

    # Outer progress bar for overall progress
    with tqdm(total=total_tasks, desc="Overall", unit="task", position=0, leave=True) as pbar_outer:
        for kind, token in patient_plan:
            if kind == "idx":
                patient_idx = int(token)
            if patient_idx >= len(all_patients):
                print(f"⚠️  Skipping index {patient_idx} (only {len(all_patients)} patients available)")
                    pbar_outer.update(len(args.thin_factors) + 1)
                continue
            patient_dir = all_patients[patient_idx]
            else:
                patient_dir = Path(token)
                patient_idx = -1
            patient_name = patient_dir.name

            # Find 20s projection file
            proj_files = list(patient_dir.glob("*_Proj4Filter.dat"))
            if not proj_files:
                print(f"⚠️  No *_Proj4Filter.dat found in {patient_dir}")
                pbar_outer.update(len(args.thin_factors))
                continue

            proj_file = proj_files[0]
            if patient_idx >= 0:
            pbar_outer.set_description(f"Patient {patient_idx:03d}: {patient_name[:20]}")
            print(f"\n👤 Patient [{patient_idx:03d}]: {patient_name}")
            else:
                pbar_outer.set_description(f"Patient: {patient_name[:24]}")
                print(f"\n👤 Patient: {patient_name}")

            # Load 20s projection
            try:
                proj_20s = np.fromfile(proj_file, dtype=np.uint16).reshape(60, 128, 128).astype(np.float32)
            except Exception as e:
                print(f"   ❌ Failed to load {proj_file}: {e}")
                pbar_outer.update(len(args.thin_factors))
                continue

            print(f"   📥 Loaded 20s projection: shape={proj_20s.shape}, sum={proj_20s.sum()/1e4:.1f}W")

            # Create BM3D cache directory (seed-scoped, no hashes)
            cache_root = patient_dir / "bm3d_cache"
            cache_dir = _seed_dir(cache_root, int(args.seed))
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Deterministic RNG per patient (match validation MP4 thinning logic)
            rng = np.random.default_rng(args.seed)

            # Always precompute 20s BM3D too (label '20s')
            k_list = [1] + list(args.thin_factors)

            # Process each thinning factor (include 20s label)
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
                if cache_file.exists() and thin_file.exists() and not args.overwrite:
                    print(f"   ⏭️  {label}: Cache exists, skipping (use --overwrite to recompute)")
                        total_skipped += 1
                        pbar_outer.update(1)
                        continue
                    total_skipped += 1
                    pbar_outer.update(1)
                    continue

                if int(k) == 1:
                    proj_low = proj_20s.astype(np.float32, copy=False)
                    print(f"   🔬 {label}: Using original 20s (no thinning)")
                else:
                    print(f"   🔬 {label}: Thinning...")
                    proj_low = poisson_thin(proj_20s, int(k), rng)
                    print(f"        Thinned: sum={proj_low.sum()/1e4:.1f}W (expected: {(proj_20s.sum()/float(k))/1e4:.1f}W)")

                # Save the exact thinned projection used for BM3D (so future validation can be perfectly aligned)
                try:
                    np.save(thin_file, proj_low.astype(np.float32, copy=False))
                except Exception as e:
                    print(f"   ⚠️  {label}: Failed to save thinned projection cache: {e}")

                if bool(args.thin_only):
                    total_computed += 1
                    print(f"   ✅ {label}: thin saved ({thin_file.name})")
                else:
                print(f"   🧹 {label}: BM3D denoising (parallel: {num_workers} workers)...")
                proj_bm3d = bm3d_denoise_projection(proj_low, sigma_psd=args.sigma_psd, num_workers=num_workers)

                print(f"   💾 {label}: Saving to {cache_file.name}")
                np.save(cache_file, proj_bm3d)
                total_computed += 1
                print(f"   ✅ {label}: Done (size: {cache_file.stat().st_size / 1e6:.1f} MB)")
                pbar_outer.update(1)

            print()

        print(f"\n🎉 Precomputation complete!")
        print(f"   Computed: {total_computed}")
        print(f"   Skipped:  {total_skipped}")
        print()
        print(f"💡 To use these caches during validation, add to your YAML:")
        print(f"   val:")
        print(f"     bm3d_cache_root: datasets/SPECT229/{{patient}}/bm3d_cache")
        print(f"     bm3d_cache_seed: {args.seed}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

