#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_patient_from_basename(basename: str) -> str:
    if "_pair" in basename:
        return basename.split("_pair", 1)[0]
    return basename.split("_", 1)[0]


def load_mu_map_zyx(mu_path: Path) -> np.ndarray:
    arr = np.fromfile(str(mu_path), dtype=np.float32)
    if arr.size == 128 * 128 * 128:
        return arr.reshape(128, 128, 128)
    raise ValueError(f"Unsupported μ-map size: {mu_path}, numel={arr.size}")


def mu_project_two_views_zyx(mu_zyx: np.ndarray, angles_deg: tuple[float, float], rotate_order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import rotate as nd_rotate

    outs_li = []
    outs_mip = []
    for a in angles_deg:
        rot = nd_rotate(
            mu_zyx,
            angle=float(a),
            axes=(1, 2),
            reshape=False,
            order=int(rotate_order),
            mode="constant",
            cval=0.0,
            prefilter=(rotate_order > 1),
        )
        outs_li.append(rot.sum(axis=1).astype(np.float32, copy=False))   # (z,x)
        outs_mip.append(rot.max(axis=1).astype(np.float32, copy=False))  # (z,x)

    li = np.stack(outs_li, axis=0)    # (2,128,128)
    mip = np.stack(outs_mip, axis=0)  # (2,128,128)
    return li, mip


def norm_clip_div(x: np.ndarray, max_value: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, float(max_value))
    return (x / float(max_value)).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "预计算 SPECT229 的 μ-map aux 缓存。\n"
            "- mode=ap: 仅生成 anterior/posterior 两张 (2,128,128)\n"
            "- mode=views: 生成所有视角 (V,128,128)，默认 V=60\n"
        )
    )
    parser.add_argument("--paired-root", type=str, default="datasets/SPECT229_projection_paired")
    parser.add_argument("--umap-root", type=str, default="datasets/SPECT229")
    parser.add_argument("--umap-filename", type=str, default="{patient}_PostAtten.dat")
    parser.add_argument(
        "--proj-pattern",
        type=str,
        default="*_Proj4Filter.dat",
        help="当 paired-root 不可用时，用 umap-root 下匹配该 pattern 的投影文件来推断病人列表。",
    )
    parser.add_argument("--out-root", type=str, default="datasets/SPECT229_umap_aux_cache")
    parser.add_argument("--mode", type=str, default="ap", choices=["ap", "views"])
    parser.add_argument("--anterior-view-index", type=int, default=0)
    parser.add_argument("--posterior-view-index", type=int, default=30)
    parser.add_argument("--views", type=int, default=60, help="mode=views 时的视角数")
    parser.add_argument("--start-angle-deg", type=float, default=-180.0)
    parser.add_argument("--angle-step-deg", type=float, default=6.0)
    parser.add_argument("--rotate-order", type=int, default=1, choices=[0, 1, 3])
    parser.add_argument("--posterior-flip", action="store_true", help="与训练 dataset 一致时才需要（默认 false）")
    parser.add_argument("--line-integral-max", type=float, default=2.0)
    parser.add_argument("--mip-max", type=float, default=0.05)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-patients", type=int, default=None)
    args = parser.parse_args()

    paired_root = Path(args.paired_root)
    umap_root = Path(args.umap_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect patients
    patients = set()
    used_source = None

    # 1) Prefer paired-root if available (legacy behavior)
    if paired_root.exists():
        for p in paired_root.glob("*.dat"):
            patients.add(parse_patient_from_basename(p.stem))
        if len(patients) > 0:
            used_source = f"paired-root: {paired_root}"

    # 2) Fallback: infer patients from umap-root projections (remote-friendly)
    # Expect layout: datasets/SPECT229/<patient>/*_Proj4Filter.dat
    if len(patients) == 0 and umap_root.exists():
        for p in umap_root.glob(f"*/{args.proj_pattern}"):
            if not p.is_file():
                continue
            patients.add(parse_patient_from_basename(p.stem))
        if len(patients) > 0:
            used_source = f"umap-root proj scan: {umap_root}/*/{args.proj_pattern}"

    patients = sorted(patients)
    if args.max_patients is not None:
        patients = patients[: int(args.max_patients)]

    if used_source is None:
        print(f"[WARN] patients: 0 (no files found in paired-root={paired_root} or umap-root={umap_root})")
        print("[HINT] On remote, either sync datasets/SPECT229_projection_paired OR just ensure datasets/SPECT229 exists.")
    else:
        print(f"[INFO] patients: {len(patients)} (source: {used_source})")

    a0 = float(args.start_angle_deg + args.anterior_view_index * args.angle_step_deg)
    a1 = float(args.start_angle_deg + args.posterior_view_index * args.angle_step_deg)
    v = int(args.views)

    done = 0
    missing = 0
    for pat in patients:
        out_path = out_root / (f"{pat}_umap_aux_ap.npz" if args.mode == "ap" else f"{pat}_umap_aux_views{v}.npz")
        if args.skip_existing and out_path.exists():
            continue

        mu_path = umap_root / pat / args.umap_filename.format(patient=pat)
        if not mu_path.exists():
            missing += 1
            if args.mode == "ap":
                li = np.zeros((2, 128, 128), dtype=np.float32)
                mip = np.zeros((2, 128, 128), dtype=np.float32)
            else:
                li = np.zeros((v, 128, 128), dtype=np.float32)
                mip = np.zeros((v, 128, 128), dtype=np.float32)
        else:
            mu = load_mu_map_zyx(mu_path)
            if args.mode == "ap":
                li, mip = mu_project_two_views_zyx(mu, angles_deg=(a0, a1), rotate_order=int(args.rotate_order))
                li = norm_clip_div(li, float(args.line_integral_max))
                mip = norm_clip_div(mip, float(args.mip_max))
                if args.posterior_flip:
                    li[1] = np.fliplr(li[1])
                    mip[1] = np.fliplr(mip[1])
            else:
                # full view stack (V,128,128)
                angles = [float(args.start_angle_deg + i * args.angle_step_deg) for i in range(v)]
                li_list = []
                mip_list = []
                for a in angles:
                    li2, mip2 = mu_project_two_views_zyx(mu, angles_deg=(a, a), rotate_order=int(args.rotate_order))
                    # li2/mip2 are (2,128,128) but both identical; take first
                    li_list.append(li2[0])
                    mip_list.append(mip2[0])
                li = norm_clip_div(np.stack(li_list, axis=0), float(args.line_integral_max))
                mip = norm_clip_div(np.stack(mip_list, axis=0), float(args.mip_max))

        if args.mode == "ap":
            np.savez_compressed(out_path, li_ap=li, mip_ap=mip)
        else:
            np.savez_compressed(out_path, li=li, mip=mip, start_angle_deg=float(args.start_angle_deg), angle_step_deg=float(args.angle_step_deg))
        done += 1
        if done % 25 == 0:
            print(f"[INFO] saved {done}/{len(patients)} (missing_mu={missing})")

    print(f"[OK] saved_cache={done}, missing_mu={missing}, out_root={out_root}")


if __name__ == "__main__":
    main()



