from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate as nd_rotate

from spect_ct.pipeline.io import ensure_dir, load_atten_f32, load_volume_f32_candidates


def _to_uint8_colormap(img: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    """把 2D 浮点图映射成 uint8 RGB（用于 GIF）。"""
    x = img.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if not np.any(finite):
        x = np.zeros_like(x, dtype=np.float32)
        vmin, vmax = 0.0, 1.0
    else:
        xf = x[finite]
        # robust range: 1%~99%
        vmin = float(np.percentile(xf, 1.0))
        vmax = float(np.percentile(xf, 99.0))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    x = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(x)  # float RGBA in [0,1]
    rgb = (rgba[..., :3] * 255.0).round().astype(np.uint8)
    return rgb


def _to_uint8_colormap_with_range(
    img: np.ndarray, vmin: float, vmax: float, cmap_name: str = "magma"
) -> np.ndarray:
    """把 2D 浮点图按给定 vmin/vmax 映射成 uint8 RGB（用于 GIF，避免帧间闪烁）。"""
    x = img.astype(np.float32, copy=False)
    vmin = float(vmin)
    vmax = float(vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    x = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(x)
    rgb = (rgba[..., :3] * 255.0).round().astype(np.uint8)
    return rgb


def _robust_vmin_vmax(stack: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> tuple[float, float]:
    x = stack.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if not np.any(finite):
        return 0.0, 1.0
    xf = x[finite]
    vmin = float(np.percentile(xf, q_low))
    vmax = float(np.percentile(xf, q_high))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _hstack_rgb(images: list[np.ndarray]) -> np.ndarray:
    """水平拼接多个 RGB uint8 图 (H,W,3)。"""
    if len(images) == 0:
        raise ValueError("images 不能为空")
    h = images[0].shape[0]
    c = images[0].shape[2]
    for im in images:
        if im.ndim != 3 or im.shape[0] != h or im.shape[2] != c:
            raise ValueError(f"RGB shape 不一致: {[x.shape for x in images]}")
    return np.concatenate(images, axis=1)


def load_orbit_csv(path: Path) -> dict[str, np.ndarray]:
    """读取 orbit.orb（CSV 文本）。

    观察到的格式（每行 8 列）：
      idx, angle_deg, radius_mm, head_id, a, b, c, d

    我们目前用于辅助输入对齐，只需要 angle_deg（以及可选画一下 radius_mm）。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"orbit 文件不存在: {path}")

    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append([float(p) for p in parts])
        except Exception:
            continue

    if len(rows) == 0:
        raise ValueError(f"orbit 文件解析失败（空/无法解析）: {path}")

    arr = np.array(rows, dtype=np.float32)
    if arr.shape[1] < 3:
        raise ValueError(f"orbit 列数不足: shape={arr.shape}, file={path}")

    idx = arr[:, 0].astype(np.int32)
    angle_deg = arr[:, 1].astype(np.float32)
    radius_mm = arr[:, 2].astype(np.float32)
    head_id = arr[:, 3].astype(np.float32) if arr.shape[1] > 3 else np.zeros_like(angle_deg)
    return {"idx": idx, "angle_deg": angle_deg, "radius_mm": radius_mm, "head_id": head_id, "raw": arr}


def _angles_deg(num_views: int, start_deg: float, range_deg: float) -> np.ndarray:
    # 与 par 文件一致：StartAngle=-180, AngleRange=360, NumViews=60
    return (start_deg + (np.arange(num_views, dtype=np.float32) * (range_deg / num_views))).astype(np.float32)


def forward_project_parallelbeam_line_integral(
    mu_zyx: np.ndarray,
    angles_deg: np.ndarray,
    pixel_size_mm: float,
    rotate_order: int = 1,
) -> np.ndarray:
    """简化正投影（平行束近似）：
    - 在 (y,x) 平面按角度旋转体数据（每个 z slice 同步旋转）
    - 沿 y 方向求和近似线积分：∫ μ ds ≈ sum_y μ * pixel_size

    输出形状: [views, z, x]，float32
    """
    if mu_zyx.ndim != 3:
        raise ValueError(f"mu_zyx 必须是 3D，实际 shape={mu_zyx.shape}")

    mu_zyx = mu_zyx.astype(np.float32, copy=False)
    pixel_size_cm = float(pixel_size_mm) / 10.0

    v = int(angles_deg.size)
    z, y, x = mu_zyx.shape
    out = np.empty((v, z, x), dtype=np.float32)

    for i, a in enumerate(angles_deg.tolist()):
        rot = nd_rotate(
            mu_zyx,
            angle=float(a),
            axes=(1, 2),  # rotate in (y,x) plane
            reshape=False,
            order=int(rotate_order),
            mode="constant",
            cval=0.0,
            prefilter=(rotate_order > 1),
        )
        out[i] = rot.sum(axis=1) * pixel_size_cm

    return out


def forward_project_parallelbeam_max_intensity(
    mu_zyx: np.ndarray,
    angles_deg: np.ndarray,
    rotate_order: int = 1,
) -> np.ndarray:
    """简化正投影（平行束近似）的“最大密度投影”(MIP-like)：
    - 在 (y,x) 平面按角度旋转体数据（每个 z slice 同步旋转）
    - 沿 y 方向取 max：max_y μ

    输出形状: [views, z, x]，float32
    """
    if mu_zyx.ndim != 3:
        raise ValueError(f"mu_zyx 必须是 3D，实际 shape={mu_zyx.shape}")

    mu_zyx = mu_zyx.astype(np.float32, copy=False)
    v = int(angles_deg.size)
    z, y, x = mu_zyx.shape
    out = np.empty((v, z, x), dtype=np.float32)

    for i, a in enumerate(angles_deg.tolist()):
        rot = nd_rotate(
            mu_zyx,
            angle=float(a),
            axes=(1, 2),  # rotate in (y,x) plane
            reshape=False,
            order=int(rotate_order),
            mode="constant",
            cval=0.0,
            prefilter=(rotate_order > 1),
        )
        out[i] = rot.max(axis=1)

    return out


def forward_project_parallelbeam_mean_intensity(
    mu_zyx: np.ndarray,
    angles_deg: np.ndarray,
    rotate_order: int = 1,
) -> np.ndarray:
    """简化正投影（平行束近似）的“平均强度投影”(AIP-like)：
    - 在 (y,x) 平面按角度旋转体数据（每个 z slice 同步旋转）
    - 沿 y 方向取 mean：mean_y μ

    输出形状: [views, z, x]，float32
    """
    if mu_zyx.ndim != 3:
        raise ValueError(f"mu_zyx 必须是 3D，实际 shape={mu_zyx.shape}")

    mu_zyx = mu_zyx.astype(np.float32, copy=False)
    v = int(angles_deg.size)
    z, y, x = mu_zyx.shape
    out = np.empty((v, z, x), dtype=np.float32)

    for i, a in enumerate(angles_deg.tolist()):
        rot = nd_rotate(
            mu_zyx,
            angle=float(a),
            axes=(1, 2),  # rotate in (y,x) plane
            reshape=False,
            order=int(rotate_order),
            mode="constant",
            cval=0.0,
            prefilter=(rotate_order > 1),
        )
        out[i] = rot.mean(axis=1)

    return out


def _threshold_multichannel(
    vol_zyx: np.ndarray,
    mode: str,
    hu_air_max: float,
    hu_soft_max: float,
    mu_air_max: float,
    mu_soft_max: float,
) -> dict[str, np.ndarray]:
    """返回 air/soft/bone 三个 mask（bool），根据 CT(HU) 或 μ-map 阈值分割。"""
    v = vol_zyx.astype(np.float32, copy=False)
    if mode == "ct_hu":
        air = v <= float(hu_air_max)
        soft = (v > float(hu_air_max)) & (v <= float(hu_soft_max))
        bone = v > float(hu_soft_max)
    elif mode == "mu":
        # 经验阈值（μ 通常在 1/cm，air≈0，soft~0.01-0.02，bone更大）
        air = v <= float(mu_air_max)
        soft = (v > float(mu_air_max)) & (v <= float(mu_soft_max))
        bone = v > float(mu_soft_max)
    else:
        raise ValueError(f"未知 mode: {mode}")
    return {"air": air, "soft": soft, "bone": bone}


def _bone_nonbone_masks(
    vol_zyx: np.ndarray,
    mode: str,
    bone_threshold: float,
    hu_air_max: float,
    mu_air_max: float,
) -> dict[str, np.ndarray]:
    """只分 bone / non-bone（bool masks）。"""
    v = vol_zyx.astype(np.float32, copy=False)
    if mode == "ct_hu":
        bone = v > float(bone_threshold)
        # 非骨骼包含空气+软组织
        nonbone = ~bone
    elif mode == "mu":
        bone = v > float(bone_threshold)
        nonbone = ~bone
    else:
        raise ValueError(f"未知 mode: {mode}")
    return {"bone": bone, "nonbone": nonbone}


def main() -> None:
    parser = argparse.ArgumentParser(description="对 μ-map(PostAtten) 做简化正投影并保存可视化图（平行束近似）。")
    parser.add_argument("--patient", type=str, required=True)
    parser.add_argument(
        "--atten",
        type=str,
        default=None,
        help="μ-map 路径（默认 datasets/SPECT229/<patient>/<patient>_PostAtten.dat）",
    )
    parser.add_argument(
        "--orbit-file",
        type=str,
        default=None,
        help="orbit.orb（若提供则使用其中的 angle_deg 作为 view 顺序；radius 仅用于可视化，不改变当前平行束近似）。",
    )
    parser.add_argument("--views", type=int, default=60)
    parser.add_argument("--start-angle-deg", type=float, default=-180.0)
    parser.add_argument("--angle-range-deg", type=float, default=360.0)
    parser.add_argument("--pixel-size-mm", type=float, default=4.4)
    parser.add_argument("--rotate-order", type=int, default=1, choices=[0, 1, 3])
    parser.add_argument("--z-index", type=int, default=64, help="用于 sinogram 可视化的 z 索引")
    parser.add_argument("--view-indices", type=int, nargs="*", default=[0, 15, 30, 45])
    parser.add_argument("--results-root", type=str, default="spect_ct/results")
    parser.add_argument("--exp-name", type=str, default="debug_umap_forward_projection")
    parser.add_argument(
        "--second-row",
        type=str,
        default="max",
        choices=["max", "exp", "none"],
        help="第二行显示内容：max=最大密度投影(max_y μ)，exp=exp(-∫μds)，none=不画第二行",
    )
    parser.add_argument(
        "--aip-row",
        action="store_true",
        help="增加一行 AIP：mean_y μ（用于更平滑的辅助投影）。",
    )
    parser.add_argument(
        "--multichannel",
        action="store_true",
        help="额外生成多通道投影（air/soft/bone）。若提供 --ct-file 则按 HU 阈值分割，否则按 μ 阈值分割。",
    )
    parser.add_argument(
        "--ct-file",
        type=str,
        default=None,
        help="可选：CT 体数据（HU）.dat 文件路径（float32），用于 HU 阈值分割多通道；若不提供则使用 μ 阈值分割。",
    )
    parser.add_argument("--hu-air-max", type=float, default=-500.0, help="HU 分割阈值：air <= 该值")
    parser.add_argument("--hu-soft-max", type=float, default=300.0, help="HU 分割阈值：soft <= 该值，bone > 该值")
    parser.add_argument("--mu-air-max", type=float, default=0.005, help="μ 分割阈值：air <= 该值（1/cm）")
    parser.add_argument("--mu-soft-max", type=float, default=0.020, help="μ 分割阈值：soft <= 该值（1/cm），bone > 该值")
    parser.add_argument(
        "--sweep-bone-thresholds",
        type=str,
        default=None,
        help="骨阈值 sweep（逗号分隔）。若提供 --ct-file，则阈值单位为 HU；否则阈值单位为 μ(1/cm)。例如: '0.015,0.02,0.025' 或 '200,300,400'.",
    )
    parser.add_argument("--no-exp", action="store_true", help="兼容旧参数：等价于 --second-row none")
    parser.add_argument("--log1p", action="store_true", help="对线积分图使用 log1p 以增强对比度")
    parser.add_argument("--gif", action="store_true", help="保存积分投影(∫μds)的 view 序列 GIF。")
    parser.add_argument("--gif-fps", type=float, default=10.0)
    parser.add_argument("--multichannel-gif", action="store_true", help="对多通道投影额外保存 GIF（air/soft/bone）。")
    parser.add_argument(
        "--multichannel-gif-mode",
        type=str,
        default="integral",
        choices=["integral", "max", "both"],
        help="分通道 GIF 类型：integral=∫μds；max=max_y μ；both=两者都输出。",
    )
    parser.add_argument(
        "--bone-threshold",
        type=float,
        default=None,
        help="仅生成 bone vs non-bone 时的 bone 阈值（μ-map 模式单位=1/cm；CT(HU) 模式单位=HU）。",
    )
    parser.add_argument(
        "--bone-gif",
        action="store_true",
        help="用 --bone-threshold 生成 bone mask 后，输出一个三联 GIF：∫投影 / MIP(max) / AIP(mean)。",
    )
    parser.add_argument("--bone-gif-fps", type=float, default=10.0)
    args = parser.parse_args()

    patient = args.patient
    if args.atten is None:
        atten_path = Path("datasets/SPECT229") / patient / f"{patient}_PostAtten.dat"
    else:
        atten_path = Path(args.atten)

    mu = load_atten_f32(atten_path)
    if mu.shape != (128, 128, 128):
        raise ValueError(f"当前脚本仅支持 128^3 μ-map，实际 shape={mu.shape}，文件={atten_path}")

    # 兼容旧参数
    if args.no_exp:
        args.second_row = "none"

    orbit = None
    radius_mm = None
    if args.orbit_file is not None:
        orbit = load_orbit_csv(Path(args.orbit_file))
        angles = orbit["angle_deg"]
        radius_mm = orbit["radius_mm"]
        if int(angles.size) != int(args.views):
            raise ValueError(
                f"orbit 视角数与 --views 不一致: orbit={int(angles.size)} vs views={int(args.views)}"
            )
    else:
        angles = _angles_deg(args.views, args.start_angle_deg, args.angle_range_deg)
    line_int = forward_project_parallelbeam_line_integral(
        mu_zyx=mu,
        angles_deg=angles,
        pixel_size_mm=args.pixel_size_mm,
        rotate_order=args.rotate_order,
    )  # [v,z,x]
    max_proj = None
    if args.second_row == "max":
        max_proj = forward_project_parallelbeam_max_intensity(
            mu_zyx=mu,
            angles_deg=angles,
            rotate_order=args.rotate_order,
        )  # [v,z,x]
    aip_proj = None
    if args.aip_row:
        aip_proj = forward_project_parallelbeam_mean_intensity(
            mu_zyx=mu,
            angles_deg=angles,
            rotate_order=args.rotate_order,
        )  # [v,z,x]

    z_idx = int(args.z_index)
    if not (0 <= z_idx < line_int.shape[1]):
        raise ValueError(f"--z-index 越界: {z_idx}, z={line_int.shape[1]}")

    # 组织输出目录
    out_root = Path(args.results_root) / args.exp_name / "umap_forward_projection" / patient
    ensure_dir(out_root)

    # 保存数值（方便你后续做定量比较）
    np.save(out_root / "umap_line_integral_views_zx.npy", line_int)  # [views,z,x]
    if max_proj is not None:
        np.save(out_root / "umap_maxproj_views_zx.npy", max_proj)  # [views,z,x]
    if aip_proj is not None:
        np.save(out_root / "umap_aip_views_zx.npy", aip_proj)  # [views,z,x]
    np.save(out_root / "angles_deg.npy", angles)
    if radius_mm is not None:
        np.save(out_root / "radius_mm.npy", radius_mm)

    # 可视化
    import matplotlib.pyplot as plt

    sel = [int(i) for i in args.view_indices]
    sel = [i for i in sel if 0 <= i < line_int.shape[0]]
    if len(sel) == 0:
        sel = [0, min(15, line_int.shape[0] - 1)]

    # row3: sinogram for one z
    sino = line_int[:, z_idx, :]  # [views,x]

    show_orbit_row = radius_mm is not None
    show_second_row = args.second_row != "none"
    show_aip_row = bool(args.aip_row)
    # rows:
    # 1) line integral (selected views)
    # 2) second row (optional)
    # 3) AIP row (optional)
    # 4) sinogram (always)
    # 5) orbit radius curve (optional)
    fig_rows = 1 + (1 if show_second_row else 0) + (1 if show_aip_row else 0) + 1 + (1 if show_orbit_row else 0)
    fig = plt.figure(figsize=(4 * len(sel), 3.2 * fig_rows), dpi=150)

    # Row 1: line integrals for selected views (z vs x)
    for j, vidx in enumerate(sel):
        ax = fig.add_subplot(fig_rows, len(sel), 1 + j)
        img = line_int[vidx]
        if args.log1p:
            img = np.log1p(np.maximum(img, 0.0))
        im = ax.imshow(img, cmap="magma", aspect="auto")
        ax.set_title(f"∫μds (view {vidx}, deg={angles[vidx]:.1f})")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    current_row = 1  # already used row 1 for line integral

    # Row 2: MIP-like max projection OR exp(-line integral)
    if show_second_row:
        current_row += 1
        for j, vidx in enumerate(sel):
            ax = fig.add_subplot(fig_rows, len(sel), (current_row - 1) * len(sel) + 1 + j)
            if args.second_row == "max":
                assert max_proj is not None
                img2 = max_proj[vidx]
                im = ax.imshow(img2, cmap="magma", aspect="auto")
                ax.set_title(f"max_y μ (view {vidx})")
            else:
                att = np.exp(-np.maximum(line_int[vidx], 0.0))
                im = ax.imshow(att, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
                ax.set_title(f"exp(-∫μds) (view {vidx})")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 3: AIP mean_y μ
    if show_aip_row:
        current_row += 1
        assert aip_proj is not None
        for j, vidx in enumerate(sel):
            ax = fig.add_subplot(fig_rows, len(sel), (current_row - 1) * len(sel) + 1 + j)
            img3 = aip_proj[vidx]
            im = ax.imshow(img3, cmap="magma", aspect="auto")
            ax.set_title(f"mean_y μ (AIP) (view {vidx})")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Last row: sinogram for one z
    current_row += 1
    ax = fig.add_subplot(fig_rows, 1, current_row)
    sino_img = sino
    if args.log1p:
        sino_img = np.log1p(np.maximum(sino_img, 0.0))
    im = ax.imshow(sino_img, cmap="magma", aspect="auto")
    ax.set_title(f"Sinogram (line integral) at z={z_idx}: shape={tuple(sino.shape)}")
    ax.set_xlabel("x")
    ax.set_ylabel("view")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    if show_orbit_row:
        current_row += 1
        ax = fig.add_subplot(fig_rows, 1, current_row)
        ax.plot(np.arange(radius_mm.size), radius_mm, "-k", linewidth=1.2)
        ax.set_title("orbit.orb radius_mm over views (for reference)")
        ax.set_xlabel("view index")
        ax.set_ylabel("radius (mm)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "μ-map forward projection (parallel-beam approx)\n"
        f"patient={patient}  atten={atten_path.name}"
        + ("" if args.orbit_file is None else f"  orbit={Path(args.orbit_file).name}"),
        y=0.995,
    )
    fig.tight_layout()
    out_png = out_root / "umap_forward_projection.png"
    fig.savefig(out_png)
    plt.close(fig)

    # GIF: line integral projections over views
    if args.gif:
        from PIL import Image

        frames: list[Image.Image] = []
        stack = line_int
        if args.log1p:
            stack = np.log1p(np.maximum(stack, 0.0))
        vmin, vmax = _robust_vmin_vmax(stack)
        for i in range(line_int.shape[0]):
            img = stack[i]
            if args.log1p:
                # already applied above
                pass
            rgb = _to_uint8_colormap_with_range(img, vmin=vmin, vmax=vmax, cmap_name="magma")
            pil = Image.fromarray(rgb, mode="RGB")
            frames.append(pil)

        gif_path = out_root / "umap_line_integral.gif"
        duration_ms = int(round(1000.0 / max(float(args.gif_fps), 1e-6)))
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
        )
        print(f"[OK] saved: {gif_path}")

    # Multi-channel projections (optional)
    if args.multichannel:
        if args.ct_file is not None:
            ct_path = Path(args.ct_file)
            ct = load_volume_f32_candidates(ct_path, candidates=[(128, 128, 128), (32, 256, 256)])
            if ct.shape != mu.shape:
                raise ValueError(f"CT shape 必须与 μ-map 一致以便同域投影：ct={ct.shape}, mu={mu.shape}")
            masks = _threshold_multichannel(
                vol_zyx=ct,
                mode="ct_hu",
                hu_air_max=args.hu_air_max,
                hu_soft_max=args.hu_soft_max,
                mu_air_max=args.mu_air_max,
                mu_soft_max=args.mu_soft_max,
            )
            mode_name = "ct_hu"
        else:
            # 没有 CT，就用 μ-map 本身做一个近似分段（仍可作为辅助通道）
            masks = _threshold_multichannel(
                vol_zyx=mu,
                mode="mu",
                hu_air_max=args.hu_air_max,
                hu_soft_max=args.hu_soft_max,
                mu_air_max=args.mu_air_max,
                mu_soft_max=args.mu_soft_max,
            )
            mode_name = "mu_threshold"

        mc_dir = out_root / "multichannel"
        ensure_dir(mc_dir)

        # 对每个 channel：对 μ-map * mask 做 ∫ 与 max（并可选 AIP）
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 8), dpi=150)
        channels = ["air", "soft", "bone"]
        for ci, ch in enumerate(channels):
            m = masks[ch].astype(np.float32)
            mu_ch = mu * m

            li = forward_project_parallelbeam_line_integral(
                mu_zyx=mu_ch, angles_deg=angles, pixel_size_mm=args.pixel_size_mm, rotate_order=args.rotate_order
            )
            mx = forward_project_parallelbeam_max_intensity(mu_zyx=mu_ch, angles_deg=angles, rotate_order=args.rotate_order)
            np.save(mc_dir / f"{ch}_umap_line_integral_views_zx.npy", li)
            np.save(mc_dir / f"{ch}_umap_maxproj_views_zx.npy", mx)

            # preview one view (first selected)
            vidx = int(sel[0])
            ax1 = fig.add_subplot(3, 2, ci * 2 + 1)
            img = li[vidx]
            if args.log1p:
                img = np.log1p(np.maximum(img, 0.0))
            im1 = ax1.imshow(img, cmap="magma", aspect="auto")
            ax1.set_title(f"{ch}: ∫μds (view {vidx})")
            ax1.set_xlabel("x")
            ax1.set_ylabel("z")
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            ax2 = fig.add_subplot(3, 2, ci * 2 + 2)
            im2 = ax2.imshow(mx[vidx], cmap="magma", aspect="auto")
            ax2.set_title(f"{ch}: max_y μ (view {vidx})")
            ax2.set_xlabel("x")
            ax2.set_ylabel("z")
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        fig.suptitle(f"Multi-channel μ projections ({mode_name})  patient={patient}", y=0.995)
        fig.tight_layout()
        mc_png = mc_dir / "multichannel_projection_preview.png"
        fig.savefig(mc_png)
        plt.close(fig)
        print(f"[OK] saved: {mc_png}")

        # per-channel GIFs
        if args.multichannel_gif:
            from PIL import Image

            duration_ms = int(round(1000.0 / max(float(args.gif_fps), 1e-6)))

            for ch in channels:
                li = np.load(mc_dir / f"{ch}_umap_line_integral_views_zx.npy")
                mx = np.load(mc_dir / f"{ch}_umap_maxproj_views_zx.npy")

                if args.multichannel_gif_mode in ("integral", "both"):
                    stack = li
                    if args.log1p:
                        stack = np.log1p(np.maximum(stack, 0.0))
                    vmin, vmax = _robust_vmin_vmax(stack)
                    frames = [
                        Image.fromarray(_to_uint8_colormap_with_range(stack[i], vmin=vmin, vmax=vmax), mode="RGB")
                        for i in range(stack.shape[0])
                    ]
                    out_gif = mc_dir / f"{ch}_umap_line_integral.gif"
                    frames[0].save(
                        out_gif,
                        save_all=True,
                        append_images=frames[1:],
                        duration=duration_ms,
                        loop=0,
                        disposal=2,
                    )
                    print(f"[OK] saved: {out_gif}")

                if args.multichannel_gif_mode in ("max", "both"):
                    stack = mx
                    vmin, vmax = _robust_vmin_vmax(stack)
                    frames = [
                        Image.fromarray(_to_uint8_colormap_with_range(stack[i], vmin=vmin, vmax=vmax), mode="RGB")
                        for i in range(stack.shape[0])
                    ]
                    out_gif = mc_dir / f"{ch}_umap_maxproj.gif"
                    frames[0].save(
                        out_gif,
                        save_all=True,
                        append_images=frames[1:],
                        duration=duration_ms,
                        loop=0,
                        disposal=2,
                    )
                    print(f"[OK] saved: {out_gif}")

    print(f"[OK] saved: {out_png}")

    # Bone threshold sweep (bone vs non-bone)
    if args.sweep_bone_thresholds is not None:
        # decide domain
        if args.ct_file is not None:
            ct_path = Path(args.ct_file)
            ct = load_volume_f32_candidates(ct_path, candidates=[(128, 128, 128), (32, 256, 256)])
            if ct.shape != mu.shape:
                raise ValueError(f"CT shape 必须与 μ-map 一致以便同域投影：ct={ct.shape}, mu={mu.shape}")
            sweep_vol = ct
            mode = "ct_hu"
        else:
            sweep_vol = mu
            mode = "mu"

        # parse thresholds
        ths: list[float] = []
        for s in str(args.sweep_bone_thresholds).split(","):
            s = s.strip()
            if not s:
                continue
            ths.append(float(s))
        if len(ths) == 0:
            raise ValueError("--sweep-bone-thresholds 为空")

        sweep_dir = out_root / "bone_threshold_sweep"
        ensure_dir(sweep_dir)

        # choose one view for preview
        vidx = int(sel[0])
        z_mid = mu.shape[0] // 2

        import matplotlib.pyplot as plt

        # Grid: each column = one threshold
        # rows:
        # 1) bone mask mid-slice
        # 2) non-bone mask mid-slice
        # 3) bone ∫μds (view vidx)
        # 4) non-bone ∫μds (view vidx)
        ncol = len(ths)
        nrow = 4
        fig = plt.figure(figsize=(3.6 * ncol, 3.2 * nrow), dpi=150)

        for j, thr in enumerate(ths):
            masks = _bone_nonbone_masks(
                vol_zyx=sweep_vol,
                mode=mode,
                bone_threshold=thr,
                hu_air_max=args.hu_air_max,
                mu_air_max=args.mu_air_max,
            )
            bone = masks["bone"].astype(np.float32)
            nonbone = masks["nonbone"].astype(np.float32)

            # slice previews
            ax = fig.add_subplot(nrow, ncol, 1 + j)
            ax.imshow(bone[z_mid], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"bone mask\\nth={thr:g} ({mode})")
            ax.axis("off")

            ax = fig.add_subplot(nrow, ncol, 1 + ncol + j)
            ax.imshow(nonbone[z_mid], cmap="gray", vmin=0, vmax=1)
            ax.set_title("non-bone mask")
            ax.axis("off")

            # projections (∫) for this threshold
            li_bone = forward_project_parallelbeam_line_integral(
                mu_zyx=mu * bone,
                angles_deg=angles,
                pixel_size_mm=args.pixel_size_mm,
                rotate_order=args.rotate_order,
            )
            li_non = forward_project_parallelbeam_line_integral(
                mu_zyx=mu * nonbone,
                angles_deg=angles,
                pixel_size_mm=args.pixel_size_mm,
                rotate_order=args.rotate_order,
            )

            # save numeric for later
            np.save(sweep_dir / f"bone_thr_{thr:g}_umap_line_integral_views_zx.npy", li_bone)
            np.save(sweep_dir / f"nonbone_thr_{thr:g}_umap_line_integral_views_zx.npy", li_non)

            img_b = li_bone[vidx]
            img_n = li_non[vidx]
            if args.log1p:
                img_b = np.log1p(np.maximum(img_b, 0.0))
                img_n = np.log1p(np.maximum(img_n, 0.0))

            ax = fig.add_subplot(nrow, ncol, 1 + 2 * ncol + j)
            im = ax.imshow(img_b, cmap="magma", aspect="auto")
            ax.set_title(f"bone ∫μds\\nview={vidx}")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax = fig.add_subplot(nrow, ncol, 1 + 3 * ncol + j)
            im = ax.imshow(img_n, cmap="magma", aspect="auto")
            ax.set_title("non-bone ∫μds")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"Bone threshold sweep (bone vs non-bone)  patient={patient}  view={vidx}", y=0.995)
        fig.tight_layout()
        out_sweep = sweep_dir / "bone_threshold_sweep.png"
        fig.savefig(out_sweep)
        plt.close(fig)
        print(f"[OK] saved: {out_sweep}")

    # Bone gif (∫ / max / AIP) for a fixed threshold
    if args.bone_gif:
        if args.bone_threshold is None:
            raise ValueError("--bone-gif 需要同时提供 --bone-threshold")

        # decide mask domain
        if args.ct_file is not None:
            ct_path = Path(args.ct_file)
            ct = load_volume_f32_candidates(ct_path, candidates=[(128, 128, 128), (32, 256, 256)])
            if ct.shape != mu.shape:
                raise ValueError(f"CT shape 必须与 μ-map 一致以便同域投影：ct={ct.shape}, mu={mu.shape}")
            sweep_vol = ct
            mode = "ct_hu"
        else:
            sweep_vol = mu
            mode = "mu"

        masks = _bone_nonbone_masks(
            vol_zyx=sweep_vol,
            mode=mode,
            bone_threshold=float(args.bone_threshold),
            hu_air_max=args.hu_air_max,
            mu_air_max=args.mu_air_max,
        )
        bone = masks["bone"].astype(np.float32)
        mu_bone = mu * bone

        li = forward_project_parallelbeam_line_integral(
            mu_zyx=mu_bone,
            angles_deg=angles,
            pixel_size_mm=args.pixel_size_mm,
            rotate_order=args.rotate_order,
        )
        mx = forward_project_parallelbeam_max_intensity(mu_zyx=mu_bone, angles_deg=angles, rotate_order=args.rotate_order)
        aip = forward_project_parallelbeam_mean_intensity(mu_zyx=mu_bone, angles_deg=angles, rotate_order=args.rotate_order)

        bone_dir = out_root / "bone_gif"
        ensure_dir(bone_dir)
        np.save(bone_dir / f"bone_thr_{args.bone_threshold:g}_line_integral_views_zx.npy", li)
        np.save(bone_dir / f"bone_thr_{args.bone_threshold:g}_maxproj_views_zx.npy", mx)
        np.save(bone_dir / f"bone_thr_{args.bone_threshold:g}_aip_views_zx.npy", aip)

        # consistent color scaling across frames per projection type
        li_stack = li
        if args.log1p:
            li_stack = np.log1p(np.maximum(li_stack, 0.0))
        li_vmin, li_vmax = _robust_vmin_vmax(li_stack)
        mx_vmin, mx_vmax = _robust_vmin_vmax(mx)
        aip_vmin, aip_vmax = _robust_vmin_vmax(aip)

        from PIL import Image

        frames: list[Image.Image] = []
        for i in range(li.shape[0]):
            img_li = li_stack[i]
            img_mx = mx[i]
            img_aip = aip[i]

            rgb_li = _to_uint8_colormap_with_range(img_li, vmin=li_vmin, vmax=li_vmax, cmap_name="magma")
            rgb_mx = _to_uint8_colormap_with_range(img_mx, vmin=mx_vmin, vmax=mx_vmax, cmap_name="magma")
            rgb_aip = _to_uint8_colormap_with_range(img_aip, vmin=aip_vmin, vmax=aip_vmax, cmap_name="magma")

            rgb = _hstack_rgb([rgb_li, rgb_mx, rgb_aip])
            frames.append(Image.fromarray(rgb, mode="RGB"))

        out_gif = bone_dir / f"bone_thr_{args.bone_threshold:g}_integral_max_aip.gif"
        duration_ms = int(round(1000.0 / max(float(args.bone_gif_fps), 1e-6)))
        frames[0].save(
            out_gif,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
        )
        print(f"[OK] saved: {out_gif}")


if __name__ == "__main__":
    main()


