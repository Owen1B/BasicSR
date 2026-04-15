from __future__ import annotations

import hashlib
import multiprocessing as mp
import re
from pathlib import Path

import numpy as np


def md5_16(s: str) -> str:
    return hashlib.md5(str(s).encode()).hexdigest()[:16]


def counts_int(x: np.ndarray) -> np.ndarray:
    y = np.rint(x.astype(np.float64, copy=False))
    y = np.clip(y, 0.0, None)
    return y.astype(np.int64, copy=False)


def pick_ckpt(exp_dir: Path, *, prefer_iter: int | None = None) -> Path:
    models = Path(exp_dir) / "models"
    if prefer_iter is not None:
        p = models / f"net_g_{int(prefer_iter)}.pth"
        if p.is_file():
            return p
    best = models / "net_g_best.pth"
    if best.is_file():
        return best
    latest = models / "net_g_latest.pth"
    if latest.is_file():
        return latest

    best_it = -1
    best_p = None
    for p in models.glob("net_g_*.pth"):
        m = re.search(r"net_g_(\d+)\.pth", p.name)
        if not m:
            continue
        it = int(m.group(1))
        if it > best_it:
            best_it = it
            best_p = p
    if best_p is None:
        raise FileNotFoundError(f"No checkpoint found under: {models}")
    return best_p


def seed_cache_dir(patient_dir: Path, seed: int) -> Path:
    root = Path(patient_dir) / "bm3d_cache"
    d = root / f"seed{int(seed)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_cache_dir(patient_dir: Path, seed: int) -> Path | None:
    cand_root = Path(patient_dir) / "bm3d_cache"
    if not cand_root.exists():
        return None
    seed_dir = cand_root / f"seed{int(seed)}"
    return seed_dir if seed_dir.exists() else cand_root


def _load_cached_array(*, patient_dir: Path, lq_path: str | Path, label: str, seed: int, suffix: str) -> np.ndarray | None:
    ds_cache_dir = _resolve_cache_dir(patient_dir, seed)
    if ds_cache_dir is None:
        return None

    readable = ds_cache_dir / f"{label}_{suffix}.npy"
    if readable.exists():
        return np.load(readable).astype(np.float32, copy=False)

    key = md5_16(f"{str(lq_path)}__{label}")
    hashed = ds_cache_dir / f"{key}_{suffix}.npy"
    if hashed.exists():
        return np.load(hashed).astype(np.float32, copy=False)
    return None


def load_cached_thin(*, patient_dir: Path, lq_path: str | Path, label: str, seed: int, require_cached: bool) -> np.ndarray:
    arr = _load_cached_array(patient_dir=patient_dir, lq_path=lq_path, label=label, seed=seed, suffix="thin")
    if arr is not None:
        return arr
    mode = "required" if bool(require_cached) else "optional"
    raise FileNotFoundError(
        f"Missing cached low-dose split ({mode}): patient={Path(patient_dir).name} label={label} "
        f"(expected under {Path(patient_dir) / 'bm3d_cache' / f'seed{int(seed)}'})."
    )


def load_cached_bm3d(*, patient_dir: Path, lq_path: str | Path, label: str, seed: int, require_cached: bool) -> np.ndarray:
    arr = _load_cached_array(patient_dir=patient_dir, lq_path=lq_path, label=label, seed=seed, suffix="bm3d")
    if arr is not None:
        return np.clip(arr.astype(np.float32, copy=False), 0.0, None)
    mode = "required" if bool(require_cached) else "optional"
    raise FileNotFoundError(
        f"Missing cached BM3D ({mode}): patient={Path(patient_dir).name} label={label} "
        f"(expected under {Path(patient_dir) / 'bm3d_cache' / f'seed{int(seed)}'})."
    )


def poisson_thin_binomial(
    y20i: np.ndarray,
    k: float,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if float(k) <= 1.0:
        return y20i.astype(np.float32, copy=False)
    if rng is None:
        if seed is None:
            raise ValueError("poisson_thin_binomial requires either seed or rng")
        rng = np.random.default_rng(int(seed))
    xk_i = rng.binomial(y20i.astype(np.int64, copy=False), 1.0 / float(k)).astype(np.int64, copy=False)
    return xk_i.astype(np.float32, copy=False)


def anscombe_forward(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(np.maximum(x + 3.0 / 8.0, 0.0))


def anscombe_inverse_unbiased(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    y = np.asarray(y).astype(np.float32, copy=False)
    y_safe = np.maximum(y, float(eps))
    inv = (
        (y * 0.5) ** 2
        - 1.0 / 8.0
        + (1.0 / 4.0) * np.sqrt(3.0 / 2.0) * (1.0 / y_safe)
        - (11.0 / 8.0) * (1.0 / (y_safe**2))
        + (5.0 / 8.0) * np.sqrt(3.0 / 2.0) * (1.0 / (y_safe**3))
        - (1.0 / 8.0) * (1.0 / (y_safe**4))
    )
    return np.maximum(inv, 0.0).astype(np.float32, copy=False)


def _bm3d_worker_one_view(args: tuple[int, np.ndarray, float]) -> tuple[int, np.ndarray]:
    view_idx, view_data, sigma_psd = args
    import bm3d  # type: ignore

    x = np.clip(view_data.astype(np.float32, copy=False), 0.0, None)
    a = anscombe_forward(x)
    den_a = bm3d.bm3d(a, sigma_psd=float(sigma_psd), stage_arg=bm3d.BM3DStages.ALL_STAGES)
    den = anscombe_inverse_unbiased(den_a)
    return int(view_idx), den.astype(np.float32, copy=False)


def bm3d_anscombe_denoise(proj_cnt: np.ndarray, *, sigma_psd: float = 1.0, num_workers: int = 0) -> np.ndarray:
    try:
        import bm3d  # type: ignore  # noqa: F401
    except Exception as e:
        raise ImportError("bm3d package not found. Install it with: pip install bm3d") from e

    proj = np.clip(proj_cnt.astype(np.float32, copy=False), 0.0, None)
    out = np.zeros_like(proj, dtype=np.float32)

    if int(num_workers) and int(num_workers) > 1:
        args_list = [(int(vi), proj[int(vi)], float(sigma_psd)) for vi in range(int(proj.shape[0]))]
        with mp.Pool(processes=int(num_workers)) as pool:
            for vi, den in pool.imap_unordered(_bm3d_worker_one_view, args_list):
                out[int(vi)] = den
    else:
        for vi in range(int(proj.shape[0])):
            _, den = _bm3d_worker_one_view((int(vi), proj[int(vi)], float(sigma_psd)))
            out[vi] = den

    return np.clip(out.astype(np.float32, copy=False), 0.0, None)
