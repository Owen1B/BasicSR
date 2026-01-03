from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


_CKPT_RE = re.compile(r"^net_g_(\d+)\.pth$")


def _iter_from_ckpt_name(name: str) -> Optional[int]:
    m = _CKPT_RE.match(str(name))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def list_experiments(experiments_root: Path) -> list[Path]:
    """List experiment directories (those containing a models/ dir with net_g_*.pth)."""
    root = Path(experiments_root)
    if not root.exists():
        return []
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        models = p / "models"
        if not models.is_dir():
            continue
        has_ckpt = any((_iter_from_ckpt_name(x.name) is not None) for x in models.iterdir() if x.is_file())
        has_latest = (models / "net_g_latest.pth").is_file()
        if has_ckpt or has_latest:
            out.append(p)
    return out


def pick_latest_ckpt(models_dir: Path) -> Path:
    """Pick the latest generator checkpoint under models_dir.

    Priority:
    1) net_g_latest.pth if present
    2) max iter net_g_<iter>.pth
    """
    models_dir = Path(models_dir)
    latest = models_dir / "net_g_latest.pth"
    if latest.is_file():
        return latest

    best: tuple[int, Path] | None = None
    for p in models_dir.iterdir():
        if not p.is_file():
            continue
        it = _iter_from_ckpt_name(p.name)
        if it is None:
            continue
        if best is None or it > best[0]:
            best = (it, p)
    if best is None:
        raise FileNotFoundError(f"No net_g_*.pth found under: {models_dir}")
    return best[1]


def find_config_for_experiment(exp_dir: Path) -> Path:
    """Find the training yaml inside the experiment directory (heuristic)."""
    exp_dir = Path(exp_dir)
    # Common: experiments/<name>/<name>.yml
    cand = exp_dir / f"{exp_dir.name}.yml"
    if cand.is_file():
        return cand
    cand2 = exp_dir / f"{exp_dir.name}.yaml"
    if cand2.is_file():
        return cand2
    # Fallback: pick any yml/yaml at exp root.
    ys = sorted([p for p in exp_dir.iterdir() if p.is_file() and p.suffix.lower() in [".yml", ".yaml"]])
    if len(ys) == 1:
        return ys[0]
    if len(ys) > 1:
        # Prefer those containing the experiment name.
        for p in ys:
            if exp_dir.name in p.stem:
                return p
        return ys[0]
    raise FileNotFoundError(f"No .yml/.yaml found under: {exp_dir}")


def resolve_patients(spect229_dir: Path) -> list[str]:
    """Return sorted patient folder names under datasets/SPECT229/."""
    root = Path(spect229_dir)
    if not root.exists():
        return []
    out: list[str] = []
    for p in sorted(root.iterdir()):
        if p.is_dir():
            out.append(p.name)
    return out


def parse_patient_indices(
    patients: list[str],
    *,
    indices: Optional[str] = None,
    index_range: Optional[str] = None,
) -> list[str]:
    """Select patients by indices or range string.

    - indices: "0,1,5"
    - index_range: "0:20" (python slice semantics, end exclusive)
    """
    if indices is None and index_range is None:
        return patients

    if indices is not None:
        idxs: list[int] = []
        for part in str(indices).split(","):
            part = part.strip()
            if part == "":
                continue
            idxs.append(int(part))
        sel: list[str] = []
        for i in idxs:
            if i < 0 or i >= len(patients):
                raise IndexError(f"patient index out of range: {i} (0..{len(patients)-1})")
            sel.append(patients[i])
        return sel

    # range
    s = str(index_range)
    if ":" not in s:
        raise ValueError(f"--patients-range should be like 0:20, got: {s}")
    a, b = s.split(":", 1)
    start = int(a) if a.strip() != "" else 0
    end = int(b) if b.strip() != "" else len(patients)
    if start < 0:
        start = 0
    if end > len(patients):
        end = len(patients)
    if end < start:
        end = start
    return patients[start:end]


