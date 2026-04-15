#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import yaml


FORBIDDEN_MODEL_TYPES = {
    "SPECT3DModel",
    "SPECT3DConsistencyModel",
    "Neighbor2NeighborModel",
    "Noise2VoidModel",
    "Noise2VoidBlindSpotModel",
    "Self2SelfModel",
}

FORBIDDEN_BOOTSTRAP_IMPORTS = {
    "prime.models.spect3d_model",
    "prime.models.spect3d_consistency_model",
    "prime.models.neighbor2neighbor_model",
    "prime.models.noise2void_model",
    "prime.models.self2self_model",
    "prime.data.spect60view_dataset",
    "prime.data.neighbor2neighbor_dataset",
    "prime.data.noise2void_dataset",
    "prime.data.noiser2noise_dataset",
    "prime.data.self2self_dataset",
    "prime.data.spect_dat_paired_dataset",
    "prime.data.spect60view_volume_dataset",
    "prime.data.nema_single_projection_dataset",
}

ALLOWED_MAINLINE_DATASET_TYPES = {
    "SPECTTrainDataset",
    "SPECTProjectionEvalDataset",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_active_option_files(root: Path):
    dirs = [
        root / "options" / "train" / "converged",
        root / "options" / "train" / "debug",
        root / "options" / "test" / "converged",
        root / "options" / "test" / "debug",
    ]
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.yml")):
            if p.is_file():
                yield p


def _check_model_types(root: Path) -> list[str]:
    violations: list[str] = []
    for yml_path in _iter_active_option_files(root):
        try:
            with open(yml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            violations.append(f"[YAML] {yml_path}: failed to parse ({e})")
            continue
        if not isinstance(cfg, dict):
            continue
        mt = str(cfg.get("model_type", "")).strip()
        if mt in FORBIDDEN_MODEL_TYPES:
            violations.append(f"[model_type] {yml_path}: forbidden model_type={mt}")
    return violations


def _check_bootstrap(root: Path) -> list[str]:
    p = root / "prime" / "bootstrap.py"
    if not p.exists():
        return [f"[bootstrap] missing file: {p}"]
    txt = p.read_text(encoding="utf-8")
    violations: list[str] = []
    for imp in sorted(FORBIDDEN_BOOTSTRAP_IMPORTS):
        if imp in txt:
            violations.append(f"[bootstrap] forbidden import registered: {imp}")
    return violations


def _check_dataset_types(root: Path) -> list[str]:
    violations: list[str] = []
    for yml_path in _iter_active_option_files(root):
        try:
            with open(yml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            violations.append(f"[YAML] {yml_path}: failed to parse ({e})")
            continue
        if not isinstance(cfg, dict):
            continue

        datasets = cfg.get("datasets", {}) or {}
        if not isinstance(datasets, dict):
            continue
        for split, ds_cfg in datasets.items():
            if not isinstance(ds_cfg, dict):
                continue
            ds_type = str(ds_cfg.get("type", "")).strip()
            if ds_type and ds_type not in ALLOWED_MAINLINE_DATASET_TYPES:
                violations.append(
                    f"[dataset_type] {yml_path} split={split}: "
                    f"unsupported dataset type={ds_type}, allowed={sorted(ALLOWED_MAINLINE_DATASET_TYPES)}"
                )
    return violations


def _check_models_imports(root: Path) -> list[str]:
    violations: list[str] = []
    models_dir = root / "prime" / "models"
    if not models_dir.exists():
        return [f"[models] missing directory: {models_dir}"]

    for py in sorted(models_dir.rglob("*.py")):
        rel = py.relative_to(models_dir)
        if rel.parts and rel.parts[0] == "_legacy":
            continue
        txt = py.read_text(encoding="utf-8")
        if "sr_model_ext" in txt:
            violations.append(f"[models] active module references sr_model_ext: {py}")
    return violations


def main() -> int:
    root = _repo_root()
    violations: list[str] = []
    violations.extend(_check_model_types(root))
    violations.extend(_check_dataset_types(root))
    violations.extend(_check_bootstrap(root))
    violations.extend(_check_models_imports(root))

    if violations:
        print("Mainline integrity check FAILED:")
        for v in violations:
            print(f"- {v}")
        return 1

    print("Mainline integrity check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
