"""Recipe bootstrap: import custom modules so they register into BasicSR registries."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path


def _safe_import(module_name: str) -> None:
    try:
        __import__(module_name)
    except AssertionError as exc:
        # In dev environments where custom modules are already loaded in a forked BasicSR,
        # duplicate registry names may appear. Keep bootstrap idempotent.
        if "already registered" not in str(exc):
            raise


def _force_arch_mapping(registry_name: str, module_name: str, class_name: str) -> None:
    """Force registry symbol to resolve to recipe implementation.

    This avoids collisions when another BasicSR installation has already
    registered classes with the same symbol (e.g., UNetRes/UNet3DRes).
    """
    from basicsr.utils.registry import ARCH_REGISTRY

    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
    except AssertionError as exc:
        # Module import may fail on duplicate @register decorator assertion.
        # Load the module with registry decorator temporarily disabled.
        if "already registered" not in str(exc):
            raise
        mod_file = Path(__file__).resolve().parent / "archs" / f"{module_name.split('.')[-1]}.py"
        if not mod_file.is_file():
            raise FileNotFoundError(f"Cannot resolve module file for forced arch mapping: {module_name}")

        orig_register = ARCH_REGISTRY.register
        try:
            ARCH_REGISTRY.register = lambda *args, **kwargs: (lambda obj: obj)
            spec = importlib.util.spec_from_file_location(f"{module_name}__noreg", str(mod_file))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to load module spec: {module_name} from {mod_file}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cls = getattr(mod, class_name)
        finally:
            ARCH_REGISTRY.register = orig_register

    ARCH_REGISTRY._obj_map[str(registry_name)] = cls  # intentionally override symbol mapping


def register_all() -> None:
    # Archs
    # Recipe archs required on top of official BasicSR.
    _safe_import("prime.archs.unet2d_res_arch")
    _safe_import("prime.archs.unet3d_res_arch")
    _safe_import("prime.archs.planar_gated_unet_arch")
    _safe_import("prime.archs.support_aware_unet3d_res_arch")
    _force_arch_mapping("UNetRes", "prime.archs.unet2d_res_arch", "UNetRes")
    _force_arch_mapping("UNet3DRes", "prime.archs.unet3d_res_arch", "UNet3DRes")
    _force_arch_mapping("PlanarGatedUNetRes", "prime.archs.planar_gated_unet_arch", "PlanarGatedUNetRes")
    _force_arch_mapping(
        "SupportAwareUNet3DRes",
        "prime.archs.support_aware_unet3d_res_arch",
        "SupportAwareUNet3DRes",
    )

    # Data
    _safe_import("prime.data.spect_train_dataset")
    _safe_import("prime.data.spect_train_support_condition_dataset")
    _safe_import("prime.data.spect_projection_eval_dataset")
    _safe_import("prime.data.spect_train_aux2ch_dataset")
    _safe_import("prime.data.spect_projection_eval_aux2ch_dataset")

    # Losses / metrics
    _safe_import("prime.losses.poisson_loss")
    _safe_import("prime.losses.gradient_loss")
    _safe_import("prime.metrics.lpips_metric")
    _safe_import("prime.metrics.poisson_metric")
    _safe_import("prime.metrics.range_metrics")

    # Models
    _safe_import("prime.models.spect_model")
