#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _setup_import_path() -> Path:
    script_path = Path(__file__).resolve()
    prime_root = script_path.parents[1]
    repo_root = prime_root.parent
    for p in (str(prime_root), str(repo_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return prime_root


PRIME_ROOT = _setup_import_path()

from prime import register_all  # noqa: E402
from basicsr.archs import build_network  # noqa: E402
from basicsr.utils.options import yaml_load  # noqa: E402


DEFAULT_MODEL_NAME = "converged_planar_2d_2view_multidose_N_clinical_fullimg_scratch"
DEFAULT_CONFIG = PRIME_ROOT / "options" / "train" / "converged" / (
    "train_planar_2d_2view_multidose_N_clinical_fullimg_scratch.yml"
)
DEFAULT_CHECKPOINT = PRIME_ROOT / "experiments" / DEFAULT_MODEL_NAME / "models" / "net_g_10000.pth"
DEFAULT_SAMPLE_INPUT = PRIME_ROOT / "datasets" / "spectH_clinical_new" / "93106_20191218145634679000.dat"
DEFAULT_OUTPUT_DIR = PRIME_ROOT / "deploy" / f"{DEFAULT_MODEL_NAME}_iter10000_onnx_mvp"
DEFAULT_INPUT_SHAPE = (1, 2, 1024, 256)
DEFAULT_RAW_SHAPE = (2, 1024, 256)


@dataclass
class ExportStats:
    max_abs_diff: float
    mean_abs_diff: float
    pytorch_min: float
    pytorch_max: float
    onnx_min: float
    onnx_max: float
    post_max_abs_diff: float
    post_mean_abs_diff: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export planar 2D PRIME model to a minimal ONNX MVP bundle.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Experiment yaml with network_g config.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Checkpoint containing params_ema.")
    parser.add_argument("--sample-input", type=Path, default=DEFAULT_SAMPLE_INPUT, help="Sample raw float32 dat.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Bundle output directory.")
    parser.add_argument("--param-key", type=str, default="params_ema", help="Checkpoint key to load.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=4,
        default=DEFAULT_INPUT_SHAPE,
        metavar=("N", "C", "H", "W"),
        help="Fixed ONNX input shape. Default: 1 2 1024 256",
    )
    parser.add_argument(
        "--raw-shape",
        type=int,
        nargs=3,
        default=DEFAULT_RAW_SHAPE,
        metavar=("C", "H", "W"),
        help="Raw planar case shape before preprocess. Default: 2 1024 256",
    )
    parser.add_argument("--max-value", type=float, default=150.0, help="Linear normalization divisor used in training.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip onnxruntime numerical verification.")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_state_dict(checkpoint_path: Path, preferred_key: str) -> tuple[dict[str, Any], str]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(ckpt).__name__}")

    if preferred_key in ckpt:
        return ckpt[preferred_key], preferred_key

    fallback_order = ["params_ema", "params"]
    for key in fallback_order:
        if key in ckpt:
            return ckpt[key], key

    raise KeyError(
        f"No usable state_dict key found in {checkpoint_path}. "
        f"Requested={preferred_key}, available={sorted(ckpt.keys())}"
    )


def build_model(config_path: Path, checkpoint_path: Path, param_key: str) -> tuple[torch.nn.Module, dict[str, Any], str]:
    register_all()
    cfg = yaml_load(str(config_path))
    net = build_network(cfg["network_g"])
    state_dict, used_key = _load_state_dict(checkpoint_path, param_key)
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    return net, cfg, used_key


def export_onnx(
    *,
    net: torch.nn.Module,
    onnx_path: Path,
    input_shape: tuple[int, int, int, int],
    opset: int,
) -> None:
    dummy_input = torch.zeros(*input_shape, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            net,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )


def load_raw_case(path: Path, raw_shape: tuple[int, int, int]) -> np.ndarray:
    x = np.fromfile(str(path), dtype=np.float32)
    expected = int(np.prod(raw_shape))
    if x.size != expected:
        raise ValueError(f"Raw sample size mismatch: expected {expected}, got {x.size}, path={path}")
    return x.reshape(raw_shape).astype(np.float32, copy=False)


def preprocess_case(raw: np.ndarray, max_value: float) -> np.ndarray:
    if raw.ndim != 3 or raw.shape[0] != 2:
        raise ValueError(f"Expected raw planar shape (2,H,W), got {raw.shape}")
    x = np.asarray(raw, dtype=np.float32).copy()
    x[1] = np.fliplr(x[1])
    x = x / float(max_value)
    return x[None, ...].astype(np.float32, copy=False)


def postprocess_output(y: np.ndarray, max_value: float) -> np.ndarray:
    x = np.asarray(y, dtype=np.float32)
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(f"Expected batched ONNX output with N=1, got {x.shape}")
        x = x[0]
    if x.shape[0] != 2:
        raise ValueError(f"Expected planar output shape (2,H,W), got {x.shape}")
    x = np.clip(x, 0.0, None)
    x = x.copy()
    x[1] = np.fliplr(x[1])
    return (x * float(max_value)).astype(np.float32, copy=False)


def verify_onnx(
    *,
    net: torch.nn.Module,
    onnx_path: Path,
    model_input: np.ndarray,
    max_value: float,
) -> tuple[ExportStats, np.ndarray, np.ndarray]:
    import onnxruntime as ort

    xt = torch.from_numpy(model_input)
    with torch.no_grad():
        torch_out = net(xt).cpu().numpy().astype(np.float32, copy=False)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {input_name: model_input})[0].astype(np.float32, copy=False)

    diff = np.abs(torch_out - ort_out)
    torch_post = postprocess_output(torch_out, max_value=max_value)
    ort_post = postprocess_output(ort_out, max_value=max_value)
    post_diff = np.abs(torch_post - ort_post)
    stats = ExportStats(
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        pytorch_min=float(torch_out.min()),
        pytorch_max=float(torch_out.max()),
        onnx_min=float(ort_out.min()),
        onnx_max=float(ort_out.max()),
        post_max_abs_diff=float(post_diff.max()),
        post_mean_abs_diff=float(post_diff.mean()),
    )
    return stats, ort_out, ort_post


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def save_dat_f32(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr, dtype=np.float32).tofile(str(path))


def build_compare_png(path: Path, raw: np.ndarray, denoised: np.ndarray, checkpoint_name: str) -> None:
    view_names = ["Anterior", "Posterior"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 16), constrained_layout=True)
    for i, name in enumerate(view_names):
        src = raw[i]
        dst = denoised[i]
        vmax = float(np.percentile(np.concatenate([src.ravel(), dst.ravel()]), 99.5))
        vmax = max(vmax, 1e-6)
        axes[i, 0].imshow(src, cmap="gray", vmin=0.0, vmax=vmax, aspect="auto")
        axes[i, 0].set_title(f"{name} input")
        axes[i, 1].imshow(dst, cmap="gray", vmin=0.0, vmax=vmax, aspect="auto")
        axes[i, 1].set_title(f"{name} denoised")
        for j in range(2):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            for spine in axes[i, j].spines.values():
                spine.set_visible(False)

    raw_sum = float(np.clip(raw, 0.0, None).sum())
    den_sum = float(np.clip(denoised, 0.0, None).sum())
    fig.suptitle(
        "Sample planar case\n"
        f"checkpoint={checkpoint_name}  total_counts: input={raw_sum:.1f}, denoised={den_sum:.1f}",
        fontsize=12,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_readme(
    *,
    model_name: str,
    checkpoint_path: Path,
    used_key: str,
    onnx_name: str,
    config_name: str,
    sample_input_name: str,
    sample_output_name: str,
    compare_png_name: str,
    input_shape: tuple[int, int, int, int],
    raw_shape: tuple[int, int, int],
    max_value: float,
    stats: ExportStats | None,
) -> str:
    verify_block = (
        f"- Verification max_abs_diff: `{stats.max_abs_diff:.8f}`\n"
        f"- Verification mean_abs_diff: `{stats.mean_abs_diff:.8f}`\n"
        f"- Postprocess max_abs_diff: `{stats.post_max_abs_diff:.8f}`\n"
        f"- Postprocess mean_abs_diff: `{stats.post_mean_abs_diff:.8f}`\n"
        if stats is not None
        else "- Verification: skipped\n"
    )
    return f"""
# Planar 2D ONNX MVP

This bundle contains the fixed-shape ONNX export for `{model_name}`.

## Files

- `{onnx_name}`: deployable ONNX model
- `{config_name}`: source experiment config used for export provenance
- `model_meta.json`: integration contract for preprocessing/postprocessing
- `export_report.json`: export provenance and verification stats
- `onnx_infer_example.py`: minimal NumPy + ONNX Runtime inference example
- `{sample_input_name}`: one raw sample case in original clinical format
- `{sample_output_name}`: ONNX output for that sample, in original orientation/count domain
- `{compare_png_name}`: quick before/after visualization for the sample

## Export Source

- Checkpoint: `{checkpoint_path}`
- State dict key: `{used_key}`
- Fixed ONNX input shape: `{list(input_shape)}`
- Raw input shape: `{list(raw_shape)}`
- Normalization divisor: `{max_value}`

## Runtime Contract

- Input tensor name: `input`
- Output tensor name: `output`
- Input dtype: `float32`
- Output dtype: `float32`
- ONNX tensor layout: `NCHW`
- Raw case layout before preprocess: `CHW`
- Channel semantics in raw file:
  - channel 0: anterior
  - channel 1: posterior in original orientation

## Preprocess

1. Read one raw planar case as `float32`.
2. Reshape to `{list(raw_shape)}`.
3. Flip channel 1 horizontally.
4. Divide by `{max_value}`.
5. Add batch dim to get `{list(input_shape)}`.

## Postprocess

1. Read output tensor `{list(input_shape)}`.
2. Remove batch dim to get `{list(raw_shape)}`.
3. Clip negatives to zero if desired.
4. Flip channel 1 back to original orientation.
5. Multiply by `{max_value}` if count-domain output is required.

## Validation

{verify_block}
""".strip()


def build_onnx_example(
    *,
    onnx_filename: str,
    raw_shape: tuple[int, int, int],
    max_value: float,
) -> str:
    c, h, w = raw_shape
    return f"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


CHANNELS = {c}
HEIGHT = {h}
WIDTH = {w}
MAX_VALUE = {max_value}


def load_raw_dat(path: str | Path) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    expected = CHANNELS * HEIGHT * WIDTH
    if arr.size != expected:
        raise ValueError(f"Expected {{expected}} float32 values, got {{arr.size}}")
    return arr.reshape(CHANNELS, HEIGHT, WIDTH).astype(np.float32, copy=False)


def preprocess(raw_case: np.ndarray) -> np.ndarray:
    if raw_case.shape != (CHANNELS, HEIGHT, WIDTH):
        raise ValueError(f"Expected shape ({{CHANNELS}}, {{HEIGHT}}, {{WIDTH}}), got {{raw_case.shape}}")
    x = raw_case.copy()
    x[1] = np.fliplr(x[1])
    x = x / MAX_VALUE
    return x[None, ...].astype(np.float32, copy=False)


def postprocess(output_tensor: np.ndarray) -> np.ndarray:
    if output_tensor.shape != (1, CHANNELS, HEIGHT, WIDTH):
        raise ValueError(f"Unexpected output shape: {{output_tensor.shape}}")
    x = np.clip(output_tensor[0], 0.0, None)
    x = x.copy()
    x[1] = np.fliplr(x[1])
    return x * MAX_VALUE


def run_one(model_path: str | Path, dat_path: str | Path) -> np.ndarray:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_input = preprocess(load_raw_dat(dat_path))
    model_output = sess.run(["output"], {{"input": model_input}})[0]
    return postprocess(model_output)


if __name__ == "__main__":
    bundle_dir = Path(__file__).resolve().parent
    model_path = bundle_dir / "{onnx_filename}"
    print("Bundle example ready. Call run_one(model_path, dat_path) from your integration code.")
""".strip()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_shape = tuple(int(v) for v in args.input_shape)
    raw_shape = tuple(int(v) for v in args.raw_shape)
    if input_shape[0] != 1:
        raise ValueError(f"Fixed MVP export expects batch size 1, got input_shape={input_shape}")
    if input_shape[1:] != raw_shape:
        raise ValueError(f"Expected input_shape[1:] == raw_shape, got {input_shape[1:]} vs {raw_shape}")

    model_name = args.config.stem
    onnx_path = output_dir / "model.onnx"
    copied_config_path = output_dir / args.config.name
    sample_input_name = f"sample_input_{args.sample_input.name}"
    sample_output_name = "sample_output_onnx_f32.dat"
    compare_png_name = "sample_before_after.png"

    net, cfg, used_key = build_model(args.config.resolve(), args.checkpoint.resolve(), args.param_key)
    export_onnx(net=net, onnx_path=onnx_path, input_shape=input_shape, opset=args.opset)

    raw_sample = load_raw_case(args.sample_input.resolve(), raw_shape=raw_shape)
    model_input = preprocess_case(raw_sample, max_value=float(args.max_value))

    stats = None
    ort_out = None
    sample_output = None
    if not args.skip_verify:
        stats, ort_out, sample_output = verify_onnx(
            net=net,
            onnx_path=onnx_path,
            model_input=model_input,
            max_value=float(args.max_value),
        )
    else:
        with torch.no_grad():
            yt = net(torch.from_numpy(model_input)).cpu().numpy()
        sample_output = postprocess_output(yt, max_value=float(args.max_value))

    shutil.copy2(args.config, copied_config_path)
    shutil.copy2(args.sample_input, output_dir / sample_input_name)
    save_dat_f32(output_dir / sample_output_name, sample_output)
    build_compare_png(
        output_dir / compare_png_name,
        raw=raw_sample,
        denoised=np.asarray(sample_output, dtype=np.float32),
        checkpoint_name=args.checkpoint.name,
    )

    meta = {
        "bundle_type": "planar2d_onnx_mvp_fixed_shape",
        "model_name": cfg.get("name", model_name),
        "network_type": cfg["network_g"]["type"],
        "checkpoint_path": str(args.checkpoint.resolve()),
        "state_dict_key": used_key,
        "onnx_model": onnx_path.name,
        "onnx_sha256": _sha256(onnx_path),
        "config_copy": copied_config_path.name,
        "sample_input": sample_input_name,
        "sample_output": sample_output_name,
        "sample_compare": compare_png_name,
        "input_tensor_name": "input",
        "output_tensor_name": "output",
        "input_shape": list(input_shape),
        "output_shape": list(input_shape),
        "input_dtype": "float32",
        "output_dtype": "float32",
        "layout": "NCHW",
        "raw_input_contract": {
            "file_pattern": "*.dat",
            "raw_dtype": "float32",
            "raw_shape": list(raw_shape),
            "raw_layout": "CHW",
            "channel_semantics": ["anterior", "posterior_original_orientation"],
        },
        "preprocess": {
            "posterior_flip_channel": 1,
            "normalize": {
                "type": "linear",
                "divide_by": args.max_value,
            },
            "add_batch_dim": True,
        },
        "postprocess": {
            "remove_batch_dim": True,
            "clip_min": 0.0,
            "undo_posterior_flip_channel": 1,
            "multiply_by": args.max_value,
            "note": "Multiply by max_value only if count-domain output is required.",
        },
        "verification": None if stats is None else asdict(stats),
    }
    write_json(output_dir / "model_meta.json", meta)

    report = {
        "config_path": str(args.config.resolve()),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "used_state_dict_key": used_key,
        "opset": args.opset,
        "input_shape": list(input_shape),
        "output_shape": list(input_shape),
        "raw_shape": list(raw_shape),
        "sample_input_path": str(args.sample_input.resolve()),
        "onnx_path": str(onnx_path),
        "onnx_sha256": meta["onnx_sha256"],
        "verification": None if stats is None else asdict(stats),
    }
    write_json(output_dir / "export_report.json", report)

    write_text(
        output_dir / "README.md",
        build_readme(
            model_name=cfg.get("name", model_name),
            checkpoint_path=args.checkpoint.resolve(),
            used_key=used_key,
            onnx_name=onnx_path.name,
            config_name=copied_config_path.name,
            sample_input_name=sample_input_name,
            sample_output_name=sample_output_name,
            compare_png_name=compare_png_name,
            input_shape=input_shape,
            raw_shape=raw_shape,
            max_value=float(args.max_value),
            stats=stats,
        ),
    )
    write_text(
        output_dir / "onnx_infer_example.py",
        build_onnx_example(
            onnx_filename=onnx_path.name,
            raw_shape=raw_shape,
            max_value=float(args.max_value),
        ),
    )

    print(f"Bundle ready: {output_dir}")
    print(f"ONNX: {onnx_path}")
    print(f"Sample input copied as: {sample_input_name}")
    print(f"Sample output saved as: {sample_output_name}")
    if stats is not None:
        print(
            "Verification "
            f"max_abs_diff={stats.max_abs_diff:.8f} "
            f"mean_abs_diff={stats.mean_abs_diff:.8f} "
            f"post_max_abs_diff={stats.post_max_abs_diff:.8f}"
        )


if __name__ == "__main__":
    main()
