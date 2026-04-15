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


DEFAULT_MODEL_NAME = "converged_tomo_3d_nview_multidose_N_scratch"
DEFAULT_CONFIG = PRIME_ROOT / "experiments" / DEFAULT_MODEL_NAME / "train_tomo_3d_nview_multidose_N_scratch.yml"
DEFAULT_CHECKPOINT = PRIME_ROOT / "experiments" / DEFAULT_MODEL_NAME / "models" / "net_g_164000.pth"
DEFAULT_OUTPUT_DIR = PRIME_ROOT / "deploy" / f"{DEFAULT_MODEL_NAME}_onnx_bundle"
DEFAULT_INPUT_SHAPE = (1, 1, 60, 128, 128)


@dataclass
class ExportStats:
    max_abs_diff: float
    mean_abs_diff: float
    pytorch_min: float
    pytorch_max: float
    onnx_min: float
    onnx_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PRIME SPECT model to an ONNX deployment bundle.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Experiment yaml with network_g config.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Checkpoint containing params_ema.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Bundle output directory.")
    parser.add_argument("--param-key", type=str, default="params_ema", help="Checkpoint key to load.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=5,
        default=DEFAULT_INPUT_SHAPE,
        metavar=("N", "C", "D", "H", "W"),
        help="Fixed input shape for export. Default: 1 1 60 128 128",
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
    input_shape: tuple[int, int, int, int, int],
    opset: int,
) -> None:
    dummy_input = torch.zeros(*input_shape, dtype=torch.float32)
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


def verify_onnx(
    *,
    net: torch.nn.Module,
    onnx_path: Path,
    input_shape: tuple[int, int, int, int, int],
) -> ExportStats:
    import onnxruntime as ort

    rng = np.random.default_rng(20260315)
    x = rng.random(input_shape, dtype=np.float32)
    with torch.no_grad():
        torch_out = net(torch.from_numpy(x)).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {input_name: x})[0]
    diff = np.abs(torch_out - ort_out)
    return ExportStats(
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        pytorch_min=float(torch_out.min()),
        pytorch_max=float(torch_out.max()),
        onnx_min=float(ort_out.min()),
        onnx_max=float(ort_out.max()),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_readme(
    *,
    model_name: str,
    checkpoint_path: Path,
    used_key: str,
    onnx_name: str,
    config_name: str,
    input_shape: tuple[int, int, int, int, int],
    max_value: float,
    stats: ExportStats | None,
) -> str:
    verify_block = (
        f"- Verification max_abs_diff: `{stats.max_abs_diff:.8f}`\n"
        f"- Verification mean_abs_diff: `{stats.mean_abs_diff:.8f}`\n"
        if stats is not None
        else "- Verification: skipped\n"
    )
    return f"""
# ONNX Deployment Bundle

This bundle contains the fixed-shape ONNX export for `{model_name}`.

## Files

- `{onnx_name}`: deployable ONNX model
- `{config_name}`: source experiment config used for export provenance
- `model_meta.json`: integration contract for preprocessing/postprocessing
- `export_report.json`: export provenance and verification stats
- `onnx_infer_example.py`: minimal NumPy + ONNX Runtime inference example
- `FOR_OTHER_AI.md`: handoff text for another coding agent

## Export Source

- Checkpoint: `{checkpoint_path}`
- State dict key: `{used_key}`
- Fixed ONNX input shape: `{list(input_shape)}`
- Output shape: `{list(input_shape)}`
- Normalization divisor: `{max_value}`

## Runtime Contract

- Input tensor name: `input`
- Output tensor name: `output`
- Input dtype: `float32`
- Output dtype: `float32`
- Input layout: `NCDHW`
- Semantic axes:
  - `N`: batch, fixed to 1 in this export
  - `C`: channel, fixed to 1
  - `D`: 60 SPECT views
  - `H`: 128
  - `W`: 128

## Preprocess

1. Read one raw projection file as `uint16`.
2. Reshape to `(60, 128, 128)`.
3. Convert to `float32`.
4. Divide by `{max_value}`.
5. Add batch and channel dims to get `(1, 1, 60, 128, 128)`.

## Postprocess

1. Read output tensor `(1, 1, 60, 128, 128)`.
2. Remove batch and channel dims to get `(60, 128, 128)`.
3. Clamp negatives to zero if desired.
4. Multiply by `{max_value}` if your software expects count-domain output.

## Validation

{verify_block}
""".strip()


def build_for_other_ai(
    *,
    model_name: str,
    onnx_name: str,
    input_shape: tuple[int, int, int, int, int],
    max_value: float,
) -> str:
    return f"""
你现在接手的是一个已经导出的 ONNX 模型，文件名是 `{onnx_name}`。

请按下面的接口契约接入，不要自行猜测预处理：

1. 输入是一个病例的完整 SPECT projection volume，不是 60 个独立文件。
2. 原始数据是 `uint16` 裸二进制 `.dat`，shape 固定为 `(60, 128, 128)`，顺序是 `view0 -> view59`，每个 view 内部是普通行优先顺序。
3. 送入 ONNX 前必须做线性归一化：`input_fp32 = raw_uint16.astype(float32) / {max_value}`。
4. ONNX 输入张量名是 `input`，shape 固定为 `{list(input_shape)}`，布局是 `NCDHW`。
5. 推理输出张量名是 `output`，shape 也是 `{list(input_shape)}`。
6. ONNX 原始输出仍然是归一化域，不是 count 域。若软件需要 count 域结果，请做：
   `output_count = clip(output, min=0) * {max_value}`。
7. 若要把结果写回 `.dat`，建议先裁负到 0，再转成目标格式；若要保持浮点结果，就保存为你们软件自己的浮点容器，不要强制转 `uint16`。

这个模型来自 `{model_name}`，是 3D UNet，语义上整块输入、整块输出：
`(1,1,60,128,128) -> (1,1,60,128,128)`。
""".strip()


def build_onnx_example(onnx_filename: str, max_value: float) -> str:
    return f"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


VIEWS = 60
HEIGHT = 128
WIDTH = 128
MAX_VALUE = {max_value}


def load_raw_dat(path: str | Path) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.uint16)
    expected = VIEWS * HEIGHT * WIDTH
    if arr.size != expected:
        raise ValueError(f"Expected {{expected}} uint16 values, got {{arr.size}}")
    return arr.reshape(VIEWS, HEIGHT, WIDTH).astype(np.float32, copy=False)


def preprocess(volume: np.ndarray) -> np.ndarray:
    if volume.shape != (VIEWS, HEIGHT, WIDTH):
        raise ValueError(f"Expected shape ({{VIEWS}}, {{HEIGHT}}, {{WIDTH}}), got {{volume.shape}}")
    return (volume / MAX_VALUE)[None, None, ...].astype(np.float32, copy=False)


def postprocess(output_tensor: np.ndarray) -> np.ndarray:
    if output_tensor.shape != (1, 1, VIEWS, HEIGHT, WIDTH):
        raise ValueError(f"Unexpected output shape: {{output_tensor.shape}}")
    volume = output_tensor[0, 0]
    volume = np.clip(volume, 0.0, None)
    return volume * MAX_VALUE


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

    model_name = args.config.stem
    onnx_path = output_dir / "model.onnx"
    copied_config_path = output_dir / args.config.name

    net, cfg, used_key = build_model(args.config.resolve(), args.checkpoint.resolve(), args.param_key)
    export_onnx(net=net, onnx_path=onnx_path, input_shape=tuple(args.input_shape), opset=args.opset)

    stats = None if args.skip_verify else verify_onnx(
        net=net,
        onnx_path=onnx_path,
        input_shape=tuple(args.input_shape),
    )

    shutil.copy2(args.config, copied_config_path)

    meta = {
        "bundle_type": "spect_onnx_fixed_shape",
        "model_name": cfg.get("name", model_name),
        "network_type": cfg["network_g"]["type"],
        "checkpoint_path": str(args.checkpoint.resolve()),
        "state_dict_key": used_key,
        "onnx_model": onnx_path.name,
        "onnx_sha256": _sha256(onnx_path),
        "config_copy": copied_config_path.name,
        "input_tensor_name": "input",
        "output_tensor_name": "output",
        "input_shape": list(args.input_shape),
        "output_shape": list(args.input_shape),
        "input_dtype": "float32",
        "output_dtype": "float32",
        "layout": "NCDHW",
        "raw_input_contract": {
            "file_pattern": "*_Proj4Filter.dat",
            "raw_dtype": "uint16",
            "raw_shape": [60, 128, 128],
            "raw_order": "view-major; within each view, row-major",
        },
        "preprocess": {
            "convert_to_float32": True,
            "normalize": {
                "type": "linear",
                "divide_by": args.max_value,
            },
            "add_batch_dim": True,
            "add_channel_dim": True,
        },
        "postprocess": {
            "remove_batch_dim": True,
            "remove_channel_dim": True,
            "clip_min": 0.0,
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
        "input_shape": list(args.input_shape),
        "output_shape": list(args.input_shape),
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
            input_shape=tuple(args.input_shape),
            max_value=args.max_value,
            stats=stats,
        ),
    )
    write_text(
        output_dir / "FOR_OTHER_AI.md",
        build_for_other_ai(
            model_name=cfg.get("name", model_name),
            onnx_name=onnx_path.name,
            input_shape=tuple(args.input_shape),
            max_value=args.max_value,
        ),
    )
    write_text(output_dir / "onnx_infer_example.py", build_onnx_example(onnx_path.name, args.max_value))

    print(f"Bundle ready: {output_dir}")
    print(f"ONNX: {onnx_path}")
    if stats is not None:
        print(f"Verification max_abs_diff={stats.max_abs_diff:.8f} mean_abs_diff={stats.mean_abs_diff:.8f}")


if __name__ == "__main__":
    main()
