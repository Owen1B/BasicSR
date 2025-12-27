"""
Denoise projection data using trained model for OSEM reconstruction.

This script loads a trained model and denoises the first sample's 20s projection data,
saving the denoised projection sequence for subsequent OSEM reconstruction.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.utils.options import yaml_load


def load_projection_u16(path: str, views: int = 60, h: int = 128, w: int = 128) -> np.ndarray:
    """Load projection sequence from uint16 file."""
    arr = np.fromfile(path, dtype=np.uint16)
    expected = views * h * w
    if arr.size != expected:
        raise ValueError(f'Invalid projection size: {path}. Expected {expected} uint16 values, got {arr.size}.')
    return arr.reshape(views, h, w).astype(np.float32, copy=False)


@torch.no_grad()
def denoise_projection(
    net: torch.nn.Module,
    proj_u16: np.ndarray,
    max_value: float,
    device: str = 'cuda',
) -> np.ndarray:
    """Denoise projection sequence using trained model.

    Args:
        net: Trained network (net_g_ema)
        proj_u16: Original projection data (60, 128, 128) float32
        max_value: Normalization max value (from training config)
        device: Device to run inference on

    Returns:
        Denoised projection (60, 128, 128) float32
    """
    net.eval()
    net.to(device)

    denoised = np.zeros_like(proj_u16, dtype=np.float32)

    print(f"🔧 Denoising {proj_u16.shape[0]} views...")
    for i in tqdm(range(proj_u16.shape[0]), desc="Denoising"):
        # Normalize: [0, max_value] -> [0, 1]
        x = proj_u16[i].astype(np.float32, copy=False) / float(max_value)
        xt = torch.from_numpy(x[None, None, ...]).to(device=device, dtype=torch.float32)

        # Inference
        yt = net(xt)
        yt = torch.clamp(yt, min=0.0)  # Ensure non-negative

        # Denormalize: [0, 1] -> [0, max_value]
        y = yt.squeeze(0).squeeze(0).detach().cpu().numpy()
        denoised[i] = y * float(max_value)

    return denoised


def main():
    parser = argparse.ArgumentParser(description='Denoise projection data for OSEM reconstruction')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training config YAML file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file (default: latest checkpoint)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='datasets/bone_20230327/BianChengMing/20s-1/ProjectionImage1.dat',
        help='Path to input projection file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='spect_ct/results/denoised_projections',
        help='Output directory for denoised projection data'
    )
    parser.add_argument(
        '--use_ema',
        action='store_true',
        default=True,
        help='Use EMA model (default: True)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run inference on'
    )
    args = parser.parse_args()

    # Load config
    opt = yaml_load(args.config)

    # Get max_value from config
    max_value = float(opt['datasets']['train'].get('max_value', 100.0))
    print(f"📊 Using max_value: {max_value}")

    # Build network
    net_g = build_network(opt['network_g'])
    net_g_ema = build_network(opt['network_g']) if opt['train'].get('ema_decay', 0) > 0 else None

    # Load checkpoint
    if args.checkpoint is None:
        # Find latest checkpoint
        exp_name = opt['name']
        models_dir = Path('experiments') / exp_name / 'models'
        checkpoints = sorted(models_dir.glob('net_g_*.pth'), key=lambda x: int(x.stem.split('_')[-1]))
        if not checkpoints:
            raise FileNotFoundError(f'No checkpoints found in {models_dir}')
        checkpoint_path = checkpoints[-1]
        print(f"📂 Using latest checkpoint: {checkpoint_path.name}")
    else:
        checkpoint_path = Path(args.checkpoint)
        print(f"📂 Using specified checkpoint: {checkpoint_path.name}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model weights
    if 'params_ema' in checkpoint and args.use_ema and net_g_ema is not None:
        net_g_ema.load_state_dict(checkpoint['params_ema'], strict=True)
        net = net_g_ema
        print("✅ Loaded EMA model")
    elif 'params' in checkpoint:
        net_g.load_state_dict(checkpoint['params'], strict=True)
        net = net_g
        print("✅ Loaded net_g model")
    else:
        net_g.load_state_dict(checkpoint, strict=True)
        net = net_g
        print("✅ Loaded model from checkpoint")

    # Load projection data
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'Input projection file not found: {input_path}')

    print(f"📂 Loading projection from: {input_path}")
    proj_u16 = load_projection_u16(str(input_path), views=60, h=128, w=128)
    print(f"📊 Projection shape: {proj_u16.shape}, dtype: {proj_u16.dtype}")
    print(f"📊 Projection range: [{proj_u16.min():.2f}, {proj_u16.max():.2f}]")

    # Denoise
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
        print("⚠️  CUDA not available, using CPU")

    denoised = denoise_projection(net, proj_u16, max_value, device=device)
    print(f"📊 Denoised range: [{denoised.min():.2f}, {denoised.max():.2f}]")

    # Save denoised projection
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for this experiment
    exp_name = opt['name']
    checkpoint_name = checkpoint_path.stem  # e.g., "net_g_28000"
    sample_name = input_path.parent.parent.name  # e.g., "BianChengMing"
    subdir = output_dir / exp_name / sample_name
    subdir.mkdir(parents=True, exist_ok=True)

    # Save as int16 (same format as original)
    # 四舍五入后转换为 int16
    denoised_i16 = np.round(denoised).astype(np.int16)
    output_file = subdir / f"ProjectionImage1_denoised_{checkpoint_name}.dat"
    denoised_i16.tofile(str(output_file))

    print(f"✅ Saved denoised projection to: {output_file}")
    print(f"📊 File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"📊 Shape: {denoised_i16.shape}, dtype: {denoised_i16.dtype}")
    print(f"📊 Value range: [{denoised_i16.min()}, {denoised_i16.max()}]")

    # Also save a summary text file
    summary_file = subdir / f"ProjectionImage1_denoised_{checkpoint_name}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Denoised Projection Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Checkpoint: {checkpoint_name}\n")
        f.write(f"Sample: {sample_name}\n")
        f.write(f"Original file: {input_path}\n")
        f.write(f"Model: {opt['network_g']['type']}\n")
        f.write(f"Max value (normalization): {max_value}\n")
        f.write(f"Use EMA: {args.use_ema}\n")
        f.write(f"\n")
        f.write(f"Original projection:\n")
        f.write(f"  Shape: {proj_u16.shape}\n")
        f.write(f"  Range: [{proj_u16.min():.2f}, {proj_u16.max():.2f}]\n")
        f.write(f"  Mean: {proj_u16.mean():.2f}\n")
        f.write(f"  Std: {proj_u16.std():.2f}\n")
        f.write(f"\n")
        f.write(f"Denoised projection:\n")
        f.write(f"  Shape: {denoised_i16.shape}\n")
        f.write(f"  Range: [{denoised_i16.min()}, {denoised_i16.max()}]\n")
        f.write(f"  Mean: {denoised_i16.mean():.2f}\n")
        f.write(f"  Std: {denoised_i16.std():.2f}\n")
        f.write(f"\n")
        f.write(f"Output file: {output_file}\n")

    print(f"✅ Saved summary to: {summary_file}")


if __name__ == '__main__':
    main()

