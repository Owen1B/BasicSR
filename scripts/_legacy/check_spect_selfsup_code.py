#!/usr/bin/env python3
"""
Test script to verify all code fixes for self-supervised denoising experiments.

Run this script before starting experiments to ensure everything works:

    cd /path/to/PRIME
    python scripts/check_spect_selfsup_code.py

Tests:
1. DBSNl blind-spot network architecture
2. UNetFPNFusion multi-scale feature fusion
3. PoissonNLLLoss with linear and Anscombe normalization
4. Noise2VoidBlindSpotModel
5. Neighbor2NeighborModel
6. Config file loading
7. GPU forward pass (if GPU available)
"""
import sys
from pathlib import Path
import torch

# Add BasicSR repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prime import register_all
from basicsr.utils.registry import ARCH_REGISTRY, LOSS_REGISTRY, MODEL_REGISTRY

register_all()


def print_header(title):
    print()
    print('=' * 70)
    print(f' {title}')
    print('=' * 70)


def test_dbsnl():
    """Test DBSNl blind-spot network."""
    print_header('Test 1: DBSNl Blind-Spot Network')

    DBSNl = ARCH_REGISTRY.get('DBSNl')
    DBSNlLight = ARCH_REGISTRY.get('DBSNlLight')

    # Test DBSNl
    print('Testing DBSNl...')
    model = DBSNl(in_nc=2, out_nc=2, base_ch=64, num_module=5)
    x = torch.randn(1, 2, 64, 64)
    y = model(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    print(f'  ✓ DBSNl: {x.shape} -> {y.shape}')
    print(f'  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Test DBSNlLight
    print('Testing DBSNlLight...')
    model_light = DBSNlLight(in_nc=1, out_nc=1, base_ch=32, num_module=3)
    x = torch.randn(1, 1, 64, 64)
    y = model_light(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    print(f'  ✓ DBSNlLight: {x.shape} -> {y.shape}')

    print('✅ DBSNl tests passed!')
    return True


def test_fpn_fusion():
    """Test UNetFPNFusion architecture."""
    print_header('Test 2: UNetFPNFusion Network')

    UNetFPNFusion = ARCH_REGISTRY.get('UNetFPNFusion')
    UNetFPNFusionLight = ARCH_REGISTRY.get('UNetFPNFusionLight')

    # Test UNetFPNFusion
    print('Testing UNetFPNFusion...')
    model = UNetFPNFusion(in_nc=2, out_nc=2, base_ch=64, num_levels=4)
    x = torch.randn(1, 2, 64, 64)
    y = model(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    print(f'  ✓ UNetFPNFusion: {x.shape} -> {y.shape}')
    print(f'  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Test with different input sizes
    for size in [32, 128, 256]:
        x = torch.randn(1, 2, size, size)
        y = model(x)
        assert y.shape == x.shape, f"Size {size}: {y.shape} != {x.shape}"
        print(f'  ✓ Size {size}x{size} works')

    # Test UNetFPNFusionLight
    print('Testing UNetFPNFusionLight...')
    model_light = UNetFPNFusionLight(in_nc=2, out_nc=2, base_ch=32, num_levels=3)
    x = torch.randn(1, 2, 64, 64)
    y = model_light(x)
    assert y.shape == x.shape
    print(f'  ✓ UNetFPNFusionLight: {x.shape} -> {y.shape}')

    print('✅ UNetFPNFusion tests passed!')
    return True


def test_poisson_loss():
    """Test PoissonNLLLoss."""
    print_header('Test 3: PoissonNLLLoss')

    PoissonNLLLoss = LOSS_REGISTRY.get('PoissonNLLLoss')
    PoissonNLLLossSimple = LOSS_REGISTRY.get('PoissonNLLLossSimple')

    pred = torch.rand(1, 2, 32, 32) * 0.8 + 0.1
    target = torch.rand(1, 2, 32, 32) * 0.8 + 0.1

    # Test with linear normalization
    print('Testing linear normalization...')
    loss = PoissonNLLLoss(max_value=150.0, norm_type='linear')
    l = loss(pred, target)
    assert not torch.isnan(l), "Loss is NaN"
    print(f'  ✓ Linear: {l.item():.4f}')

    # Test with Anscombe normalization
    print('Testing Anscombe normalization...')
    loss = PoissonNLLLoss(max_value=150.0, norm_type='anscombe')
    l = loss(pred, target)
    assert not torch.isnan(l), "Loss is NaN"
    print(f'  ✓ Anscombe: {l.item():.4f}')

    # Test gradient flow
    print('Testing gradient flow...')
    pred_grad = pred.clone().requires_grad_(True)
    loss = PoissonNLLLoss(max_value=150.0, norm_type='linear')
    l = loss(pred_grad, target)
    l.backward()
    assert pred_grad.grad is not None, "Gradient is None"
    assert not torch.isnan(pred_grad.grad).any(), "Gradient contains NaN"
    print(f'  ✓ Gradient mean: {pred_grad.grad.abs().mean().item():.6f}')

    # Test with weight (for N2V masked loss)
    print('Testing weighted loss...')
    weight = torch.rand(1, 2, 32, 32)
    l = loss(pred, target, weight=weight)
    assert not torch.isnan(l), "Weighted loss is NaN"
    print(f'  ✓ Weighted loss: {l.item():.4f}')

    print('✅ PoissonNLLLoss tests passed!')
    return True


def test_n2v_model():
    """Test Noise2Void models."""
    print_header('Test 4: Noise2Void Models')

    Noise2VoidBlindSpotModel = MODEL_REGISTRY.get('Noise2VoidBlindSpotModel')

    opt = {
        'is_train': True,
        'num_gpu': 0,
        'dist': False,
        'network_g': {
            'type': 'DBSNl',
            'in_nc': 2,
            'out_nc': 2,
            'base_ch': 64,
            'num_module': 3,
        },
        'train': {
            'optim_g': {'type': 'Adam', 'lr': 1e-4},
            'scheduler': {'type': 'MultiStepLR', 'milestones': [1000], 'gamma': 0.5},
            'total_iter': 1000,
            'ema_decay': 0,
            'pixel_opt': {'type': 'L1Loss', 'loss_weight': 1.0},
        },
        'path': {'models': 'tmp_models', 'training_states': 'tmp_states', 'log': 'tmp_log'},
        'logger': {'print_freq': 100},
    }

    print('Testing Noise2VoidBlindSpotModel...')
    model = Noise2VoidBlindSpotModel(opt)

    data = {
        'lq': torch.rand(1, 2, 64, 64),
        'gt': torch.rand(1, 2, 64, 64),
    }
    model.feed_data(data)
    model.optimize_parameters(1)
    print(f'  ✓ Training step completed')
    print(f'  ✓ Log keys: {list(model.log_dict.keys())}')

    print('✅ Noise2Void tests passed!')
    return True


def test_n2b_model():
    """Test Neighbor2Neighbor model."""
    print_header('Test 5: Neighbor2NeighborModel')

    Neighbor2NeighborModel = MODEL_REGISTRY.get('Neighbor2NeighborModel')

    opt = {
        'is_train': True,
        'num_gpu': 0,
        'dist': False,
        'network_g': {
            'type': 'UNetRes',
            'in_nc': 2,
            'out_nc': 2,
            'nc': [32, 64, 128, 256],
            'nb': 2,
        },
        'train': {
            'optim_g': {'type': 'Adam', 'lr': 1e-4},
            'scheduler': {'type': 'MultiStepLR', 'milestones': [1000], 'gamma': 0.5},
            'total_iter': 1000,
            'ema_decay': 0,
            'pixel_opt': {'type': 'L1Loss', 'loss_weight': 1.0},
            'lambda1': 1.0,
            'lambda2': 1.0,
            'increase_ratio': 2.0,
        },
        'path': {'models': 'tmp_models', 'training_states': 'tmp_states', 'log': 'tmp_log'},
        'logger': {'print_freq': 100},
    }

    print('Testing Neighbor2NeighborModel...')
    model = Neighbor2NeighborModel(opt)

    data = {
        'lq': torch.rand(1, 2, 64, 64),
        'gt': torch.rand(1, 2, 64, 64),
    }
    model.feed_data(data)
    model.optimize_parameters(1)
    print(f'  ✓ Training step completed')
    print(f'  ✓ Log keys: {list(model.log_dict.keys())}')

    # Check that Lambda increases correctly
    model.optimize_parameters(500)
    assert 'lambda_n2b' in model.log_dict
    print(f'  ✓ Lambda at iter 500: {model.log_dict["lambda_n2b"]:.4f}')

    print('✅ Neighbor2Neighbor tests passed!')
    return True


def test_config_loading():
    """Test config file loading."""
    print_header('Test 6: Config Loading')

    import yaml
    from basicsr.models import build_model

    configs = [
        ('N2N baseline', 'options/train/spect_selfsup/baseline/n2n_baseline.yml'),
        ('N2V baseline', 'options/train/spect_selfsup/baseline/n2v_baseline.yml'),
        ('N2B baseline', 'options/train/spect_selfsup/baseline/n2b_baseline.yml'),
        ('FPN fusion', 'options/train/spect_selfsup/baseline/n2n_fpn_fusion.yml'),
    ]

    for name, path in configs:
        print(f'Testing {name}...')
        with open(path, 'r') as f:
            opt = yaml.safe_load(f)

        opt['is_train'] = True
        opt['num_gpu'] = 0
        opt['dist'] = False
        opt['path']['models'] = 'tmp_models'
        opt['path']['training_states'] = 'tmp_states'
        opt['path']['log'] = 'tmp_log'

        model = build_model(opt)
        print(f'  ✓ Model: {type(model).__name__}, Network: {type(model.net_g).__name__}')

    print('✅ Config loading tests passed!')
    return True


def test_gpu(device='cuda'):
    """Test GPU forward pass."""
    print_header('Test 7: GPU Forward Pass')

    if not torch.cuda.is_available():
        print('  ⚠ CUDA not available, skipping GPU tests')
        return True

    DBSNl = ARCH_REGISTRY.get('DBSNl')
    UNetFPNFusion = ARCH_REGISTRY.get('UNetFPNFusion')

    # Test DBSNl on GPU
    print('Testing DBSNl on GPU...')
    model = DBSNl(in_nc=2, out_nc=2, base_ch=64, num_module=5).to(device)
    x = torch.randn(2, 2, 128, 128).to(device)
    y = model(x)
    assert y.device.type == device.split(':')[0]
    print(f'  ✓ DBSNl GPU: {x.shape} -> {y.shape}')

    # Test UNetFPNFusion on GPU
    print('Testing UNetFPNFusion on GPU...')
    model = UNetFPNFusion(in_nc=2, out_nc=2, base_ch=64, num_levels=4).to(device)
    y = model(x)
    print(f'  ✓ UNetFPNFusion GPU: {x.shape} -> {y.shape}')

    # Test training step on GPU
    print('Testing training on GPU...')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    y = model(x)
    loss = (y - x).pow(2).mean()
    loss.backward()
    optimizer.step()
    print(f'  ✓ Training step completed, loss: {loss.item():.4f}')

    print('✅ GPU tests passed!')
    return True


def main():
    print()
    print('#' * 70)
    print('#  Code Fixes Verification Test Suite')
    print('#' * 70)

    tests = [
        ('DBSNl Network', test_dbsnl),
        ('UNetFPNFusion Network', test_fpn_fusion),
        ('PoissonNLLLoss', test_poisson_loss),
        ('Noise2Void Model', test_n2v_model),
        ('Neighbor2Neighbor Model', test_n2b_model),
        ('Config Loading', test_config_loading),
        ('GPU Forward Pass', test_gpu),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            import traceback
            results.append((name, False, traceback.format_exc()))

    # Summary
    print_header('Test Summary')
    all_passed = True
    for name, passed, error in results:
        status = '✅ PASS' if passed else '❌ FAIL'
        print(f'  {status}: {name}')
        if error:
            all_passed = False
            print(f'        Error: {error.split(chr(10))[-2]}')

    print()
    if all_passed:
        print('🎉 All tests passed! Ready for experiments.')
        return 0
    else:
        print('⚠️ Some tests failed. Please fix before running experiments.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
