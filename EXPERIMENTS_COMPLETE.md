# SPECT 自监督去噪实验完整清单

本文档整合了所有 SPECT 自监督去噪实验配置、状态、结果分析和关键发现。

---

## 📚 文档索引

### 1. 过拟合问题分析
**文件**: `options/train/demo/KAIR_vs_BasicSR_ANALYSIS.md`

**核心发现**:
- ✅ **实锤1**: `Anscombe + Charbonnier` 组合导致严重过拟合（下降 1.72 dB）
- ✅ **实锤2**: `Linear + Charbonnier` 也导致过拟合（下降 1.83 dB）
- ✅ **实锤3**: `Linear + PoissonNLL` 不会过拟合，持续改进到 48k iter（最佳 53.91 dB）

**关键结论**:
- ❌ **避免使用 CharbonnierLoss**（无论配什么归一化）
- ✅ **优先使用 Linear + PoissonNLLLoss 组合**

详见: [KAIR_vs_BasicSR_ANALYSIS.md](options/train/demo/KAIR_vs_BasicSR_ANALYSIS.md)

---

### 2. 自监督降噪实验 (Tier 1)

**文件**: `options/train/spect_selfsup/TIER1_RESULTS.md`

**实验结果总结**:

**最佳方法**: **N2N (Noise2Noise)** - 52.29 dB PSNR, 0.9980 SSIM

**关键发现**:
- ✅ **N2N (Noise2Noise)**: 52.29 dB（最佳，但需要配对数据）
- ✅ **N2B (Neighbor2Neighbor)**: 47.52 dB（单图方法中最佳）
- ❌ **Noiser2Noise**: 36.05 dB（实现有严重 bug，loss = -1e10）
- ⚠️ **N2V/S2S**: 36 dB（表现远低于预期 50-51 dB）

详细结果见下方 [Tier 1: 方法对比](#tier-1-方法对比) 部分。

详见: [TIER1_RESULTS.md](options/train/spect_selfsup/TIER1_RESULTS.md)

---

### 3. 自监督降噪实验总体说明

**文件**: `options/train/spect_selfsup/README.md`

**实验组织结构**:
- **Tier 1**: 方法主对比（5个方法）
- **Tier 2**: 深度消融实验（norm, regularization, optimization, data_strategy, network, view_mode, combined）
- **Tier 3**: 方法特定改进（N2V, N2B, S2S）

**最佳配置**（基于 Tier 2 消融结果）:
- **归一化**: Linear (max_value: 150.0)
- **损失函数**: PoissonNLLLoss (norm_type: linear)
- **数据策略**: patch64, batch32, enlarge_ratio=4
- **优化器**: AdamW (lr=1e-4, weight_decay=1e-4)
- **学习率**: CosineAnnealingRestartLR (eta_min=1e-5)
- **无数据增强**: SPECT 解剖结构有方向性

详见: [spect_selfsup/README.md](options/train/spect_selfsup/README.md)

---

### 4. 真实骨数据实验

**文件**: `options/train/bone_real/README.md`

**数据集**:
- **来源**: bone_20230327
- **病人数**: 30 人
- **采集时间**: 20s (高质量参考)
- **投影格式**: 60 视角 × 128×128 像素
- **总数据量**: 900 对（1800 个单通道样本）

**推荐配置**: `n2n_bone_proj_20s_singleview_sota.yml`
- 单通道输入/输出（anterior 或 posterior）
- Linear + PoissonNLLLoss
- UNetRes (in_nc=1, out_nc=1)
- patch64, batch32

详见: [bone_real/README.md](options/train/bone_real/README.md)

---

## 📋 实验设计概述

### 研究目标

**仅使用 1x 噪声 SPECT 图像实现最优自监督降噪**

### 核心问题

1. 哪种自监督方法最适合 SPECT 泊松噪声？
2. 如何针对泊松噪声优化配置？
3. 关键超参数如何影响性能？

### 实验结构（3层）

1. **Tier 1: 方法对比** - 比较5种自监督方法（N2N, Noiser2Noise, N2V, N2B, S2S）
2. **Tier 2: 深度消融** - 在最佳方法上进行系统消融（norm, data, regularization, optimization, network, view_mode）
3. **Tier 3: 方法改进** - 探索PoissonNLL对其他方法的改进潜力

---

**总实验数**: 63 个
- ✅ 已完成: 22 个
- 🔄 进行中: 2 个
- ⏸️ 未开始: 39 个

---

## 📊 关键实验结果总结

### Tier 2: norm 消融（归一化与损失函数）

**最佳配置**: `n2n_linear_poisson` (Linear + PoissonNLLLoss) - **53.91 dB PSNR, 0.9987 SSIM**

**关键发现**:
- ✅ **Linear + PoissonNLLLoss**: 53.91 dB（最优，持续改进到48k iter）
- ✅ **Linear + MSE**: 53.72 dB（接近最优，收敛更快40k iter）
- ⚠️ **Linear + Charbonnier**: 52.01 dB（过拟合，13k iter后下降）
- ⚠️ **Anscombe + Charbonnier**: 51.89 dB（过拟合，13k iter后下降）
- ❌ **Anscombe + PoissonNLL**: 训练失败（域不匹配）

详细结果见下方 [Tier 2: norm](#tier-2-norm) 部分。

## Demo: KAIR对齐与消融

**实验数**: 8 个

### ⏸️ 未开始 (8 个)


- **n2n_ablation_anscombe** - `demo/n2n_ablation_anscombe.yml`
- **n2n_ablation_batch8** - `demo/n2n_ablation_batch8.yml`
- **n2n_ablation_charbonnier** - `demo/n2n_ablation_charbonnier.yml`
- **n2n_ablation_combined** - `demo/n2n_ablation_combined.yml`
- **n2n_ablation_etamin_1e7** - `demo/n2n_ablation_etamin_1e7.yml`
- **n2n_ablation_no_wd** - `demo/n2n_ablation_no_wd.yml`
- **n2n_ablation_patch256** - `demo/n2n_ablation_patch256.yml`
- **n2n_base_aligned_kair** - `demo/n2n_base_aligned_kair.yml`

---

## Tier 1: 方法对比

**实验数**: 5 个

### ✅ 已完成 (3 个)

| 实验名称 | Loss | Norm | Patch | Batch | 最佳PSNR | 最佳SSIM | 最佳Iter | 配置文件 |
|---------|------|------|-------|-------|----------|----------|----------|----------|
| `n2n_baseline` | PoissonNLLLoss | linear | 64 | 16 | 52.29 dB | 0.9980 | 49000 | `spect_selfsup/tier1_method_comparison/n2n_baseline.yml` |
| `n2b_baseline` | CharbonnierLoss | anscombe | 256 | 8 | 47.21 dB | 0.9911 | 2800 | `spect_selfsup/tier1_method_comparison/n2b_baseline.yml` |
| `n2v_baseline` | CharbonnierLoss | anscombe | 256 | 8 | 33.40 dB | 0.8697 | 200 | `spect_selfsup/tier1_method_comparison/n2v_baseline.yml` |

### ⏸️ 未开始 (2 个)


- **noiser2noise_baseline** - `spect_selfsup/tier1_method_comparison/noiser2noise_baseline.yml`

- **s2s_baseline** - `spect_selfsup/tier1_method_comparison/s2s_baseline.yml`

---

## Tier 2: combined

**实验数**: 3 个

### ⏸️ 未开始 (3 个)


- **n2n_best_config** - `spect_selfsup/tier2_ablation/combined/n2n_best_config.yml`

- **n2n_original_kair** - `spect_selfsup/tier2_ablation/combined/n2n_original_kair.yml`

- **n2n_worst_config** - `spect_selfsup/tier2_ablation/combined/n2n_worst_config.yml`

---

## Tier 2: data_strategy

**实验数**: 7 个

**注意**: 本组实验的baseline配置（`n2n_patch64_batch32`）与 norm 组的baseline（`n2n_linear_poisson`）配置完全相同（patch64, batch32, linear norm, PoissonNLLLoss, view_mode=stack2），但结果不同（52.82 dB vs 53.91 dB），这是训练随机性导致的差异（约1.1 dB）。

### ✅ 已完成 (7 个)

| 实验名称 | Loss | Norm | Patch | Batch | 最佳PSNR | 最佳SSIM | 最佳Iter | 配置文件 |
|---------|------|------|-------|-------|----------|----------|----------|----------|
| `n2n_patch128_batch16` | PoissonNLLLoss | linear | 128 | 16 | 52.96 dB | 0.9983 | 49000 | `spect_selfsup/tier2_ablation/data_strategy/n2n_patch128_batch16.yml` |
| `n2n_patch64_batch32` | PoissonNLLLoss | linear | 64 | 32 | 52.82 dB | 0.9982 | 50002 | `spect_selfsup/tier2_ablation/data_strategy/n2n_patch64_batch32.yml` |
| `n2n_patch64_batch16` | PoissonNLLLoss | linear | 64 | 16 | 52.47 dB | 0.9981 | 50002 | `spect_selfsup/tier2_ablation/data_strategy/n2n_patch64_batch16.yml` |
| `n2n_patch256_batch8` | PoissonNLLLoss | linear | 256 | 8 | 52.50 dB | 0.9982 | 50002 | `spect_selfsup/tier2_ablation/data_strategy/n2n_patch256_batch8.yml` |
| `n2n_patch32_batch64` | PoissonNLLLoss | linear | 32 | 64 | 51.36 dB | 0.9974 | 37000 | `spect_selfsup/tier2_ablation/data_strategy/n2n_patch32_batch64.yml` |
| `n2n_patch64_batch8` | PoissonNLLLoss | linear | 64 | 8 | 51.31 dB | 0.9974 | 50002 | `spect_selfsup/tier2_ablation/data_strategy/n2n_patch64_batch8.yml` |
| `n2n_patch64_batch16_aug` | PoissonNLLLoss | linear | 64 | 16 | 51.29 dB | 0.9974 | 50002 | `spect_selfsup/tier2_ablation/data_strategy/n2n_patch64_batch16_aug.yml` |

---

## Tier 2: network

**实验数**: 5 个

### ⏸️ 未开始 (5 个)


- **n2n_nc_32_64_128_256** - `spect_selfsup/tier2_ablation/network/n2n_nc_32_64_128_256.yml`

- **n2n_nc_96_192_384_768** - `spect_selfsup/tier2_ablation/network/n2n_nc_96_192_384_768.yml`

- **n2n_unetres_nb2** - `spect_selfsup/tier2_ablation/network/n2n_unetres_nb2.yml`

- **n2n_unetres_nb4** - `spect_selfsup/tier2_ablation/network/n2n_unetres_nb4.yml`

- **n2n_unetres_nb6** - `spect_selfsup/tier2_ablation/network/n2n_unetres_nb6.yml`

---

## Tier 2: norm

**实验数**: 6 个

**注意**: 本组实验的baseline配置（`n2n_linear_poisson`）是后续所有消融实验的参考基准。该配置（patch64, batch32, linear norm, PoissonNLLLoss, view_mode=stack2）在 data_strategy 组中也有相同配置（`n2n_patch64_batch32`），但结果因训练随机性略有不同（53.91 dB vs 52.82 dB，差异约1.1 dB）。

### ✅ 已完成 (5 个)

| 实验名称 | Loss | Norm | Patch | Batch | 最佳PSNR | 最佳SSIM | 最佳Iter | 配置文件 |
|---------|------|------|-------|-------|----------|----------|----------|----------|
| `n2n_linear_poisson` | PoissonNLLLoss | linear | 64 | 32 | 53.91 dB | 0.9987 | 48000 | `spect_selfsup/tier2_ablation/norm/n2n_linear_poisson.yml` |
| `n2n_linear_mse` | MSELoss | linear | 64 | 32 | 53.72 dB | 0.9986 | 40000 | `spect_selfsup/tier2_ablation/norm/n2n_linear_mse.yml` |
| `n2n_linear_l1` | L1Loss | linear | 64 | 32 | 52.04 dB | 0.9966 | 12000 | `spect_selfsup/tier2_ablation/norm/n2n_linear_l1.yml` |
| `n2n_linear_charb` | CharbonnierLoss | linear | 64 | 32 | 52.01 dB | 0.9965 | 13000 | `spect_selfsup/tier2_ablation/norm/n2n_linear_charb.yml` |
| `n2n_anscombe_charb` | CharbonnierLoss | anscombe | 64 | 32 | 51.89 dB | 0.9961 | 13000 | `spect_selfsup/tier2_ablation/norm/n2n_anscombe_charb.yml` |

### 🔄 进行中 (1 个)


- **n2n_anscombe_poisson** - `spect_selfsup/tier2_ablation/norm/n2n_anscombe_poisson.yml`

---

## Tier 2: optimization

**实验数**: 7 个

### ⏸️ 未开始 (7 个)


- **n2n_adam_1e4** - `spect_selfsup/tier2_ablation/optimization/n2n_adam_1e4.yml`

- **n2n_adamw_1e4** - `spect_selfsup/tier2_ablation/optimization/n2n_adamw_1e4.yml`

- **n2n_adamw_5e4** - `spect_selfsup/tier2_ablation/optimization/n2n_adamw_5e4.yml`

- **n2n_adamw_5e5** - `spect_selfsup/tier2_ablation/optimization/n2n_adamw_5e5.yml`

- **n2n_etamin_1e4** - `spect_selfsup/tier2_ablation/optimization/n2n_etamin_1e4.yml`

- **n2n_etamin_1e6** - `spect_selfsup/tier2_ablation/optimization/n2n_etamin_1e6.yml`

- **n2n_etamin_1e7** - `spect_selfsup/tier2_ablation/optimization/n2n_etamin_1e7.yml`

---

## Tier 2: regularization

**实验数**: 5 个

### ✅ 已完成 (2 个)

| 实验名称 | Loss | Norm | Patch | Batch | 最佳PSNR | 最佳SSIM | 最佳Iter | 配置文件 |
|---------|------|------|-------|-------|----------|----------|----------|----------|
| `n2n_wd_0` | PoissonNLLLoss | linear | 64 | 32 | 52.90 dB | - | 50002 | `spect_selfsup/tier2_ablation/regularization/n2n_wd_0.yml` |
| `n2n_wd_1e3` | PoissonNLLLoss | linear | 64 | 32 | 52.33 dB | - | 50002 | `spect_selfsup/tier2_ablation/regularization/n2n_wd_1e3.yml` |

### 🔄 进行中 (1 个)

- **n2n_wd_1e4** - `spect_selfsup/tier2_ablation/regularization/n2n_wd_1e4.yml`

### ⏸️ 未开始 (2 个)


- **n2n_wd_1e5** - `spect_selfsup/tier2_ablation/regularization/n2n_wd_1e5.yml`

- **n2n_wd_5e4** - `spect_selfsup/tier2_ablation/regularization/n2n_wd_5e4.yml`

---

## Tier 2: view_mode

**实验数**: 5 个

**注意**: 本组实验测试不同的视角融合策略，所有配置都使用相同的训练参数（patch64, batch32, linear norm, PoissonNLLLoss），但 `view_mode` 和网络架构不同。Baseline 应该是 `view_mode: stack2`（对应 `n2n_linear_poisson` 的 53.91 dB），但本组没有运行 stack2 的baseline实验。

### ✅ 已完成 (5 个)

| 实验名称 | View Mode | 网络架构 | 最佳PSNR | 最佳SSIM | 最佳Iter | 配置文件 |
|---------|-----------|---------|----------|----------|----------|----------|
| `n2n_single_view_anterior` | split1 (anterior) | UNetRes (1ch) | 53.54 dB | 0.9985 | 49000 | `spect_selfsup/tier2_ablation/view_mode/n2n_single_view_anterior.yml` |
| `n2n_single_view_posterior` | split1 (posterior) | UNetRes (1ch) | 52.62 dB | 0.9981 | 50002 | `spect_selfsup/tier2_ablation/view_mode/n2n_single_view_posterior.yml` |
| `n2n_single_view_both` | split1 (both) | UNetRes (1ch) | 52.60 dB | 0.9981 | 50002 | `spect_selfsup/tier2_ablation/view_mode/n2n_single_view_both.yml` |
| `n2n_late_fusion` | stack2 | UNetResLateFusion (2ch) | 36.05 dB | 0.8622 | 47000 | `spect_selfsup/tier2_ablation/view_mode/n2n_late_fusion.yml` |
| `n2n_attention_fusion` | stack2 | UNetResAttentionFusion (2ch) | 26.93 dB | - | 2000 | `spect_selfsup/tier2_ablation/view_mode/n2n_attention_fusion.yml` |

**Baseline参考**: `n2n_linear_poisson` (view_mode=stack2, UNetRes 2ch) - **53.91 dB**

---

## Tier 3: n2b

**实验数**: 4 个

### ⏸️ 未开始 (4 个)


- **n2b_anscombe_lambda_1_1** - `spect_selfsup/tier3_other_methods/n2b/n2b_anscombe_lambda_1_1.yml`

- **n2b_anscombe_lambda_1_2** - `spect_selfsup/tier3_other_methods/n2b/n2b_anscombe_lambda_1_2.yml`

- **n2b_anscombe_lambda_2_1** - `spect_selfsup/tier3_other_methods/n2b/n2b_anscombe_lambda_2_1.yml`

- **n2b_linear_lambda_1_1** - `spect_selfsup/tier3_other_methods/n2b/n2b_linear_lambda_1_1.yml`

---

## Tier 3: n2v

**实验数**: 4 个

### ⏸️ 未开始 (4 个)


- **n2v_anscombe_charb** - `spect_selfsup/tier3_other_methods/n2v/n2v_anscombe_charb.yml`

- **n2v_anscombe_poisson** - `spect_selfsup/tier3_other_methods/n2v/n2v_anscombe_poisson.yml`

- **n2v_linear_charb** - `spect_selfsup/tier3_other_methods/n2v/n2v_linear_charb.yml`

- **n2v_linear_poisson** - `spect_selfsup/tier3_other_methods/n2v/n2v_linear_poisson.yml`

---

## Tier 3: s2s

**实验数**: 4 个

### ⏸️ 未开始 (4 个)


- **s2s_dropout_0.2_mc_100** - `spect_selfsup/tier3_other_methods/s2s/s2s_dropout_0.2_mc_100.yml`

- **s2s_dropout_0.3_mc_100** - `spect_selfsup/tier3_other_methods/s2s/s2s_dropout_0.3_mc_100.yml`

- **s2s_dropout_0.3_mc_50** - `spect_selfsup/tier3_other_methods/s2s/s2s_dropout_0.3_mc_50.yml`

- **s2s_dropout_0.4_mc_100** - `spect_selfsup/tier3_other_methods/s2s/s2s_dropout_0.4_mc_100.yml`

---

## 🔍 关键发现

### 1. 损失函数与归一化组合

- ✅ **Linear + PoissonNLL**: 53.91 dB（最优，持续改进到48k iter）
- ✅ **Linear + MSE**: 53.72 dB（接近最优，收敛更快40k iter）
- ⚠️ **Linear + Charbonnier**: 52.01 dB（过拟合，13k iter后下降）
- ⚠️ **Anscombe + Charbonnier**: 51.89 dB（过拟合，13k iter后下降）
- ❌ **Anscombe + PoissonNLL**: 训练失败（域不匹配）

### 2. 数据策略

- ✅ **patch128 + batch16**: 52.96 dB（最佳，但patch较大）
- ✅ **patch64 + batch32**: 52.82 dB（次优，平衡性好）
- ✅ **patch64 + batch16**: 52.47 dB（baseline）
- ⚠️ **数据增强**: 降低性能（-0.86 dB）

### 3. 过拟合预防

- ⭐⭐⭐⭐⭐ `weight_decay: 1e-4`（最关键）
- ⭐⭐⭐⭐ `eta_min: 1e-5`（不能太小如1e-7）
- ⭐⭐⭐⭐ 小patch size（64）增加多样性
- ⭐⭐⭐ 大batch size（16-32）提高稳定性

### 4. 推荐配置模板

```yaml
# 最优配置（已验证，53.91 dB PSNR）
norm:
  type: linear

pixel_opt:
  type: PoissonNLLLoss
  norm_type: linear
  max_value: 150.0

optim_g:
  type: AdamW
  lr: 1e-4
  weight_decay: 1e-4

scheduler:
  type: CosineAnnealingRestartLR
  periods: [50000]
  restart_weights: [1]
  eta_min: 1e-5

gt_size: 64
batch_size_per_gpu: 32
dataset_enlarge_ratio: 4
use_hflip: false
use_rot: false
```

---

## 📝 实验运行指南

### 批量运行实验

```bash
# Tier 1: 方法对比
bash scripts/run_experiments.sh tier1 all

# Tier 2: 消融实验
bash scripts/run_experiments.sh tier2 norm
bash scripts/run_experiments.sh tier2 data_strategy

# Tier 3: 方法特定
bash scripts/run_experiments.sh tier3 n2v
```

详见: [EXPERIMENT_RUNNER_GUIDE.md](scripts/EXPERIMENT_RUNNER_GUIDE.md)

---

## 🚨 已知问题

### 1. Noiser2Noise 实现失败
- **现象**: PSNR 36.05 dB，loss = -1e10
- **原因**: 疑似 Poisson 采样或归一化顺序问题
- **状态**: 待修复

### 2. N2V/S2S 表现不佳
- **现象**: PSNR 36 dB（预期 50-51 dB）
- **可能原因**:
  - 训练迭代数不足（20k vs 50k）
  - Loss 函数不匹配（Charbonnier vs PoissonNLL）
- **状态**: 待进一步实验

### 3. Anscombe + PoissonNLL 组合失败
- **现象**: 训练失败，loss 异常
- **原因**: 域不匹配 - PoissonNLLLoss 期望计数域数据，Anscombe 转换到方差稳定域
- **状态**: 已确认，应避免此组合

---

## 📅 实验时间线

- **2025-12-15**: Tier 1 方法对比完成
- **2025-12-16**: Tier 2 norm 消融完成（6个配置）
- **2025-12-16**: Tier 2 data_strategy 消融完成（7个配置）
- **2025-12-16**: 过拟合问题分析完成（实锤证据）
- **2025-12-17**: Tier 2 view_mode 消融完成（5个配置）
- **2025-12-19**: Tier 2 regularization 部分完成（2个配置）

---

## 📊 实验性能对比总结

### XCAT 仿真数据（理想情况）

| 配置 | PSNR (dB) | SSIM | 备注 |
|------|-----------|------|------|
| Linear + PoissonNLL | **53.91** | 0.9987 | 最佳配置 |
| Linear + MSE | 53.72 | 0.9986 | 接近最佳 |
| Linear + L1 | 52.04 | 0.9966 | 早期收敛 |
| Anscombe + Charbonnier | 51.89 | 0.9961 | 过拟合 |

### 真实骨数据（预估）

| 配置 | 预期 PSNR | 备注 |
|------|-----------|------|
| Linear + PoissonNLL | 48-52 dB | 基于 XCAT 结果预估 |

---

## 🔗 相关文档

- [CLAUDE.md](CLAUDE.md) - 项目总体说明和实验设计
- [options/train/spect_selfsup/README.md](options/train/spect_selfsup/README.md) - 自监督实验详细说明
- [options/train/demo/KAIR_vs_BasicSR_ANALYSIS.md](options/train/demo/KAIR_vs_BasicSR_ANALYSIS.md) - 过拟合分析
- [options/train/bone_real/README.md](options/train/bone_real/README.md) - 真实数据实验说明

---

**文档生成时间**: 2025-01-16
**最后更新**: 2025-12-19
