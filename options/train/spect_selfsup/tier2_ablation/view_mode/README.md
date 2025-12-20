# View Mode Ablation 实验

## 📋 实验概述

测试不同的视角融合策略对SPECT去噪性能的影响。

## 🎯 实验配置

### 已就绪的配置 (2个)

| 配置 | 架构 | 融合策略 | 说明 |
|------|------|---------|------|
| **n2n_late_fusion.yml** | UNetResLateFusion | 晚期融合 | 双分支encoder独立处理，bottleneck融合特征 |
| **n2n_attention_fusion.yml** | UNetResAttentionFusion | 注意力融合 | 学习空间自适应的注意力权重来融合视角 |

### 暂时搁置的配置 (3个)

| 配置 | 说明 | 原因 |
|------|------|------|
| n2n_single_view_both.yml | 单视角训练（两个视角分开） | 需要修复validation维度不匹配问题 |
| n2n_single_view_anterior.yml | 仅前视角 | 同上 |
| n2n_single_view_posterior.yml | 仅后视角 | 同上 |

## 🚀 运行实验

### 使用统一的训练脚本 (推荐)

```bash
cd /home/owen/code/BasicSR

# 运行所有view_mode实验
bash scripts/run_experiments.sh tier2 view_mode
```

**特点**:
- ✅ 自动跳过已完成的实验
- ✅ 自动resume中断的实验
- ✅ 统一的实验管理
- ✅ 失败容错，继续下一个实验

### 单独运行某个实验

```bash
cd /home/owen/code/BasicSR

# Late Fusion
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 \
  python3 basicsr/train.py \
  -opt options/train/spect_selfsup/tier2_ablation/view_mode/n2n_late_fusion.yml \
  --auto_resume

# Attention Fusion
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 \
  python3 basicsr/train.py \
  -opt options/train/spect_selfsup/tier2_ablation/view_mode/n2n_attention_fusion.yml \
  --auto_resume
```

## ⏱️ 预计时间

- **每个实验**: ~2.2小时 (50k iterations)
- **总计**: ~4-5小时 (2个实验)

## 📊 配置细节

所有配置使用当前已验证的最优设置:

### 数据配置
- **归一化**: Linear (保留泊松特性)
- **Patch size**: 64
- **Batch size**: 32
- **数据增强**: 关闭 (SPECT特性)

### 训练配置
- **损失函数**: PoissonNLLLoss (泊松噪声最优)
- **优化器**: AdamW (lr=1e-4, weight_decay=1e-4)
- **学习率调度**: CosineAnnealing (eta_min=1e-5)
- **总迭代数**: 50,000

### 架构细节

#### Late Fusion (UNetResLateFusion)
```
输入 (2ch) → 分离为 anterior (1ch) + posterior (1ch)
           ↓
前视角 encoder → 特征1
后视角 encoder → 特征2
           ↓
         Concat融合
           ↓
      共享 decoder
           ↓
      输出 (2ch)
```

**优势**: 每个视角独立学习特征，允许不同的特征提取策略

#### Attention Fusion (UNetResAttentionFusion)
```
输入 (2ch) → 分离为 anterior (1ch) + posterior (1ch)
           ↓
前视角 encoder → 特征1
后视角 encoder → 特征2
           ↓
    注意力模块 (学习权重)
         α1, α2
           ↓
  加权融合: α1×特征1 + α2×特征2
           ↓
      共享 decoder
           ↓
      输出 (2ch)
```

**优势**: 自适应学习哪个视角在哪个位置更可靠

## 🔍 预期结果

### 性能排名 (推测)

| 策略 | 预期PSNR | 理由 |
|------|----------|------|
| Attention Fusion | ~52.5-53 dB | 最灵活，自适应权重 |
| Late Fusion | ~52-52.5 dB | 独立特征学习 |
| Baseline (stack2) | ~52 dB | 早期融合参考 |

### 对比基准

- **n2n_linear_poisson** (norm最优): 53.83 dB
- **n2n_patch128_batch16** (data最优): 52.95 dB

**注意**: view_mode实验使用patch64+batch32 (非最优data config)，所以性能可能略低于53.83 dB

## 📝 实验验证

### 已验证项 ✅

- [x] 配置文件YAML语法正确
- [x] 网络架构已注册 (UNetResLateFusion, UNetResAttentionFusion)
- [x] Debug测试通过 (100 iter)
- [x] 损失函数兼容性验证
- [x] 数据加载正常

### 架构文件位置

- `basicsr/archs/unet_latefusion_arch.py`
- `basicsr/archs/unet_attentionfusion_arch.py`

## 📈 结果查看

### 训练日志
```bash
tail -f experiments/n2n_late_fusion/train_*.log
tail -f experiments/n2n_attention_fusion/train_*.log
```

### TensorBoard
```bash
tensorboard --logdir experiments/n2n_late_fusion/tb_logger
tensorboard --logdir experiments/n2n_attention_fusion/tb_logger
```

### Wandb
项目: `spect_selfsup_denoising`
- n2n_late_fusion
- n2n_attention_fusion

### 提取最佳指标
```bash
grep "Best:" experiments/n2n_late_fusion/train_*.log | tail -5
grep "Best:" experiments/n2n_attention_fusion/train_*.log | tail -5
```

## ⚠️ 注意事项

1. **GPU内存**: 每个实验需要约10GB显存
2. **Resume支持**: 中断后重新运行会自动恢复
3. **Checkpoint保存**: 只保留最近2个checkpoint (节省空间)
4. **单视角配置**: 暂时不运行，等技术问题解决后再测试

## 🎓 科研价值

这些实验对论文的贡献:

1. **Late Fusion**:
   - 测试独立视角特征学习的效果
   - 与early fusion (baseline) 对比

2. **Attention Fusion**:
   - 展示自适应融合的优势
   - 可视化注意力权重（哪个位置用哪个视角）
   - 理论上应该最优

3. **对比意义**:
   - 如果fusion策略性能接近baseline → 说明简单的early fusion已经足够
   - 如果attention fusion显著更好 → 说明自适应融合有价值
   - 如果late fusion更好 → 说明独立特征学习重要

---

**准备就绪！现在可以运行实验了。**

运行命令:
```bash
bash scripts/run_experiments.sh tier2 view_mode
```

