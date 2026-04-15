# Converged Train Configs

主线统一为 6 个命名清晰的模式：

1. `train_planar_2d_1view.yml`
- 平片数据，单视角 2D。
- `view_sampler.mode=single_view` + `pack_mode=2d_channel`，输入 `(B,1,H,W)`。

2. `train_planar_2d_2view.yml`
- 平片数据，对位双视角 2D（A/P）。
- `view_sampler.mode=opposite_pair` + `pack_mode=2d_channel`，输入 `(B,2,H,W)`。

3. `train_tomo_2d_1view.yml`
- 断层数据，单视角 2D 基线。
- `single_view + 2d_channel`，输入 `(B,1,H,W)`。

4. `train_tomo_2d_2view.yml`
- 断层数据，对位视角对 2D 融合（`offset=V/2`）。
- `opposite_pair + 2d_channel`，输入 `(B,2,H,W)`。

5. `train_tomo_2d_nview.yml`
- 断层数据，多视角通道堆叠 2D（默认 `n=60`）。
- `n_views + 2d_channel`，输入 `(B,n,H,W)`。

6. `train_tomo_3d_nview.yml`
- 断层数据，多视角 3D 卷积。
- `n_views + 3d_depth`，输入 `(B,1,D,H,W)`。

通用约束：
- `views/height/width` 均由配置决定，不做硬编码。
- `H/W` 建议可被 8 整除（UNet 三次下采样）。
- 同一 batch 内样本形状必须一致。

验证策略：
- `SPECTProjectionEvalDataset` 用轻量验证（PLL），可配合 `scripts/async_eval_checkpoints.py` 异步评测。
- 无参考指标时建议 `save_best_ckpt: false`，避免伪 best。
