from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

import numpy as np
import torch
from contextlib import nullcontext

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.spect_vis import save_spect_2view_grid
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # If running in test/val-only mode and user requests EMA evaluation, build/load net_g_ema for comparison.
        # (In upstream BasicSR SRModel, EMA is only set up in training; we extend it for evaluation convenience.)
        if not self.is_train:
            eval_networks = self.opt.get('val', {}).get('eval_networks', None)
            if eval_networks is not None:
                eval_networks = [str(x).lower() for x in eval_networks]
            if eval_networks and 'ema' in eval_networks and not hasattr(self, 'net_g_ema'):
                self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
                strict = self.opt['path'].get('strict_load_g', True)
                if load_path is not None:
                    # Prefer EMA weights if they exist; loader will fall back to 'params' if 'params_ema' missing.
                    self.load_network(self.net_g_ema, load_path, strict, 'params_ema')
                else:
                    # No pretrain provided: just copy current net_g weights so that EMA == G.
                    self.model_ema(0)
                self.net_g_ema.eval()

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # AMP settings (optional)
        amp_opt = train_opt.get('amp', {}) or {}
        self.amp_enable = bool(amp_opt.get('enable', False))
        self.amp_loss_fp32 = bool(amp_opt.get('loss_fp32', True))
        amp_dtype = str(amp_opt.get('dtype', 'bf16')).lower()  # bf16 | fp16
        self.amp_dtype = torch.bfloat16 if amp_dtype in ['bf16', 'bfloat16'] else torch.float16
        self.amp_use_scaler = bool(amp_opt.get('use_scaler', True)) and (self.amp_dtype == torch.float16)
        self.amp_autocast = nullcontext
        self.amp_scaler = None
        if self.amp_enable and self.device.type == 'cuda':
            try:
                from torch.cuda.amp import autocast, GradScaler
                self.amp_autocast = autocast
                if self.amp_use_scaler:
                    self.amp_scaler = GradScaler()
            except Exception:
                # Fallback: disable AMP if cuda.amp is unavailable
                self.amp_enable = False

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('tv_opt'):
            self.cri_tv = build_loss(train_opt['tv_opt']).to(self.device)
        else:
            self.cri_tv = None
        # Gradient loss (edge preservation)
        if train_opt.get('gradient_opt'):
            self.cri_gradient = build_loss(train_opt['gradient_opt']).to(self.device)
        else:
            self.cri_gradient = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        # Store per-sample vmax for PoissonNLLLoss (if available)
        if 'vmax' in data:
            # vmax might be a scalar, list, or tensor
            # For batch training, we might have multiple vmax values
            vmax = data['vmax']
            if isinstance(vmax, (int, float)):
                # Single value for entire batch (all samples have same vmax)
                self.vmax = torch.tensor(float(vmax), device=self.device)
            elif isinstance(vmax, (list, tuple)):
                # List of vmax values (one per sample in batch)
                # Convert to tensor on device
                self.vmax = torch.tensor([float(v) for v in vmax], device=self.device)
            elif isinstance(vmax, torch.Tensor):
                self.vmax = vmax.to(self.device)
            else:
                # numpy array or other
                try:
                    import numpy as np
                    if isinstance(vmax, np.ndarray):
                        self.vmax = torch.from_numpy(vmax).float().to(self.device)
                    else:
                        self.vmax = torch.tensor(float(vmax), device=self.device)
                except:
                    self.vmax = torch.tensor(float(vmax), device=self.device)
        else:
            self.vmax = None

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        if getattr(self, 'amp_enable', False):
            with self.amp_autocast(enabled=True, dtype=self.amp_dtype):
                self.output = self.net_g(self.lq)
        else:
            self.output = self.net_g(self.lq)

        pred = self.output
        gt = self.gt
        lq_ref = self.lq
        if getattr(self, 'amp_enable', False) and getattr(self, 'amp_loss_fp32', True):
            pred = pred.float()
            gt = gt.float()
            lq_ref = lq_ref.float()

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            # For PoissonNLLLoss, pass per-sample vmax if available
            if hasattr(self, 'vmax') and self.vmax is not None and 'PoissonNLL' in self.cri_pix.__class__.__name__:
                l_pix = self.cri_pix(pred, gt, vmax=self.vmax)
            else:
                l_pix = self.cri_pix(pred, gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(pred, gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # TV regularization loss
        if self.cri_tv:
            l_tv = self.cri_tv(pred)
            l_total += l_tv
            loss_dict['l_tv'] = l_tv
        # Gradient loss (edge preservation)
        # For N2N training, use input (lq) as reference instead of noisy GT
        if self.cri_gradient:
            if hasattr(self.cri_gradient, 'use_input_as_ref') and self.cri_gradient.use_input_as_ref:
                # Use input as reference (recommended for N2N)
                l_gradient = self.cri_gradient(pred, input_ref=lq_ref)
            else:
                # Use GT as reference (for supervised training)
                l_gradient = self.cri_gradient(pred, target=gt)
            l_total += l_gradient
            loss_dict['l_gradient'] = l_gradient

        if getattr(self, 'amp_enable', False) and getattr(self, 'amp_scaler', None) is not None:
            self.amp_scaler.scale(l_total).backward()
            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()
        else:
            l_total.backward()
            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def optimize_parameters_accumulation(self, current_iter: int, micro_step: int, accumulation_steps: int):
        """Gradient accumulation training step.

        - We scale the loss by 1/accumulation_steps before backward to keep effective LR unchanged.
        - We only call optimizer.step()/EMA/log update on the final micro step.
        """
        if accumulation_steps <= 1:
            # Fallback to normal behavior
            return self.optimize_parameters(current_iter)

        if micro_step == 1:
            self.optimizer_g.zero_grad()

        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        if self.cri_pix:
            if hasattr(self, 'vmax') and self.vmax is not None and 'PoissonNLL' in self.cri_pix.__class__.__name__:
                l_pix = self.cri_pix(self.output, self.gt, vmax=self.vmax)
            else:
                l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        if self.cri_tv:
            l_tv = self.cri_tv(self.output)
            l_total += l_tv
            loss_dict['l_tv'] = l_tv
        if self.cri_gradient:
            if hasattr(self.cri_gradient, 'use_input_as_ref') and self.cri_gradient.use_input_as_ref:
                l_gradient = self.cri_gradient(self.output, input_ref=self.lq)
            else:
                l_gradient = self.cri_gradient(self.output, target=self.gt)
            l_total += l_gradient
            loss_dict['l_gradient'] = l_gradient

        (l_total / float(accumulation_steps)).backward()

        if micro_step >= accumulation_steps:
            self.optimizer_g.step()
            self.log_dict = self.reduce_loss_dict(loss_dict)
            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        metric_domain_default = self.opt['val'].get('metric_domain', 'norm')  # norm | count
        # Evaluate which network(s) during validation.
        # Default behavior in BasicSR SRModel is to prefer EMA if it exists; we extend it to optionally
        # evaluate both raw G and EMA for easy comparison.
        eval_networks = self.opt['val'].get('eval_networks', None)
        if eval_networks is None:
            eval_networks = ['ema'] if hasattr(self, 'net_g_ema') else ['g']
        eval_networks = [str(x).lower() for x in eval_networks]
        # Which network(s) to save images for (to avoid doubling disk usage by default).
        save_img_networks = self.opt['val'].get('save_img_networks', None)
        if save_img_networks is None:
            save_img_networks = ['ema'] if 'ema' in eval_networks else eval_networks
        save_img_networks = [str(x).lower() for x in save_img_networks]

        if with_metrics:
            metric_names = list(self.opt['val']['metrics'].keys())
            # Store metrics per network tag with suffixed names (e.g., psnr_cnt_g / psnr_cnt_ema)
            metric_keys = [f'{m}_{tag}' for tag in eval_networks for m in metric_names]
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in metric_keys}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
            # Extend best_metric_results record to include suffixed metric keys
            record = self.best_metric_results[dataset_name]
            for tag in eval_networks:
                for m, content in self.opt['val']['metrics'].items():
                    key = f'{m}_{tag}'
                    if key in record:
                        continue
                    better = content.get('better', 'higher')
                    init_val = float('-inf') if better == 'higher' else float('inf')
                    record[key] = dict(better=better, val=init_val, iter=-1)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            total = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else len(dataloader)
            pbar = tqdm(total=total, unit='image')

        ds = dataloader.dataset
        spect_vis_opt = self.opt['val'].get('spect_vis', None)
        # Optional GIF generation for volume outputs (3D models).
        # To match SPECT3DModel config style, we reuse:
        #   val.generate_gif / val.gif_output_dir / val.max_gifs_per_val
        generate_gif = bool(self.opt.get('val', {}).get('generate_gif', False))
        gif_output_dir = self.opt.get('val', {}).get('gif_output_dir', None)
        if gif_output_dir is None:
            gif_output_dir = f"experiments/{self.opt.get('name', 'unknown')}/visualization/gifs"
        else:
            gif_output_dir = str(gif_output_dir).format(name=self.opt.get('name', 'unknown'))
        gif_max = int(self.opt.get('val', {}).get('max_gifs_per_val', 3))
        gif_fps = int(self.opt.get('val', {}).get('gif_fps', 10))  # optional

        def _save_volume_gif(save_path: str, lq_v: torch.Tensor, pred_v: torch.Tensor):
            """Save a simple 2-panel (LQ|Pred) per-view GIF for volume tensors.

            Args:
                lq_v/pred_v: [1,1,D,H,W] in [0,1] domain.
            """
            from PIL import Image
            import os
            os.makedirs(osp.dirname(save_path), exist_ok=True)

            lq_v = lq_v.detach().float().cpu().clamp_(0, 1)
            pred_v = pred_v.detach().float().cpu().clamp_(0, 1)
            D = int(lq_v.shape[2])

            frames: list[Image.Image] = []
            for d in range(D):
                lq = (lq_v[0, 0, d].numpy() * 255.0).round().astype('uint8')
                pr = (pred_v[0, 0, d].numpy() * 255.0).round().astype('uint8')
                canvas = np.concatenate([lq, pr], axis=1)
                frames.append(Image.fromarray(canvas, mode='L'))

            if not frames:
                return
            duration_ms = int(1000 / max(gif_fps, 1))
            frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)

        def _slice_batch(data_dict, b, batch_size):
            one = {}
            for k, v in data_dict.items():
                if hasattr(v, 'shape') and getattr(v, 'ndim', 0) >= 1 and int(v.shape[0]) == batch_size:
                    one[k] = v[b:b + 1, ...]
                elif isinstance(v, (list, tuple)) and len(v) == batch_size:
                    one[k] = [v[b]]
                else:
                    one[k] = v
            return one

        def _select_net(net_tag: str):
            if net_tag == 'ema':
                return getattr(self, 'net_g_ema', None)
            if net_tag == 'g':
                return self.net_g
            raise ValueError(f'Unknown validation network tag: {net_tag}')

        def _tensor2img_multi(t: torch.Tensor):
            """Convert tensor to image(s) for metric/visualization code.

            Supports:
            - 2D image: [1,C,H,W]
            - 3D volume: [1,C,D,H,W]  -> pick one D slice (view) then convert
            - edge-case: [1,D,H,W] (treat D as "views", pick one)
            """
            view_idx = int(self.opt.get('val', {}).get('volume_view_index', 0))

            # 3D volume: pick one view along D
            if t.ndim == 5:
                d = int(t.shape[2])
                vi = max(0, min(view_idx, d - 1))
                t = t[:, :, vi, :, :]
            # sometimes a volume may be returned as [1,D,H,W]
            if t.ndim == 4 and int(t.shape[1]) not in [1, 3] and int(t.shape[1]) == int(getattr(ds, 'views', -1)):
                d = int(t.shape[1])
                vi = max(0, min(view_idx, d - 1))
                t = t[:, vi:vi + 1, :, :]

            c = int(t.shape[1])
            if c in [1, 3]:
                return tensor2img([t])
            return [tensor2img([t[:, i:i + 1, ...]]) for i in range(c)]

        def _to_hwc_float(t: torch.Tensor) -> np.ndarray:
            """Convert [1,C,H,W] -> HWC float32 (no scaling/clipping)."""
            return t.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32, copy=False)

        def _maybe_inverse_count(x_norm_hwc: np.ndarray):
            if hasattr(ds, 'inverse_from_norm'):
                return np.clip(ds.inverse_from_norm(x_norm_hwc), 0.0, None)
            return None

        def _maybe_save_spect_grid(img_name: str, net_tag: str, lq_cnt: np.ndarray, pred_cnt: np.ndarray,
                                   gt_cnt: np.ndarray):
            if not (spect_vis_opt and spect_vis_opt.get('enable', False)):
                return
            if net_tag not in save_img_networks:
                return

            vmax_cnt = float(spect_vis_opt.get('vmax_cnt', getattr(ds, 'max_value', 1.0)))
            vlim_res = spect_vis_opt.get('vlim_res', None)
            if vlim_res is None:
                noise = lq_cnt - pred_cnt
                err = pred_cnt - gt_cnt
                vlim_res = float(np.max(np.abs([noise, err])))
                if vlim_res <= 0:
                    vlim_res = 1.0
            else:
                vlim_res = float(vlim_res)
            vlim_anscombe = spect_vis_opt.get('vlim_anscombe', None)

            if self.opt['is_train']:
                vis_path = osp.join(self.opt['path']['visualization'], img_name,
                                    f'{img_name}_{current_iter}_{net_tag}_grid.png')
            else:
                vis_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                    f"{img_name}_{self.opt['name']}_{net_tag}_grid.png")

            # Handle single-channel data: duplicate to 2-channel for visualization
            # This allows single-view experiments to use the same visualization code
            if lq_cnt.ndim == 3 and lq_cnt.shape[2] == 1:
                lq_cnt = np.repeat(lq_cnt, 2, axis=2)
                pred_cnt = np.repeat(pred_cnt, 2, axis=2)
                gt_cnt = np.repeat(gt_cnt, 2, axis=2)

            save_spect_2view_grid(
                save_path=vis_path,
                lq_cnt=lq_cnt,
                pred_cnt=pred_cnt,
                gt_cnt=gt_cnt,
                vmax_cnt=vmax_cnt,
                vlim_res=vlim_res,
                vlim_anscombe=vlim_anscombe,
            )

        def _maybe_save_img(img_name: str, net_tag: str, metric_img):
            """Save standard visualization image(s). For multi-net eval, only save for selected networks."""
            if not save_img:
                return
            if net_tag not in save_img_networks:
                return

            if self.opt['is_train']:
                save_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}.png')
            else:
                if self.opt['val']['suffix']:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                             f"{img_name}_{self.opt['val']['suffix']}.png")
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                             f"{img_name}_{self.opt['name']}.png")

            base, ext = osp.splitext(save_img_path)
            base = f'{base}_{net_tag}'
            if isinstance(metric_img, list):
                for i, img_i in enumerate(metric_img):
                    imwrite(img_i, f'{base}_c{i}{ext}')
            else:
                imwrite(metric_img, f'{base}{ext}')

        def _accumulate_metrics(net_tag: str, metric_data_norm: dict, metric_data_count: dict):
            if not with_metrics:
                return
            for name, opt_ in self.opt['val']['metrics'].items():
                out_key = f'{name}_{net_tag}'
                opt_domain = opt_.get('domain', metric_domain_default) if isinstance(opt_, dict) else metric_domain_default
                if opt_domain == 'count' and metric_data_count:
                    opt_type = opt_.get('type')
                    data_range = opt_.get('data_range', getattr(ds, 'max_value', None))
                    if data_range is None:
                        data_range = self.opt['val'].get('data_range', 1.0)
                    data_range = float(data_range)

                    if opt_type == 'calculate_lpips':
                        pred01 = (metric_data_count['img'] / data_range).astype(np.float32, copy=False)
                        gt01 = (metric_data_count['img2'] / data_range).astype(np.float32, copy=False)
                        pred01 = np.clip(pred01, 0.0, 1.0)
                        gt01 = np.clip(gt01, 0.0, 1.0)
                        self.metric_results[out_key] += calculate_metric({'img': pred01, 'img2': gt01}, opt_)
                    elif opt_type == 'calculate_poisson_fit':
                        self.metric_results[out_key] += calculate_metric({
                            'noise': metric_data_count['pred_noise'],
                            'gt': metric_data_count['img2'],
                        }, opt_)
                    else:
                        if 'data_range' not in opt_:
                            self.metric_results[out_key] += calculate_metric(
                                {'img': metric_data_count['img'], 'img2': metric_data_count['img2'], 'data_range': data_range},
                                opt_,
                            )
                        else:
                            self.metric_results[out_key] += calculate_metric(metric_data_count, opt_)
                else:
                    # norm domain
                    if isinstance(metric_data_norm.get('img'), list):
                        val = 0
                        for i in range(len(metric_data_norm['img'])):
                            val += calculate_metric({'img': metric_data_norm['img'][i], 'img2': metric_data_norm['img2'][i]},
                                                    opt_)
                        val /= len(metric_data_norm['img'])
                        self.metric_results[out_key] += val
                    else:
                        self.metric_results[out_key] += calculate_metric(metric_data_norm, opt_)

        num_images = 0
        num_volume_gifs = 0
        for idx, val_data in enumerate(dataloader):
            # val_data may contain a batch (e.g., when test.py mistakenly includes a train-phase dataloader).
            # Handle batch>1 robustly by iterating per-sample.
            batch_size = int(val_data['lq'].shape[0]) if hasattr(val_data.get('lq', None), 'shape') else 1
            for b in range(batch_size):
                # Slice one sample
                one = _slice_batch(val_data, b, batch_size)

                img_name = osp.splitext(osp.basename(one['lq_path'][0]))[0]
                self.feed_data(one)

                # Evaluate each requested network on the same input
                for net_tag in eval_networks:
                    net = _select_net(net_tag)
                    if net is None:
                        continue
                    net.eval()
                    with torch.no_grad():
                        # Handle channel mismatch: single-channel network with dual-channel validation data
                        # This happens when using split1 + force_stack_for_val
                        # Try to infer if network expects single channel by testing with 1-channel input
                        if self.lq.shape[1] == 2:
                            try:
                                # Try processing first channel only
                                _ = net(self.lq[:, 0:1, :, :])
                                # If successful, network expects 1 channel - process separately
                                out_ch0 = net(self.lq[:, 0:1, :, :])  # Anterior view
                                out_ch1 = net(self.lq[:, 1:2, :, :])  # Posterior view
                                self.output = torch.cat([out_ch0, out_ch1], dim=1)
                            except RuntimeError:
                                # Network expects 2 channels - process normally
                                self.output = net(self.lq)
                        else:
                            self.output = net(self.lq)
                    visuals = self.get_current_visuals()
                    # For 3D outputs, tensor2img can fail if we try to interpret the view dimension as channels.
                    # Only convert to images when we actually need it (save_img / metrics / spect_vis).
                    need_norm_img = bool(save_img) or bool(with_metrics)
                    metric_data_norm = {}
                    if need_norm_img:
                        metric_data_norm = {'img': _tensor2img_multi(visuals['result'])}
                        if 'gt' in visuals:
                            metric_data_norm['img2'] = _tensor2img_multi(visuals['gt'])

                    metric_data_count = {}
                    if 'gt' in visuals and hasattr(ds, 'inverse_from_norm'):
                        pred_cnt = _maybe_inverse_count(_to_hwc_float(visuals['result']))
                        gt_cnt = _maybe_inverse_count(_to_hwc_float(visuals['gt']))
                        lq_cnt = _maybe_inverse_count(_to_hwc_float(visuals['lq']))
                        if pred_cnt is not None and gt_cnt is not None and lq_cnt is not None:
                            metric_data_count['img'] = pred_cnt
                            metric_data_count['img2'] = gt_cnt
                            metric_data_count['pred_noise'] = lq_cnt - pred_cnt  # LQ - Pred
                            metric_data_count['lq'] = lq_cnt
                            _maybe_save_spect_grid(img_name, net_tag, lq_cnt=lq_cnt, pred_cnt=pred_cnt, gt_cnt=gt_cnt)

                    if need_norm_img:
                        _maybe_save_img(img_name, net_tag, metric_data_norm.get('img'))
                    _accumulate_metrics(net_tag, metric_data_norm, metric_data_count)

                    # GIF for 3D volume outputs (uses the same config keys as SPECT3DModel)
                    if (
                        generate_gif
                        and num_volume_gifs < gif_max
                        and isinstance(visuals.get('result', None), torch.Tensor)
                        and visuals['result'].ndim == 5
                        and int(visuals['result'].shape[1]) == 1
                    ):
                        sample_dir = osp.join(gif_output_dir, img_name)
                        gif_path = osp.join(sample_dir, f'{self.opt.get("name","unknown")}_iter{current_iter}_{net_tag}.gif')
                        _save_volume_gif(gif_path, visuals['lq'], visuals['result'])
                        num_volume_gifs += 1

                # tentative for out of GPU memory
                del self.lq
                del self.output
                torch.cuda.empty_cache()
                if 'gt' in visuals:
                    del self.gt

                num_images += 1
                if use_pbar:
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= max(num_images, 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            # Optionally save a "best" checkpoint when the chosen metric hits a new best at this iteration.
            # This writes/overwrites `net_g_best.pth` and does not create per-iter files.
            if bool(self.opt.get('val', {}).get('save_best_ckpt', False)):
                best_metric = self.opt.get('val', {}).get('best_metric', None)
                record = getattr(self, 'best_metric_results', {}).get(dataset_name, {})
                if best_metric is None:
                    # Default: first metric + first evaluated network tag (metrics are stored with suffix).
                    metric_names = list(self.opt['val']['metrics'].keys())
                    best_metric = metric_names[0] if metric_names else None
                    if best_metric is not None and isinstance(eval_networks, list) and len(eval_networks) > 0:
                        best_metric = f'{best_metric}_{eval_networks[0]}'
                else:
                    # Allow passing metric name without suffix when only one network is evaluated.
                    if best_metric not in record and isinstance(eval_networks, list) and len(eval_networks) == 1:
                        cand = f'{best_metric}_{eval_networks[0]}'
                        if cand in record:
                            best_metric = cand

                if best_metric in record and int(record[best_metric].get('iter', -1)) == int(current_iter):
                    logger = get_root_logger()
                    logger.info(f'Saving best checkpoint: {best_metric} = {record[best_metric]["val"]:.6f} @ {current_iter}')
                    if hasattr(self, 'net_g_ema'):
                        self.save_network([self.net_g, self.net_g_ema], 'net_g', 'best', param_key=['params', 'params_ema'])
                    else:
                        self.save_network(self.net_g, 'net_g', 'best')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
            # Also log best metric values
            if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
                for metric, record in self.best_metric_results[dataset_name].items():
                    tb_logger.add_scalar(f'best_metrics/{dataset_name}/{metric}', record['val'], current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
