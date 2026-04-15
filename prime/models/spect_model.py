from __future__ import annotations

from collections import OrderedDict
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from prime.pipeline.eval_models_mp4 import ModelPredictor, generate_eval_mp4_from_projection
from prime.pipeline.inference import denoise_views


def _to_int_iter(current_iter: Any) -> int:
    try:
        return int(current_iter)
    except Exception:
        return 0


def _extract_projection(val_data: dict) -> np.ndarray | None:
    """Extract one projection volume as float32 array in shape (V, H, W)."""
    for key in ("proj_sequence", "lq"):
        if key not in val_data:
            continue
        data = val_data[key]

        if isinstance(data, np.ndarray):
            arr = data
        elif isinstance(data, torch.Tensor):
            arr = data.detach().cpu().numpy()
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            head = data[0]
            if isinstance(head, np.ndarray):
                arr = head
            elif isinstance(head, torch.Tensor):
                arr = head.detach().cpu().numpy()
            else:
                continue
        else:
            continue

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 5:  # [B,C,V,H,W]
            if arr.shape[0] < 1 or arr.shape[1] < 1:
                return None
            return arr[0, 0]
        if arr.ndim == 4:  # [B,V,H,W] or [C,V,H,W]
            return arr[0]
        if arr.ndim == 3:  # [V,H,W]
            return arr
        return None
    return None


def _extract_model_input(val_data: dict) -> np.ndarray | None:
    """Extract model input volume as float32 array in shape (V,H,W) or (C,V,H,W)."""
    if "lq" not in val_data:
        return None
    data = val_data["lq"]

    if isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    elif isinstance(data, (list, tuple)) and len(data) > 0:
        head = data[0]
        if isinstance(head, np.ndarray):
            arr = head
        elif isinstance(head, torch.Tensor):
            arr = head.detach().cpu().numpy()
        else:
            return None
    else:
        return None

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 5:  # [B,C,V,H,W]
        if arr.shape[0] < 1:
            return None
        return arr[0]
    if arr.ndim == 4:  # [B,V,H,W] or [C,V,H,W]
        if arr.shape[0] == 1:
            return arr[0]
        return arr
    if arr.ndim == 3:  # [V,H,W]
        return arr
    return None


def _extract_path_str(val_data: dict, key: str) -> str | None:
    if key not in val_data:
        return None
    data = val_data[key]
    if isinstance(data, (list, tuple)) and len(data) > 0:
        data = data[0]
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, str):
        return data
    return None


def _parse_thin_factors(value: Any) -> list[float]:
    if value is None:
        return [2.0, 3.0, 4.0, 5.0]
    if isinstance(value, (list, tuple)):
        out = [float(x) for x in value]
    else:
        out = []
        for p in str(value).split(","):
            p = p.strip()
            if p:
                out.append(float(p))
    return [k for k in out if float(k) > 1.0]


def _weighted_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = torch.abs(pred - target)
    if channel_weights is not None:
        diff = diff * channel_weights
    return diff.mean()


@MODEL_REGISTRY.register()
class SPECTModel(SRModel):
    """SPECT task model: official SRModel training + lightweight projection validation.

    - Base training/optimizer/scheduler/save logic stays in upstream SRModel.
    - Only override `nondist_validation` for SPECTProjectionEvalDataset.
    - Heavy visualization/analysis should run in external scripts.
    """

    def __init__(self, opt: dict):
        super().__init__(opt)
        self.logger = get_root_logger()
        self._warned_no_metric_for_best = False
        self._unsupported_metric_warned: set[str] = set()
        self._missing_eval_net_warned: set[str] = set()
        self._eval_mp4_warned = False
        self._eval_mp4_async_opt_warned = False
        self._view_fusion_reg_opt = dict((self.opt.get("train", {}) or {}).get("view_fusion_regularizer", {}) or {})
        self._support_reg_opt = dict((self.opt.get("train", {}) or {}).get("support_condition_regularizer", {}) or {})
        self._maybe_apply_custom_warmstart()

    def _maybe_apply_custom_warmstart(self) -> None:
        path_opt = self.opt.get("path", {}) or {}
        warmstart_path = path_opt.get("warmstart_network_g", None)
        if not warmstart_path:
            return

        bare_net = self.get_bare_model(self.net_g)
        if not hasattr(bare_net, "warmstart_from_legacy_unetres"):
            self.logger.warning(
                "warmstart_network_g is set, but current network does not support custom legacy warm-start."
            )
            return

        warm_param_key = str(path_opt.get("warmstart_param_key_g", "params_ema"))
        warm_report = bare_net.warmstart_from_legacy_unetres(warmstart_path, param_key=warm_param_key)
        self.logger.info(
            "[warmstart] net_g <- %s (%s), reused=%s skipped=%s",
            warm_report["path"],
            warm_report["param_key"],
            warm_report["reused_keys"],
            warm_report["skipped_keys"],
        )
        if hasattr(self, "net_g_ema"):
            bare_ema = self.get_bare_model(self.net_g_ema)
            if hasattr(bare_ema, "warmstart_from_legacy_unetres"):
                warm_report_ema = bare_ema.warmstart_from_legacy_unetres(warmstart_path, param_key=warm_param_key)
                self.logger.info(
                    "[warmstart] net_g_ema <- %s (%s), reused=%s skipped=%s",
                    warm_report_ema["path"],
                    warm_report_ema["param_key"],
                    warm_report_ema["reused_keys"],
                    warm_report_ema["skipped_keys"],
                )

    def _supports_fusion_controls(self) -> bool:
        return bool(getattr(self.get_bare_model(self.net_g), "supports_fusion_controls", False))

    def _supports_aux_outputs(self) -> bool:
        bare = self.get_bare_model(self.net_g)
        return bool(
            getattr(bare, "supports_aux_outputs", False)
            or getattr(bare, "supports_fusion_controls", False)
        )

    def _should_return_aux(self) -> bool:
        if not self._supports_aux_outputs():
            return False
        if bool(self._view_fusion_reg_opt.get("enable", False)):
            return True
        if not bool(self._support_reg_opt.get("enable", False)):
            return False
        if float(self._support_reg_opt.get("gate_sparsity_weight", 0.0)) > 0.0:
            return True
        return False

    def _forward_train_network(
        self,
        x: torch.Tensor,
        *,
        return_aux: bool = False,
        disable_fusion: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        kwargs: dict[str, Any] = {}
        if self._supports_aux_outputs():
            kwargs["return_aux"] = return_aux
        if self._supports_fusion_controls():
            kwargs["disable_fusion"] = disable_fusion

        if getattr(self, "amp_enable", False):
            with self.amp_autocast(enabled=True, dtype=self.amp_dtype):
                output = self.net_g(x, **kwargs) if kwargs else self.net_g(x)
        else:
            output = self.net_g(x, **kwargs) if kwargs else self.net_g(x)

        if isinstance(output, tuple):
            return output[0], output[1]
        return output, None

    def feed_data(self, data):
        super().feed_data(data)
        self.support_mask = data["support_mask"].to(self.device) if "support_mask" in data else None
        self.support_distance = data["support_distance"].to(self.device) if "support_distance" in data else None
        self.support_boundary_weight = (
            data["support_boundary_weight"].to(self.device) if "support_boundary_weight" in data else None
        )

    def _build_channel_weights(self, ref: torch.Tensor) -> torch.Tensor | None:
        posterior_weight = float(self._view_fusion_reg_opt.get("posterior_channel_weight", 1.0))
        if posterior_weight == 1.0 or ref.ndim != 4 or int(ref.shape[1]) < 2:
            return None
        weights = torch.ones((1, int(ref.shape[1]), 1, 1), device=ref.device, dtype=ref.dtype)
        weights[:, 1] = posterior_weight
        return weights

    @torch.no_grad()
    def _self_only_reference(self, ref_input: torch.Tensor) -> torch.Tensor | None:
        if not self._supports_fusion_controls():
            return None
        reference, _ = self._forward_train_network(ref_input, return_aux=False, disable_fusion=True)
        return reference.detach()

    def _compute_primary_losses(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        lq_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
        if getattr(self, "amp_enable", False) and getattr(self, "amp_loss_fp32", True):
            pred = pred.float()
            gt = gt.float()
            lq_ref = lq_ref.float()

        l_total = pred.new_tensor(0.0)
        loss_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        if self.cri_pix:
            if hasattr(self, "vmax") and self.vmax is not None and "PoissonNLL" in self.cri_pix.__class__.__name__:
                l_pix = self.cri_pix(pred, gt, vmax=self.vmax)
            else:
                l_pix = self.cri_pix(pred, gt)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(pred, gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict["l_percep"] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict["l_style"] = l_style
        if self.cri_tv:
            l_tv = self.cri_tv(pred)
            l_total += l_tv
            loss_dict["l_tv"] = l_tv
        if self.cri_gradient:
            if hasattr(self.cri_gradient, "use_input_as_ref") and self.cri_gradient.use_input_as_ref:
                l_gradient = self.cri_gradient(pred, input_ref=lq_ref)
            else:
                l_gradient = self.cri_gradient(pred, target=gt)
            l_total += l_gradient
            loss_dict["l_gradient"] = l_gradient
        return l_total, loss_dict

    def _compute_view_fusion_regularizers(
        self,
        pred: torch.Tensor,
        aux: dict[str, Any] | None,
        ref_input: torch.Tensor,
    ) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
        opt = self._view_fusion_reg_opt
        if not bool(opt.get("enable", False)) or not self._supports_fusion_controls():
            return pred.new_tensor(0.0), OrderedDict()
        if pred.ndim != 4 or int(pred.shape[1]) != 2 or int(ref_input.shape[0]) < 1:
            return pred.new_tensor(0.0), OrderedDict()

        loss_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        total = pred.new_tensor(0.0)
        channel_weights = self._build_channel_weights(pred)

        self_only = None
        self_weight = float(opt.get("self_consistency_weight", 0.0))
        self_prob = float(opt.get("self_consistency_prob", 1.0))
        if self_weight > 0.0 and torch.rand((), device=pred.device).item() <= self_prob:
            self_only = self._self_only_reference(ref_input)
            if self_only is not None:
                l_self = _weighted_l1(pred.float(), self_only.float(), channel_weights)
                l_self = l_self * self_weight
                total += l_self
                loss_dict["l_view_self"] = l_self

        wrong_pair_weight = float(opt.get("wrong_pair_weight", 0.0))
        wrong_pair_prob = float(opt.get("wrong_pair_prob", 0.25))
        if wrong_pair_weight > 0.0 and int(ref_input.shape[0]) > 1 and torch.rand((), device=pred.device).item() <= wrong_pair_prob:
            if self_only is None:
                self_only = self._self_only_reference(ref_input)
            if self_only is not None:
                perm = torch.roll(torch.arange(int(ref_input.shape[0]), device=pred.device), shifts=1, dims=0)

                swap_ant = ref_input.clone()
                swap_ant[:, 0] = ref_input[perm, 0]
                pred_swap_ant, _ = self._forward_train_network(swap_ant, return_aux=False, disable_fusion=False)
                l_swap_post = F.l1_loss(pred_swap_ant[:, 1:2].float(), self_only[:, 1:2].float())

                swap_post = ref_input.clone()
                swap_post[:, 1] = ref_input[perm, 1]
                pred_swap_post, _ = self._forward_train_network(swap_post, return_aux=False, disable_fusion=False)
                l_swap_ant = F.l1_loss(pred_swap_post[:, 0:1].float(), self_only[:, 0:1].float())

                l_wrong_pair = 0.5 * (l_swap_post + l_swap_ant) * wrong_pair_weight
                total += l_wrong_pair
                loss_dict["l_view_wrong_pair"] = l_wrong_pair

        gate_weight = float(opt.get("gate_sparsity_weight", 0.0))
        gate_mean = None if aux is None else aux.get("gate_mean", None)
        if gate_weight > 0.0 and gate_mean is not None:
            l_gate = gate_mean.float() * gate_weight
            total += l_gate
            loss_dict["l_view_gate"] = l_gate

        return total, loss_dict

    def _compute_support_regularizers(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        aux: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
        opt = self._support_reg_opt
        if not bool(opt.get("enable", False)):
            return pred.new_tensor(0.0), OrderedDict()

        loss_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        total = pred.new_tensor(0.0)

        if self.support_mask is not None:
            outside_weight = float(opt.get("outside_weight", 0.0))
            if outside_weight > 0.0:
                outside_mask = (1.0 - self.support_mask.float()).clamp(min=0.0)
                denom = outside_mask.sum().clamp_min(1.0)
                l_out = (torch.abs(pred.float()) * outside_mask).sum() / denom
                l_out = l_out * outside_weight
                total += l_out
                loss_dict["l_support_outside"] = l_out

        boundary_weight = float(opt.get("boundary_l1_weight", 0.0))
        if boundary_weight > 0.0 and self.support_boundary_weight is not None:
            weights = self.support_boundary_weight.float()
            denom = weights.sum().clamp_min(1.0)
            l_boundary = (torch.abs(pred.float() - gt.float()) * weights).sum() / denom
            l_boundary = l_boundary * boundary_weight
            total += l_boundary
            loss_dict["l_support_boundary"] = l_boundary

        gate_weight = float(opt.get("gate_sparsity_weight", 0.0))
        gate_mean = None if aux is None else aux.get("gate_mean", None)
        if gate_weight > 0.0 and gate_mean is not None:
            l_gate = gate_mean.float() * gate_weight
            total += l_gate
            loss_dict["l_support_gate"] = l_gate

        return total, loss_dict

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output, self.output_aux = self._forward_train_network(
            self.lq,
            return_aux=self._should_return_aux(),
            disable_fusion=False,
        )
        l_total, loss_dict = self._compute_primary_losses(self.output, self.gt, self.lq)
        l_view, view_loss_dict = self._compute_view_fusion_regularizers(self.output, self.output_aux, self.lq)
        l_support, support_loss_dict = self._compute_support_regularizers(self.output, self.gt, self.output_aux)
        l_total += l_view
        l_total += l_support
        loss_dict.update(view_loss_dict)
        loss_dict.update(support_loss_dict)

        if getattr(self, "amp_enable", False) and getattr(self, "amp_scaler", None) is not None:
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
        if accumulation_steps <= 1:
            return self.optimize_parameters(current_iter)

        if micro_step == 1:
            self.optimizer_g.zero_grad()

        self.output, self.output_aux = self._forward_train_network(
            self.lq,
            return_aux=self._should_return_aux(),
            disable_fusion=False,
        )
        l_total, loss_dict = self._compute_primary_losses(self.output, self.gt, self.lq)
        l_view, view_loss_dict = self._compute_view_fusion_regularizers(self.output, self.output_aux, self.lq)
        l_support, support_loss_dict = self._compute_support_regularizers(self.output, self.gt, self.output_aux)
        l_total += l_view
        l_total += l_support
        loss_dict.update(view_loss_dict)
        loss_dict.update(support_loss_dict)

        (l_total / float(accumulation_steps)).backward()
        if micro_step >= accumulation_steps:
            self.optimizer_g.step()
            self.log_dict = self.reduce_loss_dict(loss_dict)
            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)

    def _select_eval_network(self, net_tag: str):
        if net_tag == "ema":
            net = getattr(self, "net_g_ema", None)
            if net is None and net_tag not in self._missing_eval_net_warned:
                self._missing_eval_net_warned.add(net_tag)
                self.logger.warning("Validation requested eval_networks=[ema], but EMA weights are unavailable.")
            return net
        if net_tag == "g":
            return self.net_g
        if net_tag not in self._missing_eval_net_warned:
            self._missing_eval_net_warned.add(net_tag)
            self.logger.warning(f"Unknown validation network tag: {net_tag}.")
        return None

    def _resolve_eval_networks(self, merged_val_opt: dict) -> list[str]:
        eval_networks = merged_val_opt.get("eval_networks", None)
        if eval_networks is None:
            eval_networks = ["ema"] if hasattr(self, "net_g_ema") else ["g"]
        return [str(x).lower() for x in eval_networks]

    def _resolve_best_metric_key(self, dataset_name: str, merged_val_opt: dict, eval_networks: list[str]) -> str | None:
        record = getattr(self, "best_metric_results", {}).get(dataset_name, {})
        if not record:
            return None

        best_metric = merged_val_opt.get("best_metric", None)
        if best_metric is None:
            metrics = merged_val_opt.get("metrics", {}) or {}
            metric_names = list(metrics.keys())
            if not metric_names:
                return None
            best_metric = metric_names[0]
            if eval_networks:
                best_metric = f"{best_metric}_{eval_networks[0]}"
            return best_metric if best_metric in record else None

        best_metric = str(best_metric)
        if best_metric in record:
            return best_metric
        if len(eval_networks) == 1:
            cand = f"{best_metric}_{eval_networks[0]}"
            if cand in record:
                return cand
        return None

    def _maybe_save_best_checkpoint(self, dataset_name: str, current_iter: int, merged_val_opt: dict,
                                    eval_networks: list[str]) -> None:
        if not bool(merged_val_opt.get("save_best_ckpt", False)):
            return

        best_key = self._resolve_best_metric_key(dataset_name, merged_val_opt, eval_networks)
        if best_key is None:
            if not self._warned_no_metric_for_best:
                self._warned_no_metric_for_best = True
                self.logger.warning("save_best_ckpt=true but no usable metric is available. Skip best checkpoint.")
            return

        record = self.best_metric_results[dataset_name][best_key]
        if int(record.get("iter", -1)) != int(current_iter):
            return

        self.logger.info(f"Saving best checkpoint: {best_key} = {record['val']:.6f} @ {current_iter}")
        if hasattr(self, "net_g_ema"):
            self.save_network([self.net_g, self.net_g_ema], "net_g", "best", param_key=["params", "params_ema"])
        else:
            self.save_network(self.net_g, "net_g", "best")

    @torch.no_grad()
    def _maybe_render_eval_mp4(self, *, val_data: dict, proj_full: np.ndarray, model_input_full: np.ndarray,
                               current_iter: int, merged_val_opt: dict, max_value: float) -> bool:
        mp4_opt = merged_val_opt.get("eval_mp4", {}) or {}
        if not bool(mp4_opt.get("enable", False)):
            return False
        if (not self._eval_mp4_async_opt_warned) and any(str(k).startswith("async_") for k in mp4_opt.keys()):
            self._eval_mp4_async_opt_warned = True
            self.logger.warning("val.eval_mp4.async_* options are deprecated and ignored; MP4 generation is synchronous.")

        try:
            every = int(mp4_opt.get("every", 0))
        except Exception:
            every = 0
        if every > 0 and (int(current_iter) % every != 0):
            return False

        net_tag = str(mp4_opt.get("network", "ema" if hasattr(self, "net_g_ema") else "g")).lower().strip()
        net = self._select_eval_network(net_tag)
        if net is None:
            return False

        model_label = str(mp4_opt.get("model_label", net_tag)).strip() or net_tag
        lq_path = _extract_path_str(val_data, "lq_path")
        if lq_path is None:
            lq_path = _extract_path_str(val_data, "gt_path")

        exp_name = str(self.opt.get("name", "experiment"))
        out_root_tmpl = str(mp4_opt.get("output_dir", "experiments/{name}/visualization/eval_mp4"))
        out_root = Path(out_root_tmpl.format(name=exp_name))
        out_root.mkdir(parents=True, exist_ok=True)

        stem = Path(lq_path).stem if lq_path else "val_sample"
        label_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_label).strip("_") or "model"
        out_path = out_root / f"{stem}_iter{int(current_iter)}_{label_slug}.mp4"
        if out_path.exists() and not bool(mp4_opt.get("overwrite", False)):
            return False

        thin_factors = _parse_thin_factors(mp4_opt.get("thin_factors", [2, 3, 4, 5]))
        patient_dir = Path(lq_path).resolve().parent if lq_path else None

        try:
            def _predict(
                x: np.ndarray,
                *,
                _net=net,
                _mv=float(max_value),
                _device=str(self.device),
                _model_input_full=np.asarray(model_input_full, dtype=np.float32),
            ) -> np.ndarray:
                in_arr: np.ndarray
                if _model_input_full.ndim == 4 and _model_input_full.shape[0] >= 2:
                    if tuple(_model_input_full.shape[1:]) == tuple(x.shape):
                        in_arr = np.concatenate(
                            [x.astype(np.float32, copy=False)[None, ...], _model_input_full[1:]],
                            axis=0,
                        )
                    else:
                        in_arr = x.astype(np.float32, copy=False)
                else:
                    in_arr = x.astype(np.float32, copy=False)
                den = denoise_views(
                    net=_net,
                    proj_count=in_arr,
                    max_value=float(_mv),
                    device=str(_device),
                )
                return np.clip(np.asarray(den, dtype=np.float32), 0.0, None)

            predictors = [ModelPredictor(label=model_label, predict_fn=_predict)]
            generate_eval_mp4_from_projection(
                proj_20s=np.asarray(proj_full, dtype=np.float32),
                predictors=predictors,
                out_path=out_path,
                thin_factors=thin_factors,
                seed=int(mp4_opt.get("seed", 123)),
                patient_dir=patient_dir,
                lq_path=lq_path,
                require_cached=bool(mp4_opt.get("require_cached", True)),
                allow_on_the_fly_cache=bool(mp4_opt.get("allow_on_the_fly_cache", False)),
                bm3d_sigma=float(mp4_opt.get("bm3d_sigma", 1.0)),
                bm3d_workers=int(mp4_opt.get("bm3d_workers", 0)),
                include_bm3d=bool(mp4_opt.get("with_bm3d", False)),
                use_log1p=bool(mp4_opt.get("log1p", True)),
                show_global_text=bool(mp4_opt.get("show_global_text", True)),
                fps=float(mp4_opt.get("fps", 10.0)),
                crf=int(mp4_opt.get("crf", 18)),
                turns=int(mp4_opt.get("turns", 1)),
                diff_percentile=float(mp4_opt.get("diff_percentile", 99.5)),
                vmax_percentile=float(mp4_opt.get("vmax_percentile", 100.0)),
                show_factor_col=bool(mp4_opt.get("show_factor_col", True)),
                factor_col_width=int(mp4_opt.get("factor_col_width", 64)),
            )
            self.logger.info(f"[val][mp4] saved: {out_path}")
            return True
        except Exception as e:
            if not self._eval_mp4_warned:
                self._eval_mp4_warned = True
                self.logger.warning(f"[val][mp4] generation failed: {e}", exc_info=True)
            return False

    @torch.no_grad()
    def _nondist_validation_projection_light(self, dataloader, current_iter, tb_logger, merged_val_opt: dict) -> None:
        dataset_name = dataloader.dataset.opt["name"]
        eval_networks = self._resolve_eval_networks(merged_val_opt)
        metrics = merged_val_opt.get("metrics", {}) or {}
        with_metrics = len(metrics) > 0

        if with_metrics:
            metric_names = list(metrics.keys())
            metric_keys = [f"{m}_{tag}" for tag in eval_networks for m in metric_names]
            if not hasattr(self, "metric_results"):
                self.metric_results = {metric: 0.0 for metric in metric_keys}
            for key in metric_keys:
                self.metric_results[key] = 0.0

            self._initialize_best_metric_results(dataset_name)
            record = self.best_metric_results[dataset_name]
            for tag in eval_networks:
                for m, content in metrics.items():
                    key = f"{m}_{tag}"
                    if key in record:
                        continue
                    better = (content or {}).get("better", "higher")
                    init_val = float("-inf") if better == "higher" else float("inf")
                    record[key] = dict(better=better, val=init_val, iter=-1)
        elif bool(merged_val_opt.get("save_best_ckpt", False)) and not self._warned_no_metric_for_best:
            self._warned_no_metric_for_best = True
            self.logger.warning("save_best_ckpt=true but val.metrics is empty. Skip best checkpoint selection.")

        ds_train_opt = (self.opt.get("datasets", {}) or {}).get("train", {}) or {}
        max_value = float(merged_val_opt.get("max_value", ds_train_opt.get("max_value", 150.0)))
        if max_value <= 0:
            max_value = 1.0

        try:
            view_stride = int(merged_val_opt.get("view_stride", 1))
        except Exception:
            view_stride = 1
        if view_stride < 1:
            view_stride = 1

        pll_alias = {"pll", "pllloss", "poisson_nll", "poisson_nll_mean", "poisson_nll_loss"}
        used_samples = 0
        mp4_opt = merged_val_opt.get("eval_mp4", {}) or {}
        mp4_enable = bool(mp4_opt.get("enable", False))
        try:
            mp4_max_samples = max(0, int(mp4_opt.get("max_samples", 1)))
        except Exception:
            mp4_max_samples = 1
        mp4_generated = 0

        for val_data in dataloader:
            proj = _extract_projection(val_data)
            if proj is None:
                continue
            if proj.ndim != 3 or int(proj.shape[0]) < 1:
                continue

            proj_full = np.asarray(proj, dtype=np.float32)
            model_input_full = _extract_model_input(val_data)
            if model_input_full is None:
                model_input_full = proj_full

            if view_stride > 1:
                proj = proj_full[::view_stride]
                if model_input_full.ndim == 4:
                    model_input = np.asarray(model_input_full[:, ::view_stride], dtype=np.float32)
                else:
                    model_input = np.asarray(model_input_full[::view_stride], dtype=np.float32)
            else:
                proj = proj_full
                model_input = np.asarray(model_input_full, dtype=np.float32)

            proj = np.asarray(proj, dtype=np.float32)
            obs_counts = np.clip(np.rint(proj), 0.0, None).astype(np.float32, copy=False)

            if mp4_enable and mp4_generated < mp4_max_samples:
                if self._maybe_render_eval_mp4(
                    val_data=val_data,
                    proj_full=proj_full,
                    model_input_full=model_input_full,
                    current_iter=int(current_iter),
                    merged_val_opt=merged_val_opt,
                    max_value=float(max_value),
                ):
                    mp4_generated += 1

            for tag in eval_networks:
                net = self._select_eval_network(tag)
                if net is None:
                    continue

                den = denoise_views(
                    net=net,
                    proj_count=model_input,
                    max_value=max_value,
                    device=str(self.device),
                )
                den = np.clip(np.asarray(den, dtype=np.float32), 0.0, None)

                if not with_metrics:
                    continue

                for metric_name, metric_opt in metrics.items():
                    opt_type = str((metric_opt or {}).get("type", "")).strip().lower()
                    if opt_type not in pll_alias:
                        if metric_name not in self._unsupported_metric_warned:
                            self._unsupported_metric_warned.add(metric_name)
                            self.logger.warning(
                                f"Metric '{metric_name}' (type={opt_type}) is unsupported in "
                                "SPECTProjectionEvalDataset light validation; skipped."
                            )
                        continue

                    eps = float((metric_opt or {}).get("eps", 1.0e-8))
                    full = bool((metric_opt or {}).get("full", True))

                    y = torch.from_numpy(obs_counts).to(self.device, dtype=torch.float32)
                    lam = torch.from_numpy(den).to(self.device, dtype=torch.float32)
                    lam = torch.clamp(lam, min=max(eps, 1.0e-12))
                    pll_val = F.poisson_nll_loss(torch.log(lam), y, log_input=True, full=full, reduction="mean")
                    self.metric_results[f"{metric_name}_{tag}"] += float(pll_val.detach().cpu().item())

            used_samples += 1

        if not with_metrics:
            self.logger.info(f"[val][light] done: dataset={dataset_name}, samples={used_samples}, view_stride={view_stride}")
            return

        if used_samples <= 0:
            self.logger.warning(f"[val][light] no valid samples for dataset={dataset_name}; metrics are not updated.")
            return

        for metric in list(self.metric_results.keys()):
            if metric.split("_")[-1] in eval_networks:
                self.metric_results[metric] /= float(used_samples)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

        if mp4_enable:
            self.logger.info(f"[val][mp4] generated={mp4_generated}")

        self._maybe_save_best_checkpoint(dataset_name, current_iter, merged_val_opt, eval_networks)
        self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Use SRModel validation by default; switch to light path for projection datasets."""
        current_iter_i = _to_int_iter(current_iter)
        val_opt = self.opt.get("val", {}) or {}
        datasets_val_opt = self.opt.get("datasets", {}).get("val", {}) or {}
        merged_val_opt = {**val_opt, **datasets_val_opt}

        ds_opt = getattr(dataloader.dataset, "opt", {}) or {}
        ds_type = str(ds_opt.get("type", "")).strip().lower()
        if not ds_type.startswith("spectprojectioneval"):
            return super().nondist_validation(dataloader, current_iter_i, tb_logger, save_img)

        return self._nondist_validation_projection_light(
            dataloader=dataloader,
            current_iter=current_iter_i,
            tb_logger=tb_logger,
            merged_val_opt=merged_val_opt,
        )
