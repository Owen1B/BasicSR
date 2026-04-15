from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from prime.models._legacy.spect3d_model import SPECT3DModel
from prime.losses.poisson_loss import anscombe_forward_torch
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class SPECT3DConsistencyModel(SPECT3DModel):
    """SPECT3DModel + N2N loss + Poisson-consistency loss.

    Consistency idea (teacher-student):
      1) teacher (EMA) denoises the current input:  y = EMA(x)
      2) sample Poisson noise from y in *count domain*:  y' ~ Poisson(y_count)
      3) student (G) denoises y':  G(y')
      4) enforce G(y') ≈ G(x) (or optionally ≈ EMA(x))

    This is a "cycle/consistency under Poisson corruption" regularization on top of N2N.
    """

    def __init__(self, opt):
        super().__init__(opt)
        train_opt = self.opt.get('train', {}) or {}
        self.cons_opt: Dict[str, Any] = train_opt.get('consistency_opt', {}) or {}
        self.lowdose_opt: Dict[str, Any] = train_opt.get('lowdose_opt', {}) or {}

        # Defaults
        self.cons_weight: float = float(self.cons_opt.get('loss_weight', 0.0))
        self.cons_type: str = str(self.cons_opt.get('loss_type', 'l1')).lower()  # l1 | l2 | poisson
        self.cons_start_iter: int = int(self.cons_opt.get('start_iter', 0))
        self.cons_target: str = str(self.cons_opt.get('target', 'g_on_input')).lower()  # g_on_input | ema_on_input
        self.cons_use_ema_teacher: bool = bool(self.cons_opt.get('use_ema_teacher', True))
        self.cons_detach_target: bool = bool(self.cons_opt.get('detach_target', True))
        # Domain for consistency loss computation: 'norm' (0..1) or 'count' (0..max_value)
        self.cons_domain: str = str(self.cons_opt.get('domain', 'norm')).lower()  # norm | count | anscombe

        # Normalization assumptions (this project uses linear for Poisson NLL)
        self.cons_norm_type: str = str(self.cons_opt.get('norm_type', 'linear')).lower()

        # max_value used to convert norm<->count for Poisson sampling
        # Prefer dataset max_value; allow override via consistency_opt
        ds_train = (self.opt.get('datasets', {}) or {}).get('train', {}) or {}
        self.cons_max_value: float = float(self.cons_opt.get('max_value', ds_train.get('max_value', 1.0)))

        # ===== Low-dose Poisson consistency (teacher scaled down by a random factor) =====
        self.lowdose_weight: float = float(self.lowdose_opt.get('loss_weight', 0.0))
        self.lowdose_start_iter: int = int(self.lowdose_opt.get('start_iter', 0))
        self.lowdose_use_ema_teacher: bool = bool(self.lowdose_opt.get('use_ema_teacher', True))
        self.lowdose_detach_target: bool = bool(self.lowdose_opt.get('detach_target', True))
        self.lowdose_domain: str = str(self.lowdose_opt.get('domain', 'count')).lower()  # norm | count | anscombe
        self.lowdose_type: str = str(self.lowdose_opt.get('loss_type', 'l1')).lower()  # l1 | l2 | poisson
        self.lowdose_factor_min: float = float(self.lowdose_opt.get('factor_min', 1.0))
        self.lowdose_factor_max: float = float(self.lowdose_opt.get('factor_max', 1.0))
        self.lowdose_per_sample: bool = bool(self.lowdose_opt.get('per_sample', False))
        self.lowdose_max_value: float = float(self.lowdose_opt.get('max_value', ds_train.get('max_value', self.cons_max_value)))
        self.lowdose_norm_type: str = str(self.lowdose_opt.get('norm_type', 'linear')).lower()

        if self.cons_weight < 0:
            raise ValueError(f'consistency_opt.loss_weight must be >= 0. Got: {self.cons_weight}')
        if self.cons_type not in ['l1', 'l2', 'mse', 'poisson', 'pll', 'poisson_nll', 'poisson_kl']:
            raise ValueError(
                f'consistency_opt.loss_type must be l1|l2|mse|poisson. Got: {self.cons_type}'
            )
        if self.cons_norm_type != 'linear':
            # We can extend this later, but Poisson sampling must happen in count domain.
            raise ValueError(
                f'consistency_opt currently supports norm_type=linear only. Got: {self.cons_norm_type}'
            )
        if self.cons_target not in ['g_on_input', 'ema_on_input']:
            raise ValueError(f'consistency_opt.target must be g_on_input|ema_on_input. Got: {self.cons_target}')
        if self.cons_domain not in ['norm', 'count', 'anscombe']:
            raise ValueError(f'consistency_opt.domain must be norm|count|anscombe. Got: {self.cons_domain}')

        if self.lowdose_weight < 0:
            raise ValueError(f'lowdose_opt.loss_weight must be >= 0. Got: {self.lowdose_weight}')
        if self.lowdose_domain not in ['norm', 'count', 'anscombe']:
            raise ValueError(f'lowdose_opt.domain must be norm|count|anscombe. Got: {self.lowdose_domain}')
        if self.lowdose_type not in ['l1', 'l2', 'mse', 'poisson', 'pll', 'poisson_nll', 'poisson_kl']:
            raise ValueError(f'lowdose_opt.loss_type must be l1|l2|mse|poisson. Got: {self.lowdose_type}')
        if self.lowdose_factor_min < 1e-12 or self.lowdose_factor_max < 1e-12:
            raise ValueError(
                f'lowdose_opt.factor_min/factor_max must be > 0. Got: {self.lowdose_factor_min}, {self.lowdose_factor_max}'
            )
        if self.lowdose_factor_max < self.lowdose_factor_min:
            raise ValueError(
                f'lowdose_opt.factor_max must be >= factor_min. Got: {self.lowdose_factor_max} < {self.lowdose_factor_min}'
            )
        if self.lowdose_norm_type != 'linear':
            raise ValueError(f'lowdose_opt currently supports norm_type=linear only. Got: {self.lowdose_norm_type}')

    def _poisson_rate_loss_count(
        self, pred_count: torch.Tensor, target_count: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Poisson 'PLL' between rates (means) in count domain.

        This is equivalent to Poisson cross-entropy / generalized I-divergence between
        target rate (teacher) and predicted rate (student), up to constants:
            L = pred - target * log(pred)
        (target-dependent constants are omitted; gradients match Poisson NLL w.r.t pred.)
        """
        eps = float(eps)
        pred_safe = torch.clamp(pred_count, min=eps)
        target_safe = torch.clamp(target_count, min=0.0)
        loss = pred_safe - target_safe * torch.log(pred_safe)
        return loss.mean()

    def _rate_loss(self, pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
        """Loss between two *rates* (not observations). pred/target are in the same domain."""
        lt = str(loss_type).lower()
        if lt in ['l1']:
            return F.l1_loss(pred, target, reduction='mean')
        if lt in ['l2', 'mse']:
            return F.mse_loss(pred, target, reduction='mean')
        if lt in ['poisson', 'pll', 'poisson_nll', 'poisson_kl']:
            # For poisson loss, pred/target must already be in count domain.
            return self._poisson_rate_loss_count(pred, target, eps=1e-8)
        raise ValueError(f'Unsupported loss_type: {loss_type}')

    def _to_anscombe_from_count(self, x_count: torch.Tensor) -> torch.Tensor:
        """Convert count-domain tensor to Anscombe domain (differentiable)."""
        return anscombe_forward_torch(x_count)

    @torch.no_grad()
    def _get_teacher_output(self, x: torch.Tensor) -> torch.Tensor:
        """Return teacher output in normalized domain (same domain as network outputs)."""
        if self.cons_use_ema_teacher and hasattr(self, 'net_g_ema'):
            return self.net_g_ema(x)
        return self.net_g(x)

    @torch.no_grad()
    def _get_lowdose_teacher_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.lowdose_use_ema_teacher and hasattr(self, 'net_g_ema'):
            return self.net_g_ema(x)
        return self.net_g(x)

    def _sample_lowdose_factor(self, ref: torch.Tensor) -> torch.Tensor:
        """Sample a scaling factor s in [min,max]. Returns tensor on same device.

        - per_sample=False: returns scalar tensor ()
        - per_sample=True: returns shape (B, 1, 1, ..., 1) broadcastable to ref
        """
        if self.lowdose_factor_max == self.lowdose_factor_min:
            s = ref.new_tensor(self.lowdose_factor_min)
        else:
            u = torch.rand((), device=ref.device, dtype=ref.dtype)
            s = self.lowdose_factor_min + (self.lowdose_factor_max - self.lowdose_factor_min) * u
        if not self.lowdose_per_sample:
            return s
        b = int(ref.shape[0])
        u = torch.rand((b,), device=ref.device, dtype=ref.dtype)
        s = self.lowdose_factor_min + (self.lowdose_factor_max - self.lowdose_factor_min) * u
        shape = (b,) + (1,) * (ref.dim() - 1)
        return s.view(shape)

    def optimize_parameters(self, current_iter):
        """Same as SRModel.optimize_parameters + optional consistency loss."""
        self.optimizer_g.zero_grad()

        # Main forward on input
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # Pixel (N2N) loss (e.g., PoissonNLLLoss) — keep identical behavior to SRModel
        if self.cri_pix:
            if hasattr(self, 'vmax') and self.vmax is not None and 'PoissonNLL' in self.cri_pix.__class__.__name__:
                l_pix = self.cri_pix(self.output, self.gt, vmax=self.vmax)
            else:
                l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # Perceptual loss (rare for this project)
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        # TV
        if self.cri_tv:
            l_tv = self.cri_tv(self.output)
            l_total += l_tv
            loss_dict['l_tv'] = l_tv

        # Gradient loss (edge preservation)
        if self.cri_gradient:
            if hasattr(self.cri_gradient, 'use_input_as_ref') and self.cri_gradient.use_input_as_ref:
                l_gradient = self.cri_gradient(self.output, input_ref=self.lq)
            else:
                l_gradient = self.cri_gradient(self.output, target=self.gt)
            l_total += l_gradient
            loss_dict['l_gradient'] = l_gradient

        # Pre-populate consistency keys for logging so users can always see them.
        # (Otherwise, before start_iter they won't appear in logs/TensorBoard.)
        if self.cons_weight > 0:
            zero = self.output.detach().new_tensor(0.0)
            loss_dict['l_consistency'] = zero
            loss_dict['l_consistency_raw'] = zero

        if self.lowdose_weight > 0:
            zero = self.output.detach().new_tensor(0.0)
            loss_dict['l_lowdose'] = zero
            loss_dict['l_lowdose_raw'] = zero
            loss_dict['lowdose_factor'] = zero

        # ===== Consistency loss =====
        if self.cons_weight > 0 and current_iter >= self.cons_start_iter:
            # teacher output on original input (normalized domain)
            with torch.no_grad():
                y_teacher = self._get_teacher_output(self.lq)

                # Convert to count domain and Poisson sample
                y_count = torch.clamp(y_teacher * self.cons_max_value, min=0.0)
                y_poisson = torch.poisson(y_count)
                y_poisson_norm = y_poisson / self.cons_max_value

                # Target: either G(x) or EMA(x)
                if self.cons_target == 'ema_on_input':
                    target = y_teacher
                else:
                    target = self.output
                if self.cons_detach_target:
                    target = target.detach()

            # Student forward on Poisson-sampled teacher output
            y_student = self.net_g(y_poisson_norm)

            if self.cons_domain == 'count':
                y_student_cmp = torch.clamp(y_student, min=0.0) * self.cons_max_value
                target_cmp = torch.clamp(target, min=0.0) * self.cons_max_value
            elif self.cons_domain == 'anscombe':
                y_student_count = torch.clamp(y_student, min=0.0) * self.cons_max_value
                target_count = torch.clamp(target, min=0.0) * self.cons_max_value
                y_student_cmp = self._to_anscombe_from_count(y_student_count)
                target_cmp = self._to_anscombe_from_count(target_count)
            else:
                y_student_cmp = y_student
                target_cmp = target

            if self.cons_type in ['poisson', 'pll', 'poisson_nll', 'poisson_kl'] and self.cons_domain != 'count':
                raise ValueError('consistency_opt.loss_type=poisson requires domain=count')
            l_cons_raw = self._rate_loss(y_student_cmp, target_cmp, self.cons_type)
            l_cons = l_cons_raw * self.cons_weight
            l_total += l_cons
            loss_dict['l_consistency'] = l_cons
            loss_dict['l_consistency_raw'] = l_cons_raw

        # ===== Low-dose loss (random scale -> Poisson -> denoise) =====
        if self.lowdose_weight > 0 and current_iter >= self.lowdose_start_iter:
            with torch.no_grad():
                y_teacher = self._get_lowdose_teacher_output(self.lq)  # normalized
                # teacher in count domain
                y_count = torch.clamp(y_teacher * self.lowdose_max_value, min=0.0)

                s = self._sample_lowdose_factor(y_count)
                a_count = y_count / s  # low-dose expectation in count domain (float)
                b_count = torch.poisson(a_count)
                b_norm = b_count / self.lowdose_max_value

                # Target is A (scaled teacher), in the chosen domain
                if self.lowdose_domain == 'count':
                    target_cmp = a_count
                elif self.lowdose_domain == 'anscombe':
                    target_cmp = self._to_anscombe_from_count(a_count)
                else:
                    target_cmp = torch.clamp(y_teacher, min=0.0) / s
                if self.lowdose_detach_target:
                    target_cmp = target_cmp.detach()

            y_student = self.net_g(b_norm)
            if self.lowdose_domain == 'count':
                y_student_cmp = torch.clamp(y_student, min=0.0) * self.lowdose_max_value
            elif self.lowdose_domain == 'anscombe':
                y_student_count = torch.clamp(y_student, min=0.0) * self.lowdose_max_value
                y_student_cmp = self._to_anscombe_from_count(y_student_count)
            else:
                y_student_cmp = y_student

            if self.lowdose_type in ['poisson', 'pll', 'poisson_nll', 'poisson_kl'] and self.lowdose_domain != 'count':
                raise ValueError('lowdose_opt.loss_type=poisson requires domain=count')
            l_low_raw = self._rate_loss(y_student_cmp, target_cmp, self.lowdose_type)
            l_low = l_low_raw * self.lowdose_weight
            l_total += l_low
            loss_dict['l_lowdose'] = l_low
            loss_dict['l_lowdose_raw'] = l_low_raw
            # log scalar factor (mean over batch if per-sample)
            loss_dict['lowdose_factor'] = s.detach().mean()

        # Backward + step
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
