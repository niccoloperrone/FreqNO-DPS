# Copyright 2024 The swirl_dynamics Authors.
# Modifications made by the CAM Lab at ETH Zurich.
# Modifications made by Niccolò Perrone, Politecnico di Milano /
# CentraleSupélec LMPS, 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or isourmplied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diffusion samplers."""

from typing import Any, Protocol, Sequence, Mapping, Optional, Callable

import torch
from torch.autograd import grad
import numpy as np

from diffusion import diffusion, guidance
from solvers import sde, ode


Tensor = torch.Tensor
TensorMapping = Mapping[str, Tensor]
Params = Mapping[str, Any]


class DenoiseFn(Protocol):

    def __call__(
        self, x: Tensor, sigma: Tensor, cond: TensorMapping | None
    ) -> Tensor: ...


ScoreFn = DenoiseFn


def dlog_dt(f: diffusion.ScheduleFn) -> diffusion.ScheduleFn:
    """Returns d/dt log(f(t)) = ḟ(t)/f(t) given f(t)."""
    return lambda t: grad(torch.log(f(t)), t, create_graph=True)[0]


def dsquare_dt(f: diffusion.ScheduleFn) -> diffusion.ScheduleFn:
    """Returns d/dt (f(t))^2 = 2ḟ(t)f(t) given f(t)."""
    return lambda t: grad(torch.square(f(t)), t, create_graph=True)[0]


def denoiser2score(denoise_fn: DenoiseFn, scheme: diffusion.Diffusion) -> ScoreFn:
    """Converts a denoiser to the corresponding score function."""

    def _score(x: Tensor, sigma: Tensor, cond: TensorMapping | None = None) -> Tensor:
        # Reference: eq. 74 in Karras et al. (https://arxiv.org/abs/2206.00364).
        scale = scheme.scale(scheme.sigma.inverse(sigma))
        x_hat = x / scale
        target = denoise_fn(x_hat, sigma, cond)
        return (target - x_hat) / (scale * sigma**2)

    return _score


def denoise_fn_output(
    denoise_fn: DenoiseFn,
    x: Tensor,
    sigma: Tensor,
    cond: TensorMapping | None = None,
    y: Tensor = None,
    lead_time: Tensor = None,
) -> Tensor:
    """Depending on the task 'y' and the 'lead_time' compute the result of the
    denoise_fn. Note that whenever lead_time is a Tensor there can not be None 
    values. Thus it's enough to check whether lead_time is an instance of a Tensor
    """
  
    if y is None and not isinstance(lead_time, Tensor):
        return denoise_fn(x, sigma, cond)

    elif y is not None and not isinstance(lead_time, Tensor):
        return denoise_fn(x, sigma, cond)

    elif y is not None and isinstance(lead_time, Tensor):
        return denoise_fn(x, sigma, y, lead_time, cond)


# ********************
# Samplers
# ********************


class Sampler:
    """Base class for denoising-based diffusion samplers.

    Attributes:
      input_shape: The tensor shape of a sample (excluding any batch dimensions).
      scheme: The diffusion scheme which contains the scale and noise schedules.
      denoise_fn: A function to remove noise from input data. Must handle batched
        inputs, noise levels and conditions.
      tspan: Full diffusion time steps for iterative denoising, decreasing from 1
        to (approximately) 0.
      guidance_transforms: An optional sequence of guidance transforms that
        modifies the denoising function in a post-process fashion.
      apply_denoise_at_end: If `True`, applies the denoise function another time
        to the terminal states, which are typically at a small but non-zero noise
        level.
      return_full_paths: If `True`, the output of `.generate()` and `.denoise()`
        will contain the complete sampling paths. Otherwise only the terminal
        states are returned.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        scheme: diffusion.Diffusion,
        denoise_fn: DenoiseFn,
        tspan: Tensor,
        guidance_transforms: Sequence[guidance.Transform] = (),
        apply_denoise_at_end: bool = True,
        return_full_paths: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.input_shape = input_shape
        self.scheme = scheme
        self.denoise_fn = denoise_fn
        self.tspan = tspan
        self.guidance_transforms = guidance_transforms
        self.apply_denoise_at_end = apply_denoise_at_end
        self.return_full_paths = return_full_paths
        self.device = device
        self.dtype = dtype

    def generate(
        self,
        num_samples: int,
        y: Tensor = None,
        lead_time: Tensor = None,
        cond: TensorMapping | None = None,
        guidance_inputs: TensorMapping | None = None,
    ) -> Tensor:
        """Generates a batch of diffusion samples from scratch.

        Args:
          num_samples: The number of samples to generate in a single batch.
          cond: Explicit conditioning inputs for the denoising function. These
            should be provided **without** batch dimensions (one should be added
            inside this function based on `num_samples`).
          y: is the output and result of the solver
          lead_time: keeps track not of the diffusion time but of the timestep of the solver
            this is relevant for an all to all training strategy
          guidance_inputs: Inputs used to construct the guided denoising function.
            These also should in principle not include a batch dimension.

        Returns:
          The generated samples.
        """
        if self.tspan is None or self.tspan.ndim != 1:
            raise ValueError("`tspan` must be a 1-d Tensor.")

        x_shape = (num_samples,) + self.input_shape
        x1 = torch.randn(x_shape, dtype=self.dtype, device=self.device)
        x1 = x1 * self.scheme.sigma(self.tspan[0]) * self.scheme.scale(self.tspan[0])

        if cond is not None:
            new = {}
            for k, v in cond.items():
                if v.shape[0] == 1:
                    # single example → tile to batch
                    new[k] = v.repeat(num_samples, *([1] * (v.dim() - 1)))
                elif v.shape[0] == num_samples:
                    # already batched correctly → leave as is
                    new[k] = v
                else:
                    raise ValueError(
                        f"cond[{k}] has batch {v.shape[0]} but num_samples={num_samples}. "
                        "Provide (1, ...) or (num_samples, ...)."
                    )
            cond = new

        denoised = self.denoise(
            noisy=x1,
            tspan=self.tspan,
            y=y,
            lead_time=lead_time,
            cond=cond,
            guidance_inputs=guidance_inputs,
        )

        samples = denoised[-1] if self.return_full_paths else denoised
        if self.apply_denoise_at_end:
            denoise_fn = self.get_guided_denoise_fn(guidance_inputs=guidance_inputs)
            samples = denoise_fn_output(
                denoise_fn=denoise_fn,
                x=samples / self.scheme.scale(self.tspan[-1]),
                sigma=self.scheme.sigma(self.tspan[-1]),
                cond=cond,
                y=y,
                lead_time=lead_time,
            )

            if self.return_full_paths:
                denoised = torch.cat([denoised, samples.unsqueeze(0)], axis=0)

        return denoised if self.return_full_paths else samples

    def denoise(
        self,
        noisy: Tensor,
        tspan: Tensor,
        y: Tensor = None,
        lead_time: Tensor = None,
        cond: TensorMapping | None = None,
        guidance_inputs: TensorMapping | None = None,
    ) -> Tensor:
        """Applies iterative denoising to given noisy states.

        Args:
          noisy: A batch of noisy states (all at the same noise level). Can be fully
            noisy or partially denoised.
          tspan: A decreasing sequence of diffusion time steps within the interval
            [1, 0). The first element aligns with the time step of the `noisy`
            input.
          cond: (Optional) Conditioning inputs for the denoise function. The batch
            dimension should match that of `noisy`.
          guidance_inputs: Inputs for constructing the guided denoising function.

        Returns:
          The denoised output.
        """
        raise NotImplementedError

    def get_guided_denoise_fn(self, guidance_inputs: Mapping[str, Tensor]) -> DenoiseFn:
        """Returns a guided denoise function."""
        denoise_fn = self.denoise_fn
        for transform in self.guidance_transforms:
            denoise_fn = transform(denoise_fn, guidance_inputs)
        return denoise_fn


class SdeSampler(Sampler):
    """Draws samples by solving an SDE.

    Attributes:
      integrator: The SDE solver for solving the sampling SDE.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        scheme: diffusion.Diffusion,
        denoise_fn: DenoiseFn,
        tspan: Tensor,
        integrator: sde.SdeSolver = None,
        guidance_transforms: Sequence[guidance.Transform] = (),
        apply_denoise_at_end: bool = True,
        return_full_paths: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            input_shape=input_shape,
            scheme=scheme,
            denoise_fn=denoise_fn,
            tspan=tspan,
            guidance_transforms=guidance_transforms,
            apply_denoise_at_end=apply_denoise_at_end,
            return_full_paths=return_full_paths,
            device=device,
            dtype=dtype,
        )
        self.integrator = integrator

    def denoise(
        self,
        noisy: Tensor,
        tspan: Tensor,
        y: Tensor = None,
        lead_time: Tensor = None,
        cond: TensorMapping | None = None,
        guidance_inputs: TensorMapping | None = None,
    ) -> Tensor:
        """Applies iterative denoising to given noisy states."""
        if self.integrator is None:
            self.integrator = sde.EulerMaruyama(terminal_only=True) 

        if self.integrator.terminal_only and self.return_full_paths:
            raise ValueError(
                f"Integrator type `{type(self.integrator)}` does not support"
                " returning full paths."
            )

        params = dict(
            drift=dict(guidance_inputs=guidance_inputs, cond=cond), diffusion={}
        )
        denoised = self.integrator(
            dynamics=self.dynamics,
            x0=noisy,
            tspan=tspan,
            params=params,
            y=y,
            lead_time=lead_time,
        )
        # SDE solvers may return either the full paths or the terminal state only.
        # If the former, the lead axis should be time.
        samples = denoised if self.integrator.terminal_only else denoised[-1]
        return denoised if self.return_full_paths else samples

    @property
    def dynamics(self) -> sde.SdeDynamics:
        """Drift and diffusion terms of the sampling SDE.

        In score function:

          dx = [ṡ(t)/s(t) x - 2 s(t)²σ̇(t)σ(t) ∇pₜ(x)] dt + s(t) √[2σ̇(t)σ(t)] dωₜ,

        obtained by substituting eq. 28, 34 of Karras et al.
        (https://arxiv.org/abs/2206.00364) into the reverse SDE formula - eq. 6 in
        Song et al. (https://arxiv.org/abs/2011.13456). Alternatively, it may be
        rewritten in terms of the denoise function (plugging in eq. 74 of
        Karras et al.) as:

          dx = [2 σ̇(t)/σ(t) + ṡ(t)/s(t)] x - [2 s(t)σ̇(t)/σ(t)] D(x/s(t), σ(t)) dt
            + s(t) √[2σ̇(t)σ(t)] dωₜ

        where s(t), σ(t) are the scale and noise schedule of the diffusion scheme
        respectively.
        """

        def _drift(
            x: Tensor,
            t: Tensor,
            params: Params,
            y: Tensor = None,
            lead_time: Tensor = None,
        ) -> Tensor:
            assert t.ndim == 0, "`t` must be a scalar."
            denoise_fn = self.get_guided_denoise_fn(
                guidance_inputs=params["guidance_inputs"]
            )
            s, sigma = self.scheme.scale(t), self.scheme.sigma(t)
            x_hat = x / s
            if not t.requires_grad:
                t.requires_grad_(True)
            dlog_sigma_dt = dlog_dt(self.scheme.sigma)(t)
            dlog_s_dt = dlog_dt(self.scheme.scale)(t)
            drift = (2 * dlog_sigma_dt + dlog_s_dt) * x
            denoiser_output = denoise_fn_output(
                denoise_fn=denoise_fn,
                x=x_hat,
                sigma=sigma,
                cond=params["cond"],
                y=y,
                lead_time=lead_time,
            )
            drift = drift - 2 * dlog_sigma_dt * s * denoiser_output
            return drift

        def _diffusion(x: Tensor, t: Tensor, params: Params) -> Tensor:
            del x, params
            assert t.ndim == 0, "`t` must be a scalar."
            if not t.requires_grad:
                t.requires_grad_(True)
            dsquare_sigma_dt = dsquare_dt(self.scheme.sigma)(t)
            return torch.sqrt(dsquare_sigma_dt) * self.scheme.scale(t) 

        return sde.SdeDynamics(_drift, _diffusion)


class DPSSampler(SdeSampler):
    """
    Posterior SDE sampler — SDE analogue of DPSOdeSampler.

    The reverse-time SDE for p(u_τ | y, u_NO) is:

        du = [(2σ̇/σ + ṡ/s) u  −  (2σ̇ s/σ) D(u/s, σ)
              −  2 s² σ σ̇  ·  ∇ log p(y, u_NO | u_τ)] dt
             + s √(2σσ̇) dω

    The posterior score decomposes as:

        ∇ log p_τ(u_τ | y, u_NO) = ∇ log p_τ(u_τ)        [prior, via denoiser]
                                  + ∇ log p(y | u_τ)       [sensor, DPS approx]
                                  + ∇ log p(u_NO | u_τ)    [NO, spectral]

    The likelihood scores enter the SDE drift with coefficient
        g²(τ) = 2 s² σ² (σ̇/σ) = 2 · c_ODE(τ),
    i.e., twice the probability-flow ODE coefficient. This factor of 2
    is the standard SDE / probability-flow-ODE relationship.

    Hyperparameters (matching DPSOdeSampler):
        lambda_sensor : DPS step-size scale       (≈ 1)
        lambda_NO     : spectral NO scale         (≈ 1)
    """

    def __init__(
        self,
        input_shape:      tuple[int, ...],
        scheme:           diffusion.Diffusion,
        denoise_fn:       DenoiseFn,
        tspan:            Tensor,
        likelihood,                                # CombinedLikelihood
        lambda_sensor:    float = 1.0,
        lambda_NO:        float = 1.0,
        integrator:       sde.SdeSolver = None,
        guidance_transforms: Sequence[guidance.Transform] = (),
        apply_denoise_at_end: bool = True,
        return_full_paths:    bool = False,
        device:           torch.device = None,
        dtype:            torch.dtype = torch.float32,
    ):
        super().__init__(
            input_shape=input_shape, scheme=scheme, denoise_fn=denoise_fn,
            tspan=tspan, integrator=integrator,
            guidance_transforms=guidance_transforms,
            apply_denoise_at_end=apply_denoise_at_end,
            return_full_paths=return_full_paths,
            device=device, dtype=dtype,
        )
        self.likelihood    = likelihood
        self.lambda_sensor = float(lambda_sensor)
        self.lambda_NO     = float(lambda_NO)

    @property
    def dynamics(self) -> sde.SdeDynamics:

        def _drift(
            x: Tensor, t: Tensor, params: Params,
            y: Tensor = None, lead_time: Tensor = None,
        ) -> Tensor:
            assert t.ndim == 0

            denoise_fn = self.get_guided_denoise_fn(params.get("guidance_inputs"))
            s     = self.scheme.scale(t)
            sigma = self.scheme.sigma(t)

            if not t.requires_grad:
                t.requires_grad_(True)
            dlog_sigma_dt = dlog_dt(self.scheme.sigma)(t)
            dlog_s_dt     = dlog_dt(self.scheme.scale)(t)

            # ── (i) Prior drift (SDE form: 2× ODE score coefficient) ────
            x_in   = x.detach().requires_grad_(True)
            x0_hat = denoise_fn_output(
                denoise_fn=denoise_fn, x=x_in / s, sigma=sigma,
                cond=params["cond"], y=None, lead_time=None,
            )
            drift_prior = (
                (2 * dlog_sigma_dt + dlog_s_dt) * x
                - 2 * dlog_sigma_dt * s * x0_hat
            )

            # ── (ii)+(iii) Guidance ──────────────────────────────
            if y is not None:
                x0 = s * x0_hat

                g_sensor, g_NO, res_norm = self.likelihood.grad_log_likelihood(
                    x0_hat    = x0,
                    x_hat_tau = (x_in / s).detach(),
                    y         = y,
                    sigma_tau = sigma.detach(),
                )

                # ── Sensor: DPS step-size rule ───────────────────
                #
                #   drift_sensor = λ_s / ‖M(x̂₀)−y‖ · (∂x̂₀/∂u_τ)ᵀ ∇ log p(y|x̂₀)
                #
                g_vjp, = torch.autograd.grad(
                    outputs      = x0,
                    inputs       = x_in,
                    grad_outputs = g_sensor,
                    retain_graph = False,
                    create_graph = False,
                )
                g_vjp = g_vjp.detach()
                drift_sensor = self.lambda_sensor * g_vjp / (res_norm + 1e-8)

                # ── NO: principled SDE coefficient ──────────────
                #
                #   drift_NO = g²(τ) · λ_NO · ∇_{u_τ} log p(u_NO | u_τ)
                #            = 2 c_ODE(τ) · λ_NO · score_NO
                #
                score_NO  = g_NO.detach() / s.detach()
                g_squared = 2.0 * (s ** 2) * (sigma ** 2) * dlog_sigma_dt
                drift_NO  = g_squared * self.lambda_NO * score_NO

                drift_guidance = -(drift_sensor + drift_NO)
            else:
                drift_guidance = 0.0

            return drift_prior + drift_guidance

        def _diffusion(x: Tensor, t: Tensor, params: Params) -> Tensor:
            del x, params
            assert t.ndim == 0
            if not t.requires_grad:
                t.requires_grad_(True)
            return torch.sqrt(dsquare_dt(self.scheme.sigma)(t)) * self.scheme.scale(t)

        return sde.SdeDynamics(_drift, _diffusion)

class OdeSampler(Sampler):
    """Use a probability flow ODE to generate samples or compute log likelihood.

    Attributes:
        integrator: The ODE solver for solving the sampling ODE.
        num_probes: The number of probes to use for Hutchinson's trace estimator
        when computing the log likelihood of samples. If `None`, the trace is
        computed exactly.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        scheme: diffusion.Diffusion,
        denoise_fn: DenoiseFn,
        tspan: Tensor,
        guidance_transforms: Sequence[guidance.Transform] = (),
        apply_denoise_at_end: bool = True,
        return_full_paths: bool = False,
        integrator: ode.OdeSolver = ode.HeunsMethod(),
        num_probes: int | None = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__(
            input_shape=input_shape,
            scheme=scheme,
            denoise_fn=denoise_fn,
            tspan=tspan,
            guidance_transforms=guidance_transforms,
            apply_denoise_at_end=apply_denoise_at_end,
            return_full_paths=return_full_paths,
            device=device,
            dtype=dtype,
        )

        self.integrator = integrator
        self.num_probes = num_probes

    def denoise(
        self,
        noisy: Tensor,
        tspan: Tensor,
        y: Optional[Tensor] = None,
        lead_time: Optional[Tensor] = None,
        cond: TensorMapping | None = None,
        guidance_inputs: TensorMapping | None = None,
    ) -> Tensor:
        """Applies iterative denoising to given noisy states."""

        if self.integrator is None:
            self.integrator = ode.HeunsMethod()


        if self.integrator.terminal_only and self.return_full_paths:
            raise ValueError(
                f"Integrator type `{type(self.integrator)}` does not support"
                " returning full paths."
            )

        params = dict(cond=cond, guidance_inputs=guidance_inputs)
        # The lead axis should always be time.
        denoised = self.integrator(
            func=self.dynamics, 
            x0=noisy, 
            tspan=tspan, 
            params=params,
            y=y,
            lead_time=lead_time
        )
        # ODE solvers may return either the full paths or the terminal state only.
        # If the former, the lead axis should be time.
        samples = denoised if self.integrator.terminal_only else denoised[-1]
        return denoised if self.return_full_paths else samples

    @property
    def dynamics(self) -> ode.OdeDynamics:
        """The right-hand side function of the sampling ODE.

        In score function (eq. 3 in Karras et al. https://arxiv.org/abs/2206.00364):

        dx = [ṡ(t)/s(t) x - s(t)² σ̇(t)σ(t) ∇pₜ(x)] dt,

        or, in terms of denoise function (eq. 81):

        dx = [σ̇(t)/σ(t) + ṡ(t)/s(t)] x - [s(t)σ̇(t)/σ(t)] D(x/s(t), σ(t)) dt

        where s(t), σ(t) are the scale and noise schedule of the diffusion scheme.
        """

        def _dynamics(
            x: Tensor, 
            t: Tensor, 
            params: Params,
            y: Optional[Tensor] = None,
            lead_time: Optional[Tensor] = None
        ) -> Tensor:
            assert t.ndim == 0, "`t` must be a scalar."
            denoise_fn = self.get_guided_denoise_fn(
                guidance_inputs=params["guidance_inputs"]
            )
            s, sigma = self.scheme.scale(t), self.scheme.sigma(t)
            x_hat = x / s
            if not t.requires_grad:
                t.requires_grad_(True)
            dlog_sigma_dt = dlog_dt(self.scheme.sigma)(t)
            dlog_s_dt = dlog_dt(self.scheme.scale)(t)

            denoiser_output = denoise_fn_output(
                denoise_fn=denoise_fn,
                x=x_hat, 
                sigma=sigma, 
                cond=params["cond"],
                y=y,
                lead_time=lead_time
            )
            return (dlog_sigma_dt + dlog_s_dt) * x - dlog_sigma_dt * s * denoiser_output

        return _dynamics


class DPSOdeSampler(OdeSampler):
    """
    Posterior probability-flow ODE sampler.

    The probability-flow ODE for p(u_τ | y, u_NO) is:

        du/dt = (ṡ/s) u  −  s² σ σ̇  ∇ log p_τ(u_τ | y, u_NO)

    The posterior score decomposes as:

        ∇ log p_τ(u_τ | y, u_NO) = ∇ log p_τ(u_τ)        [prior, via denoiser]
                                  + ∇ log p(y | u_τ)       [sensor, DPS approx]
                                  + ∇ log p(u_NO | u_τ)    [NO, spectral]

    The prior score enters the ODE with coefficient  c(τ) = s² σ² (σ̇/σ).
    Both likelihood scores enter with the SAME coefficient — this is what
    the probability-flow ODE prescribes.

    Hyperparameters:
        lambda_sensor : correction for DPS approximation quality  (≈ 1)
        lambda_NO     : correction for spectral model calibration (≈ 1)
    """

    def __init__(
        self,
        input_shape:      tuple[int, ...],
        scheme:           diffusion.Diffusion,
        denoise_fn:       DenoiseFn,
        tspan:            Tensor,
        likelihood,                                # CombinedLikelihood
        lambda_sensor:    float = 1.0,
        lambda_NO:        float = 1.0,
        integrator:       ode.OdeSolver = ode.HeunsMethod(terminal_only=True),
        guidance_transforms: Sequence[guidance.Transform] = (),
        apply_denoise_at_end: bool = True,
        return_full_paths:    bool = False,
        device:           torch.device = None,
        dtype:            torch.dtype = torch.float32,
    ):
        super().__init__(
            input_shape=input_shape, scheme=scheme, denoise_fn=denoise_fn,
            tspan=tspan, integrator=integrator,
            guidance_transforms=guidance_transforms,
            apply_denoise_at_end=apply_denoise_at_end,
            return_full_paths=return_full_paths,
            device=device, dtype=dtype,
        )
        self.likelihood     = likelihood
        self.lambda_sensor  = float(lambda_sensor)
        self.lambda_NO      = float(lambda_NO)
    @property
    def dynamics(self) -> ode.OdeDynamics:

        def _dynamics(
            x:         Tensor,
            t:         Tensor,
            params:    Params,
            y:         Tensor = None,
            lead_time: Tensor = None,
        ) -> Tensor:
            assert t.ndim == 0

            denoise_fn = self.get_guided_denoise_fn(params.get("guidance_inputs"))
            s     = self.scheme.scale(t)
            sigma = self.scheme.sigma(t)

            if not t.requires_grad:
                t.requires_grad_(True)
            dlog_sigma_dt = dlog_dt(self.scheme.sigma)(t)
            dlog_s_dt     = dlog_dt(self.scheme.scale)(t)

            # ── (i) Prior drift ──────────────────────────────────
            x_in   = x.detach().requires_grad_(True)
            x0_hat = denoise_fn_output(
                denoise_fn=denoise_fn, x=x_in / s, sigma=sigma,
                cond=params["cond"], y=None, lead_time=None,
            )
            drift_prior = (
                (dlog_sigma_dt + dlog_s_dt) * x
                - dlog_sigma_dt * s * x0_hat
            )

            # ── (ii)+(iii) Guidance ──────────────────────────────
            if y is not None:
                x0 = s * x0_hat

                g_sensor, g_NO, res_norm = self.likelihood.grad_log_likelihood(
                    x0_hat    = x0,
                    x_hat_tau = (x_in / s).detach(),
                    y         = y,
                    sigma_tau = sigma.detach(),
                )

                # ── Sensor: DPS step-size rule ───────────────────
                #
                #   The DPS approximation breaks the magnitude
                #   contract that c(τ) relies on (score vanishes
                #   as residual → 0, compounding with c(τ) → 0).
                #   The 1/‖residual‖ normalization compensates.
                #
                #   drift_sensor = λ_s / ‖M(x̂₀)−y‖ · (∂x̂₀/∂u_τ)ᵀ ∇ log p(y|x̂₀)
                #
                g_vjp, = torch.autograd.grad(
                    outputs      = x0,
                    inputs       = x_in,
                    grad_outputs = g_sensor,
                    retain_graph = False,
                    create_graph = False,
                )
                g_vjp = g_vjp.detach()
                drift_sensor = self.lambda_sensor * g_vjp / (res_norm + 1e-8)

                # ── NO: principled ODE coefficient ───────────────
                #
                #   The spectral score has the correct σ_τ-dependent
                #   magnitude by construction (via α(k) and Λ_τ(k)),
                #   so it gets the same c(τ) as the prior score.
                #
                #   drift_NO = c(τ) · λ_NO · ∇_{u_τ} log p(u_NO | u_τ)
                #
                score_NO = g_NO.detach() / s.detach()
                c_tau = (s ** 2) * (sigma ** 2) * dlog_sigma_dt
                drift_NO = c_tau * self.lambda_NO * score_NO

                drift_guidance = -(drift_sensor + drift_NO)
            else:
                drift_guidance = 0.0

            return drift_prior + drift_guidance

        return _dynamics

def _should_log(t: Tensor, period: float = 0.1, width: float = 0.01) -> bool:
    """Log roughly once per `period` in diffusion time."""
    try:
        return (t.item() % period) < width
    except Exception:
        return False