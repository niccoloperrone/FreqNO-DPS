# FreqNO-DPS: Correcting Neural Operator Spectral Bias via Diffusion 
# Posterior Sampling with Sparse Observations.
# Copyright (C) 2026 Niccolò Perrone, Fanny Lehmann, Stefania Fresca,
# Filippo Gatti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Likelihoods for diffusion posterior sampling

Three likelihoods and two combiners:

- SpectralNOLikelihood       : per-mode spectrally-shaped NO guidance (ours)
- PixelSpaceNOLikelihood     : isotropic NO guidance (DPS baseline)
- SparseSensorLikelihood     : sparse sensor observations
- CombinedLikelihood         : SpectralNO + Sensor, two-path output
- StandardDPSCombinedLikelihood : PixelSpaceNO + Sensor, both via VJP
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpectralNOLikelihood(nn.Module):
    """Neural-operator guidance via exact spectral marginalization

    The exact marginal is

        F(u_NO)(c,k) | u_tau  ~  CN( H(k) alpha(k) F(x_hat_tau)(c,k),  Lambda_tau(k) )

    with alpha(k) = P_u(k) / (sigma_tau^2 + P_u(k))
    and  Lambda_tau = sigma_NO^2 + |H|^2 * Sigma_post

    The score w.r.t. x_hat_tau is computed directly (no denoiser, no VJP):

        nabla_{x_hat_tau} log p(u_NO | u_tau) = F^{-1}[ Lambda_tilde * r_tilde ]

    with Lambda_tilde(k) = 2 H*(k) alpha(k) / Lambda_tau(k)
    and  r_tilde(c,k)    = F(u_NO)(c,k) - H(k) alpha(k) F(x_hat_tau)(c,k)

    Parameters
    ----------
    H         : (C, Nx, Ny, Nf) float32   transfer function per channel
    sigma2_NO : (C, Nx, Ny, Nf) float32   residual variance per channel
    P_u       : (C, Nx, Ny, Nf) float32   signal PSD per channel
    """

    def __init__(self, H: Tensor, sigma2_NO: Tensor, P_u: Tensor, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("H",         H.to(torch.float32))
        self.register_buffer("sigma2_NO", sigma2_NO.to(torch.float32))
        self.register_buffer("P_u",       P_u.to(torch.float32))
        self.eps = float(eps)

    def grad_log_likelihood(
        self,
        x_hat_tau: Tensor,
        u_NO:      Tensor,
        sigma_tau: Tensor,
    ) -> Tensor:
        """Direct score w.r.t. x_hat_tau

        Parameters
        ----------
        x_hat_tau : (B, C, Nx, Ny, Nt) noisy input
        u_NO      : (B, C, Nx, Ny, Nt) NO prediction (fixed, real)
        sigma_tau : scalar             current noise level

        Returns
        -------
        (B, C, Nx, Ny, Nt) real-valued gradient
        """
        sigma2_tau = float(sigma_tau.detach().item()) ** 2

        F_xt  = torch.fft.rfftn(x_hat_tau, dim=(-3, -2, -1), norm="ortho")
        F_uNO = torch.fft.rfftn(u_NO,      dim=(-3, -2, -1), norm="ortho")

        H  = self.H.to(device=x_hat_tau.device)
        s2 = self.sigma2_NO.to(device=x_hat_tau.device)
        Pu = self.P_u.to(device=x_hat_tau.device)

        # Wiener filter and posterior variance
        alpha      = Pu / (sigma2_tau + Pu + self.eps)
        Sigma_post = sigma2_tau * alpha

        # Residual centered at marginal mean H*alpha*F(x_hat_tau)
        H_alpha = H * alpha.to(torch.complex64)
        r = F_uNO - H_alpha.unsqueeze(0) * F_xt

        # Weight: 2 H*(k) alpha(k) / (sigma_NO^2 + |H|^2 Sigma_post)
        H2    = torch.abs(H).pow(2)
        denom = s2 + H2 * Sigma_post + self.eps
        Lambda_tilde = 2.0 * H_alpha / denom

        # Weighted residual back to spatial domain
        g_fourier = Lambda_tilde.unsqueeze(0) * r
        g = torch.fft.irfftn(
            g_fourier, s=x_hat_tau.shape[-3:], dim=(-3, -2, -1), norm="ortho",
        )
        return g.real


class PixelSpaceNOLikelihood(nn.Module):
    """Isotropic DPS likelihood for the NO prediction

    Model:
        u_NO | x0 ~ N(x0, sigma2_NO * I)

    Gradient w.r.t. x0:
        nabla_{x0} log p = (u_NO - x0) / sigma2_NO

    This gradient is back-propagated through the denoiser via VJP
    (standard DPS mechanism)
    """

    def __init__(self, sigma2_NO: float, eps: float = 1e-8):
        super().__init__()
        self.sigma2_NO = float(sigma2_NO)
        self.eps = float(eps)

    def grad_log_likelihood_x0(self, x0: Tensor, u_NO: Tensor) -> Tensor:
        return (u_NO - x0) / (self.sigma2_NO + self.eps)


class SparseSensorLikelihood(nn.Module):
    """Sparse sensor observations: y = M(x) + noise"""

    def __init__(self, operator, sigma_obs: float = 0.01, eps: float = 1e-8):
        super().__init__()
        self.operator = operator
        self.sigma_obs = float(sigma_obs)
        self.eps = float(eps)

    def grad_and_residual_norm(self, x0_hat: Tensor, y: Tensor):
        """Return (M^T(y - Mx0), ||y - Mx0||) for DPS step-size normalization"""
        Mx0      = self.operator(x0_hat)
        residual = y - Mx0
        r_norm = (
            torch.linalg.vector_norm(residual.reshape(residual.shape[0], -1), dim=1)
            .clamp(min=self.eps)
            .view(-1, 1, 1, 1, 1)
        )
        return self.operator.adjoint(residual), r_norm


class CombinedLikelihood(nn.Module):
    """SpectralNO + Sensor with two-path output

    Returns (g_sensor, g_NO, res_norm):
        g_sensor : raw M^T(y - Mx0)         for VJP path, scaled by zeta/||r|| in sampler
        g_NO     : spectral score w.r.t. x_hat_tau, for direct path, scaled by g^2(tau)
        res_norm : ||M(x0) - y||             for DPS step-size normalization
    """

    def __init__(
        self,
        likelihood_NO:  SpectralNOLikelihood,
        likelihood_obs: SparseSensorLikelihood,
    ):
        super().__init__()
        self.likelihood_NO  = likelihood_NO
        self.likelihood_obs = likelihood_obs

    def grad_log_likelihood(
        self,
        x0_hat:    Tensor,
        x_hat_tau: Tensor,
        y:         tuple,
        sigma_tau: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        u_NO, y_obs = y
        g_sensor, res_norm = self.likelihood_obs.grad_and_residual_norm(x0_hat, y_obs)
        g_NO = self.likelihood_NO.grad_log_likelihood(x_hat_tau, u_NO, sigma_tau)
        return g_sensor, g_NO, res_norm


class StandardDPSCombinedLikelihood(nn.Module):
    """PixelSpaceNO + Sensor, both gradients via VJP

    Both terms back-propagate through the denoiser. This matches the
    standard DPS formulation (Chung et al. 2023) used as the DPS_NO_iso
    baseline

    Returns (g_combined, zeros, ones) so the sampler's standard pipeline
    is reused: VJP is applied to g_combined, no separate NO path
    """

    def __init__(
        self,
        likelihood_NO:  PixelSpaceNOLikelihood,
        likelihood_obs: SparseSensorLikelihood,
        lambda_NO:  float = 1.0,
        lambda_obs: float = 1.0,
    ):
        super().__init__()
        self.likelihood_NO  = likelihood_NO
        self.likelihood_obs = likelihood_obs
        self.lambda_NO  = float(lambda_NO)
        self.lambda_obs = float(lambda_obs)

    def grad_log_likelihood(
        self,
        x0_hat:    Tensor,
        x_hat_tau: Tensor,
        y:         tuple,
        sigma_tau: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        u_NO, y_obs = y

        # Sensor: M^T(y - Mx0) with per-sample residual norm
        g_sensor, r_sensor = self.likelihood_obs.grad_and_residual_norm(x0_hat, y_obs)

        # NO: isotropic L2 with per-sample residual norm
        g_NO = self.likelihood_NO.grad_log_likelihood_x0(x0_hat, u_NO)
        diff = u_NO - x0_hat
        r_NO = (
            torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=1)
            .clamp(min=1e-8)
            .view(-1, 1, 1, 1, 1)
        )

        # Pre-combine with DPS step-size rule (linear in J^T)
        g_combined = (
            self.lambda_obs * g_sensor / r_sensor
            + self.lambda_NO * g_NO     / r_NO
        )

        # Sampler does VJP(g_sensor) * lambda_sensor / res_norm
        # We want VJP(g_combined) * 1 / 1, so return res_norm=1 and g_NO=0
        return g_combined, torch.zeros_like(x0_hat), torch.ones_like(r_sensor)


def load_spectral_likelihood(
    pt_path: str,
    device:  torch.device,
    eps:     float = 1e-8,
) -> SpectralNOLikelihood:
    ckpt = torch.load(pt_path, map_location=device)
    return SpectralNOLikelihood(
        H=ckpt["H_real"].to(device),
        sigma2_NO=ckpt["sigma2_real"].to(device),
        P_u=ckpt["P_u"].to(device),
        eps=eps,
    )


def load_pixel_space_NO_likelihood(
    pt_path: str,
    device:  torch.device,
    eps:     float = 1e-8,
) -> PixelSpaceNOLikelihood:
    """Build isotropic NO likelihood from estimate_sigma2_iso.py output"""
    ckpt = torch.load(pt_path, map_location=device)
    sigma2_iso = float(ckpt["sigma2_iso_global"].item())
    print(f"[PixelSpaceNOLikelihood] sigma2_NO,iso (global) = {sigma2_iso:.6e}")
    return PixelSpaceNOLikelihood(sigma2_NO=sigma2_iso, eps=eps)