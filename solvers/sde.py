# Copyright 2024 The swirl_dynamics Authors.
# Modifications made by the CAM Lab at ETH Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Solvers for stochastic differential equations (SDEs)."""

from collections.abc import Mapping
from typing import Any, NamedTuple, Protocol, Literal, ClassVar
import torch
from torch import nn

Tensor = torch.Tensor
SdeParams = Mapping[str, Any]

class SdeCoefficientFn(Protocol):
    """A callable type for the drift or diffusion coefficients of an SDE."""

    def forward(self, x: Tensor, t: Tensor, params: SdeParams) -> Tensor:
        """Evaluates the drift or diffusion coefficients."""
        ...

class SdeDynamics(NamedTuple):
    """The drift and diffusion functions that represents the SDE dynamics."""

    drift: SdeCoefficientFn
    diffusion: SdeCoefficientFn

def _check_sde_params_fields(params: SdeParams) -> None:
    if not ("drift" in params.keys() and "diffusion" in params.keys()):
        raise ValueError(
            "'params' must contain both 'drift' and 'diffusion' fields."
        )
    
def output_drift(
        drift: SdeCoefficientFn, 
        x: Tensor, 
        t: Tensor, 
        params: SdeParams, 
        y: Tensor = None, 
        lead_time: Tensor = None
    ) -> Tensor:
    """Evaluate if y or the lead time is required and output the corresponding 
    result
    """
    if y is None and lead_time is None:
        return drift(x=x, t=t, params=params)
    elif y is not None and lead_time is None:
        return drift(x=x, t=t,params=params, y=y)
    elif y is not None and lead_time is not None:
        return drift(x=x, t=t,params=params, y=y, lead_time=lead_time)

  
class SdeSolver(nn.Module):
    """A callable type implementation a SDE solver.
    
    Attributes:
      terminal_only: If 'True' the solver only returns the terminal state,
        i.e., corresponding to the last time stamp in 'tspan'. If 'False',
        returns the full path containing all steps.
    """

    def __init__(self, terminal_only: bool = False):
        super().__init__()
        self.terminal_only = terminal_only

    def forward(
            self,
            dynamics: SdeDynamics,
            x0: Tensor,
            tspan: Tensor,
            y: Tensor = None,
            lead_time: Tensor = None
      ) -> Tensor:
        """Solves an SDE at given time stamps.

        Args:
          dynamics: The SDE dynamics that evaluates the drift and diffusion
            coefficients.
          x0: Initial condition.
          tspan: The sequence of time points on which the approximate solution
            of the SDE are evaluated. The first entry corresponds to the time for x0.

        Returns:
          Integrated SDE trajectory (initial condition included at time position 0).
        """
        raise NotImplementedError
    

class IterativeSdeSolver(nn.Module):
    """A SDE solver based on an iterative step function using PyTorch
    
    Attributes:
      time_axis_pos: The index where the time axis should be placed. Defaults
      to the lead axis (index 0).
    """

    def __init__(
            self,
            time_axis_pos: int = 0, 
            terminal_only: bool = False
    ):
        super().__init__()
        self.time_axis_pos = time_axis_pos
        self.terminal_only = terminal_only

    def step(
            self,
            dynamics: SdeDynamics,
            x0: Tensor,
            t0: Tensor,
            dt: Tensor,
            params: SdeParams,
            y: Tensor = None,
            lead_time: Tensor = None
    ) -> Tensor:
        """Advances the current state one step forward in time."""
        raise NotImplementedError

    def forward(
            self,
            dynamics: SdeDynamics,
            x0: Tensor,
            tspan: Tensor,
            params: SdeParams,
            y: Tensor = None,
            lead_time: Tensor = None
    ) -> Tensor:
        """Solves an SDE by iterating the step function."""
        
        if not self.terminal_only:
            # store the entire path
            x_path = [x0]

        current_state = x0
        for i in range(len(tspan) - 1):
            t0 = tspan[i]
            t_next = tspan[i + 1]
            dt = t_next - t0
            current_state = self.step(
                dynamics=dynamics, 
                x0=current_state, 
                t0=t0, 
                dt=dt, 
                params=params,
                y=y,
                lead_time=lead_time
            ).detach() # to avoid memory issues!

            if not self.terminal_only:
                x_path.append(current_state)
        
        if self.terminal_only:
            return current_state
        else:
            out = torch.stack(x_path, dim=0)
            if self.time_axis_pos != 0:
                out = out.movedim(0, self.time_axis_pos)
            return out
    

class EulerMaruyamaStep(nn.Module):
    """The Euler-Maruyama scheme for integrating the Ito SDE"""

    def step(
            self,
            dynamics: SdeDynamics,
            x0: Tensor,
            t0: Tensor,
            dt: Tensor,
            params: SdeParams,
            y: Tensor = None,
            lead_time: Tensor = None
    ) -> Tensor:
        """Makes one Euler-Maruyama integration step in time."""
        _check_sde_params_fields(params)
        drift_coeffs = output_drift(
            drift=dynamics.drift,
            x=x0,
            y=y,
            t=t0,
            params=params["drift"],
            lead_time=lead_time
        )
        diffusion_coeffs = dynamics.diffusion(x0, t0, params["diffusion"])

        noise = torch.randn(
            size=x0.shape, dtype=x0.dtype, device=x0.device
        )
        return (
            x0 + 
            dt * drift_coeffs + 
            # abs to enable integration backward in time
            diffusion_coeffs * noise * torch.sqrt(torch.abs(dt))
        )
    
class EDMOdeStep(nn.Module):
    """
    Deterministic ODE step that integrates only the drift part of the SDE:
      dx = drift(x, t) dt
    ignoring diffusion. This can be seen as a minimal implementation
    of an ODE-based sampler (like in Karras et al. or DDIM).
    """

    def step(
            self,
            dynamics: SdeDynamics,
            x0: Tensor,
            t0: Tensor,
            dt: Tensor,
            params: SdeParams,
            y: Tensor = None,
            lead_time: Tensor = None
    ) -> Tensor:
        """
        Integrate the ODE using a simple Euler step:
            x_{n+1} = x_n + drift(x_n, t_n)*dt
        ignoring any diffusion term.
        """
        _check_sde_params_fields(params)

        # Evaluate drift
        drift_coeffs = output_drift(
            drift=dynamics.drift,
            x=x0,
            y=y,
            t=t0,
            params=params["drift"],
            lead_time=lead_time
        )

        # ODE update
        return x0 + drift_coeffs * dt
    
    
class EulerMaruyama(EulerMaruyamaStep, IterativeSdeSolver):
    """Solver using the Euler-Maruyama with iteration (i.e. looping through time steps)."""
    def __init__(
            self,
            time_axis_pos: int = 0, 
            terminal_only: bool = False
        ):
        super().__init__()
        IterativeSdeSolver.__init__(
            self, time_axis_pos=time_axis_pos, terminal_only=terminal_only
        )


class EDMOdeSolver(EDMOdeStep, IterativeSdeSolver):
    """Deterministic ODE solver that iterates the ODE step in a loop."""

    def __init__(
        self,
        time_axis_pos: int = 0,
        terminal_only: bool = False
    ):
        super().__init__()
        IterativeSdeSolver.__init__(
            self, time_axis_pos=time_axis_pos, terminal_only=terminal_only
        )


class EDMOdeStepRK4(nn.Module):
    """
    Deterministic ODE step using 4th-order Runge–Kutta (RK4).
    We assume x'(t) = drift(x, t), ignoring any diffusion term.
    """

    def step(
        self,
        dynamics: SdeDynamics,
        x0: Tensor,
        t0: Tensor,
        dt: Tensor,
        params: SdeParams,
        y: Tensor = None,
        lead_time: Tensor = None
    ) -> Tensor:
        """
        Integrate x'(t) = drift(x, t) from t0 to t0 + dt using RK4:
            k1 = f(x0, t0)
            k2 = f(x0 + dt/2 * k1, t0 + dt/2)
            k3 = f(x0 + dt/2 * k2, t0 + dt/2)
            k4 = f(x0 + dt * k3, t0 + dt)
            x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        Returns the state x at t0+dt.
        """
        _check_sde_params_fields(params)  # ensure "drift" in params

        # for convenience:
        # we might also handle time step as scalars
        # but your code uses t0 as a 0D tensor
        dt_2 = dt * 0.5

        # 1) Evaluate drift at (x0, t0)
        k1 = self._drift_eval(dynamics, x0, t0, params, y, lead_time)

        # 2) Evaluate drift at (x0 + dt/2*k1, t0+dt/2)
        x_temp = x0 + dt_2 * k1
        t_temp = t0 + dt_2
        k2 = self._drift_eval(dynamics, x_temp, t_temp, params, y, lead_time)

        # 3) Evaluate drift again with updated state
        x_temp2 = x0 + dt_2 * k2
        k3 = self._drift_eval(dynamics, x_temp2, t_temp, params, y, lead_time)

        # 4) Evaluate drift at final sub-step
        x_temp3 = x0 + dt * k3
        t_temp2 = t0 + dt
        k4 = self._drift_eval(dynamics, x_temp3, t_temp2, params, y, lead_time)

        # Weighted sum
        return x0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _drift_eval(
        self,
        dynamics: SdeDynamics,
        x: Tensor,
        t: Tensor,
        params: SdeParams,
        y: Tensor,
        lead_time: Tensor
    ) -> Tensor:
        """
        Evaluate the drift from your existing code, ignoring diffusion.
        """
        return output_drift(
            drift=dynamics.drift,
            x=x,
            t=t,
            params=params["drift"],
            y=y,
            lead_time=lead_time
        )



class EDMOdeSolverRK4(EDMOdeStepRK4, IterativeSdeSolver):
    """
    Deterministic ODE solver using RK4 steps, iterating over tspan.
    Ignores diffusion, relies on drift only.
    """

    def __init__(
        self,
        time_axis_pos: int = 0,
        terminal_only: bool = True
    ):
        super().__init__()
        IterativeSdeSolver.__init__(
            self, time_axis_pos=time_axis_pos, terminal_only=terminal_only
        )
