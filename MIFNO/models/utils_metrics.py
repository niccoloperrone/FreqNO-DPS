import numpy as np
from scipy.fft import rfft, rfftfreq
import torch


def fourier_spectra(velocity_values, low_freq, mid_freq, high_freq, dt=0.01):
    ''' Compute the mean of Fourier coefficients in three frequency intervals

    Input
    velocity values: array (N, Nx, Ny, Nt) where N: number of samples, Nx, Ny: number of points lalong x and y, Nt is the number of equally-spaced time steps
    low_freq: tuple (float, float), frequency range of the low frequency band
    mid_freq: tuple (float, float), frequency range of the medium frequency band
    high_freq: tuple (float, float), frequency range of the high frequency band
    dt: float, time step in seconds 
    
    Output: tuple (array (N, Nx, Ny), array (N, Nx, Ny), array (N, Nx, Ny)) where each array contains the mean of Fourier coefficients
    for the corresponding frequencies '''
    
    # compute the Fourier coefficients of each signal and associate the corresponding frequency
    list_freq = rfftfreq(velocity_values.shape[-1], d=dt)
    indices_low_freq = np.where((list_freq>=low_freq[0])&(list_freq<=low_freq[1]))[0]
    indices_mid_freq = np.where((list_freq>mid_freq[0])&(list_freq<=mid_freq[1]))[0]
    indices_high_freq = np.where((list_freq>high_freq[0])&(list_freq<=high_freq[1]))[0]    
    
    fourier = rfft(velocity_values)
    fourier_low_freq = np.mean(np.abs(fourier[:, :, :, indices_low_freq]), axis=-1)
    fourier_mid_freq = np.mean(np.abs(fourier[:, :, :, indices_mid_freq]), axis=-1)
    fourier_high_freq = np.mean(np.abs(fourier[:, :, :, indices_high_freq]), axis=-1)
    
    return (fourier_low_freq, fourier_mid_freq, fourier_high_freq)


def lowpass_torch_butter(
    u: torch.Tensor,
    cutoff_hz: float = 2.5,
    dt: float = 0.02,
    order: int = 4,
    dim: int = -1,          # which axis is time
) -> torch.Tensor:
    """
    Butterworth-like low-pass along dimension `dim`.
    Zero-phase (frequency-domain magnitude filter).

    Works for any tensor shape, as long as the chosen `dim` is the time axis.
    """
    dim = dim % u.ndim  # support negative dims safely

    # FFT along selected dim
    U = torch.fft.rfft(u, dim=dim)
    T = u.shape[dim]
    freqs = torch.fft.rfftfreq(T, d=dt).to(device=u.device)

    # Butterworth magnitude response
    H = 1.0 / torch.sqrt(1.0 + (freqs / cutoff_hz) ** (2 * order))

    # reshape H to broadcast on all dims except `dim`
    shape = [1] * u.ndim
    shape[dim] = H.numel()
    H = H.view(*shape)

    # Apply and invert
    U_filt = U * H
    u_lp = torch.fft.irfft(U_filt, n=T, dim=dim)
    return u_lp

