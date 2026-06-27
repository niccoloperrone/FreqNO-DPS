import torch
from torch.nn import L1Loss, MSELoss
import numpy as np
import torch.nn.functional as F


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device

def get_batch_size(x_input):
    if isinstance(x_input, list) or isinstance(x_input, tuple):
        return x_input[0].size(0)
    else:
        return x_input.size(0)

def loss_rel_global(pred, targ, eps=1e-6):
    # pred,targ: (B,32,32,T,1)
    scale = targ.abs().mean(dim=(1,2,3,4), keepdim=True) + eps
    return ((pred - targ).abs() / scale).mean()

def loss_rel_trace(pred, targ, eps=1e-6, s_min=None):
    # pred,targ: (B,32,32,T,1)
    # scale per trace (x,y): mean over time (and channel dim=1)
    scale = targ.abs().mean(dim=(3,4), keepdim=True)  # (B,32,32,1,1)

    if s_min is not None:
        # floor in the same units as scale (mean abs amplitude)
        scale = torch.clamp(scale, min=s_min)

    scale = scale + eps
    return ((pred - targ).abs() / scale).mean()

def loss_asinh_rel(pred, targ, eps=1e-6):
    # Apply asinh to the individual predictions and targets
    #p_asinh = torch.asinh(pred)
    t_asinh = torch.asinh(targ)
    
    # Calculate the scale in the transformed space
    # Using the mean of the absolute transformed targets
    scale = t_asinh.abs().mean(dim=(1,2,3,4), keepdim=True) + eps
    
    # Return the relative error in asinh space
    return (pred - t_asinh).abs().mean() / (scale)

def loss_criterion(a, b, weights=(1.0, 0.0), eps=1e-6, relative=True):
    # tuple support (E,N,Z)
    if isinstance(a, tuple):
        if relative:
            L1 = sum(loss_rel_global(ai, bi, eps=eps) for ai, bi in zip(a,b))
            L2 = 0.0
        else:
            L1 = sum((ai - bi).abs().mean() for ai, bi in zip(a,b))
            L2 = sum(((ai - bi)**2).mean() for ai, bi in zip(a,b))
    else:
        if relative:
            L1 = sum(loss_rel_global(ai, bi, eps=eps) for ai, bi in zip(a,b))
            L2 = 0.0
        else:
            L1 = (a - b).abs().mean()
            L2 = ((a - b)**2).mean()

    return weights[0]*L1 + weights[1]*L2

class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
