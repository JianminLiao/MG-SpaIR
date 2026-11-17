import math

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def coordinate_grid(height, width, device):
    """Uniform grid over [-1, 1]^2, shape [1, height, width, 2] with (x, y) in the last axis."""
    xs = np.linspace(-1, 1, width, dtype=np.float32)
    ys = np.linspace(-1, 1, height, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1)
    return torch.from_numpy(grid).unsqueeze(0).to(device)


def gaussian_kernel(sigma, channels):
    """Normalized Gaussian kernel of size 2 * ceil(3 * sigma) + 1, shaped [C, 1, K, K]."""
    size = 2 * math.ceil(3 * sigma) + 1
    x = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g).view(1, 1, size, size).repeat(channels, 1, 1, 1)


def blur(x, kernel):
    pad = kernel.shape[-1] // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x, kernel, groups=x.shape[1])


def downsample(x, scale):
    return F.interpolate(x, scale_factor=1.0 / scale, mode="bicubic", align_corners=True)


def gradients(x):
    """Forward differences along x then y, concatenated to [B, 2C, H, W]."""
    padded_x = F.pad(x, (0, 1, 0, 0), mode="replicate")
    padded_y = F.pad(x, (0, 0, 0, 1), mode="replicate")
    gx = padded_x[..., 1:] - padded_x[..., :-1]
    gy = padded_y[..., 1:, :] - padded_y[..., :-1, :]
    return torch.cat([gx, gy], dim=1)


def psnr_ssim(reference, image):
    psnr = peak_signal_noise_ratio(reference, image, data_range=1)
    ssim = structural_similarity(reference, image, channel_axis=2, win_size=7, data_range=1)
    return psnr, ssim
