import math

import numpy as np
import torch
import torch.nn as nn

from wire import ComplexGaborLayer2D


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Grade(nn.Module):
    """One grade: hidden layers `base` followed by a linear head `output_layer`.

    The input is [B, H, W, C_in]: coordinates for the first grade, the hidden
    features of the previous grade otherwise. Returns the hidden features
    [B, H, W, hidden] and the image-space output [B, C_out, H, W].
    """

    def forward(self, x):
        b, h, w, c = x.shape
        features = self.base(x.view(-1, c))
        out = self.to_image(self.output_layer(features).view(b, h, w, -1))
        return features.view(b, h, w, -1), out

    @staticmethod
    def to_image(out):
        return out.permute(0, 3, 1, 2)


class SirenGrade(Grade):
    def __init__(self, in_channels, out_channels, num_layers, hidden_dim):
        super().__init__()
        from_coords = in_channels == 2
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_channels if i == 0 else hidden_dim, hidden_dim))
            layers.append(Sine(30.0 if i == 0 and from_coords else 1.0))
        self.base = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, out_channels)

        linears = [m for m in self.base if isinstance(m, nn.Linear)] + [self.output_layer]
        for i, linear in enumerate(linears):
            fan_in = linear.weight.size(1)
            bound = 1.0 / fan_in if i == 0 and from_coords else math.sqrt(6.0 / fan_in)
            nn.init.uniform_(linear.weight, -bound, bound)
            nn.init.zeros_(linear.bias)


class WireGrade(Grade):
    def __init__(self, in_channels, out_channels, num_layers, hidden_dim):
        super().__init__()
        layers = [ComplexGaborLayer2D(in_channels, hidden_dim, is_first=in_channels == 2)]
        for _ in range(num_layers - 1):
            layers.append(ComplexGaborLayer2D(hidden_dim, hidden_dim))
        self.base = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, out_channels, dtype=torch.cfloat)

    @staticmethod
    def to_image(out):
        return out.real.permute(0, 3, 1, 2).contiguous()


def build_grade(backbone, in_channels, out_channels, num_layers, hidden_dim):
    grades = {"siren": SirenGrade, "wire": WireGrade}
    if backbone not in grades:
        raise ValueError(f"Unknown backbone '{backbone}', expected one of {sorted(grades)}")
    return grades[backbone](in_channels, out_channels, num_layers, hidden_dim)


@torch.inference_mode()
def render(grades, height, width, device, tiles=4):
    """Evaluate the sum of all grade outputs on a height x width grid over [-1, 1]^2.

    Returns a float32 array [height, width, C] clipped to [0, 1]. The grid is
    processed in `tiles` horizontal strips to bound memory use.
    """
    grid = np.stack(
        np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height)), axis=-1
    ).astype(np.float32)

    strips = []
    for rows in np.array_split(grid, tiles, axis=0):
        x = torch.from_numpy(rows).unsqueeze(0).to(device)
        total = 0
        for net in grades:
            x, out = net(x)
            total = total + out
        strips.append(total.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1))
    return np.vstack(strips)
