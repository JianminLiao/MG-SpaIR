# The layer below is adapted from WIRE (https://github.com/vishwa91/wire),
# which is distributed under the following license.
#
# MIT License
#
# Copyright (c) 2022 Vishwanath Saragadam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn


class ComplexGaborLayer2D(nn.Module):
    """Linear map followed by a 2D complex Gabor wavelet nonlinearity."""

    def __init__(self, in_features, out_features, is_first=False, omega0=10.0, sigma0=10.0):
        super().__init__()
        # The first layer reads real coordinates; later layers read complex features.
        dtype = torch.float if is_first else torch.cfloat
        self.omega_0 = nn.Parameter(omega0 * torch.ones(1), requires_grad=False)
        self.scale_0 = nn.Parameter(sigma0 * torch.ones(1), requires_grad=False)
        self.linear = nn.Linear(in_features, out_features, dtype=dtype)
        self.scale_orth = nn.Linear(in_features, out_features, dtype=dtype)

    def forward(self, x):
        lin = self.linear(x)
        scale_y = self.scale_orth(x)
        freq_term = torch.exp(1j * self.omega_0 * lin)
        arg = lin.abs().square() + scale_y.abs().square()
        gauss_term = torch.exp(-self.scale_0 * self.scale_0 * arg)
        return freq_term * gauss_term
