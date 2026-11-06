"""ShallowNetT: plain-torch copy of braindecode 1.8.1 ShallowFBCSPNet (sprint 2026-11-09 W0).

The server image's braindecode is unpinned, so the submission defines the net
itself. Same layers and forward maths as braindecode 1.8.1 (BSD-3-Clause):
Ensure4d -> (B, 1, T, C) -> CombinedConv (time conv x spatial conv merged into
one conv, bias = spat_w.sum(-1) @ time_bias) -> BatchNorm2d -> square ->
AvgPool (75, stride 15) -> log(clamp(x, 1e-6)) -> dropout -> conv classifier
(final length 26 at 480 samples) -> squeeze. State-dict keys are braindecode's,
so ``load_state_dict`` takes a braindecode state dict unchanged.
This block is embedded verbatim in solvers/bci_decoding/ens_x4f_shallow.py.
"""

import torch
from torch import nn
from torch.nn import functional as F


class _CombinedConv(nn.Module):
    def __init__(self, in_chans, n_time=40, n_spat=40, k=25):
        super().__init__()
        self.conv_time = nn.Conv2d(1, n_time, (k, 1), bias=True)
        self.conv_spat = nn.Conv2d(n_time, n_spat, (1, in_chans), bias=False)

    def forward(self, x):
        w = (self.conv_time.weight * self.conv_spat.weight.permute(1, 0, 2, 3)).sum(0).unsqueeze(1)
        b = self.conv_spat.weight.squeeze().sum(-1).mm(self.conv_time.bias.unsqueeze(-1)).squeeze()
        return F.conv2d(x, weight=w, bias=b, stride=(1, 1))


class _Final(nn.Module):
    def __init__(self, n_in, n_out, length):
        super().__init__()
        self.conv_classifier = nn.Conv2d(n_in, n_out, (length, 1), bias=True)

    def forward(self, x):
        x = self.conv_classifier(x)[..., 0]          # (B, n_out, 1)
        return x.squeeze(-1) if x.shape[-1] == 1 else x


class ShallowNetT(nn.Module):
    def __init__(self, n_chans=41, n_outputs=3, n_times=480, n_filters=40,
                 filter_len=25, pool_len=75, pool_stride=15, drop=0.5):
        super().__init__()
        self.conv_time_spat = _CombinedConv(n_chans, n_filters, n_filters, filter_len)
        self.bnorm = nn.BatchNorm2d(n_filters, momentum=0.1, affine=True)
        self.pool = nn.AvgPool2d((pool_len, 1), stride=(pool_stride, 1))
        self.drop = nn.Dropout(drop)
        length = ((n_times - filter_len + 1) - pool_len) // pool_stride + 1
        self.final_layer = _Final(n_filters, n_outputs, length)

    def forward(self, x):
        while x.dim() < 4:
            x = x.unsqueeze(-1)                       # (B, C, T, 1)
        x = x.permute(0, 3, 2, 1)                     # (B, 1, T, C)
        x = self.conv_time_spat(x)
        x = self.bnorm(x)
        x = x * x
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.drop(x)
        return self.final_layer(x)
