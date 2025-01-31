import torch

import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.size()

        h_delta = x[:, :, 1:, :] - x[:, :, :-1, :]
        w_delta = x[:, :, :, 1:] - x[:, :, :, :-1]

        h_variance = torch.pow(h_delta, 2).sum()
        w_variance = torch.pow(w_delta, 2).sum()

        h_variance /= b * c * (h - 1) * w
        w_variance /= b * c * h * (w - 1)

        average_tv = w_variance + h_variance

        return average_tv


class WassersteinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        return torch.mean(y_pred * y)
