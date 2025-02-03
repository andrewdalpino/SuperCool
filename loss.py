import torch

from torch import Tensor

from torch.nn import Module


class TVLoss(Module):
    """Total variation (TV) penalty as a loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor):
        b, c, h, w = y_pred.size()

        h_delta = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        w_delta = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]

        h_variance = torch.pow(h_delta, 2).sum()
        w_variance = torch.pow(w_delta, 2).sum()

        h_variance /= b * c * (h - 1) * w
        w_variance /= b * c * h * (w - 1)

        penalty = w_variance + h_variance

        return penalty
