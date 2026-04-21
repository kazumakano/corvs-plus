from typing import Literal, Optional
import torch
from torch import nn
from torch.nn.modules import loss
from torchvision import ops


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """
    Modified from `torch.nn.BCEWithLogitsLoss`.
    This class supports label smoothing.
    """

    def __init__(self, weight: Optional[torch.Tensor] = None, reduction: Literal["mean", "sum", "none"] = "mean", pos_weight: Optional[torch.Tensor] = None, label_smoothing: float = 0) -> None:
        super().__init__(weight=weight, reduction=reduction, pos_weight=pos_weight)
        self.label_smoothing = label_smoothing

    def forward(self, input: torch.FloatTensor, target: torch.FloatTensor) -> torch.FloatTensor:  # (..., ), (..., ) -> (..., )
        with torch.no_grad():
            target = (1 - self.label_smoothing) * target + 0.5 * self.label_smoothing
        return super().forward(input, target)

class FocalWithLogitsLoss(loss._Loss):
    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: Literal["mean", "sum", "none"] = "mean") -> None:
        super().__init__(reduction=reduction)
        self.alpha, self.gamma = alpha, gamma

    def forward(self, input: torch.FloatTensor, target: torch.FloatTensor) -> torch.FloatTensor:  # (..., ), (..., ) -> (..., )
        return ops.sigmoid_focal_loss(input, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
