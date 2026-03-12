from typing import Literal
import torch
from torch.nn.modules import loss
from torchvision import ops


class FocalWithLogitsLoss(loss._Loss):
    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: Literal["mean", "sum", "none"] = "mean") -> None:
        super().__init__(reduction=reduction)
        self.alpha, self.gamma = alpha, gamma

    def forward(self, input: torch.FloatTensor, target: torch.FloatTensor) -> torch.FloatTensor:    # (*, ), (*, ) -> (*, )
        return ops.sigmoid_focal_loss(input, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
