import einops
import torch
from torch import nn
from torch.nn import functional as F


class MaskedGlobalAttnPool1d(nn.Module):
    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("dimension must be divisible by number of heads")

        self.nhead = nhead
        self.proj = nn.Linear(d_model, self.nhead)

    def forward(self, input: torch.FloatTensor, valid_mask: torch.BoolTensor) -> torch.FloatTensor:    # (*, dim, time), (*, time) -> (*, dim)
        score: torch.FloatTensor = self.proj(input.transpose(-2, -1))
        weight = F.softmax(score.masked_fill(~valid_mask.unsqueeze(-1), -torch.inf), dim=-2)
        weight = einops.rearrange(weight, "... t nh -> ... t nh 1")
        input = einops.rearrange(input, "... (nh dh) t -> ... t nh dh", nh=self.nhead)
        output = (weight * input).sum(dim=-3).flatten(start_dim=-2)
        return output

def masked_global_avg_pool1d(input: torch.FloatTensor, valid_mask: torch.BoolTensor | torch.FloatTensor | torch.IntTensor) -> torch.FloatTensor:
    """
    Apply masked global average pooling along time dimension.

    Parameters
    ----------
    input : FloatTensor
        Input.
        Shape is (*, channel, time).
    valid_mask : BoolTensor | FloatTensor | IntTensor
        Mask of valid times.
        It takes True for valid and False for invalid.
        Shape is (*, time).

    output : FloatTensor
        Averaged output.
        Shape is (*, channel).
    """

    return (valid_mask.unsqueeze(-2) * input).sum(dim=-1) / valid_mask.sum(dim=-1, keepdim=True)

def masked_global_max_pool1d(input: torch.FloatTensor, valid_mask: torch.BoolTensor) -> torch.FloatTensor:
    """
    Apply masked global max pooling along time dimension.

    Parameters
    ----------
    input : FloatTensor
        Input.
        Shape is (*, channel, time).
    valid_mask : BoolTensor
        Mask of valid times.
        It takes True for valid and False for invalid.
        Shape is (*, time).

    output : FloatTensor
        Maximum output.
        Shape is (*, channel).
    """

    return input.masked_fill(~valid_mask.unsqueeze(-2), -torch.inf).max(dim=-1).values

def masked_global_softmax_pool1d(input: torch.FloatTensor, valid_mask: torch.BoolTensor, temp: float | nn.Parameter = 1) -> torch.FloatTensor:
    """
    Apply masked global softmax pooling along time dimension.

    Parameters
    ----------
    input : FloatTensor
        Input.
        Shape is (*, channel, time).
    valid_mask : BoolTensor
        Mask of valid times.
        It takes True for valid and False for invalid.
        Shape is (*, time).
    temp : float | Parameter
        Temperature parameter.
        Standard if 1.

    Returns
    -------
    output : FloatTensor
        Soft maximum output.
        Shape is (*, channel).
    """

    return (F.softmax((input / temp).masked_fill(~valid_mask.unsqueeze(-2), -torch.inf), dim=-1) * input).sum(dim=-1)
