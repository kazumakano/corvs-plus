import einops
import torch
from torch import nn
from torch.nn import functional as F


def masked_global_avg_pool1d(input: torch.FloatTensor, valid_mask: torch.BoolTensor | torch.FloatTensor | torch.IntTensor) -> torch.FloatTensor:
    """
    Apply masked global average pooling along temporal dimension.

    Parameters
    ----------
    input : FloatTensor
        Input.
        Shape is (*, ch, time).
    valid_mask : BoolTensor | FloatTensor | IntTensor
        Mask of valid times.
        True for valid and False for invalid.
        Shape is (*, time).

    output : FloatTensor
        Averaged output.
        Shape is (*, ch).
    """

    return (valid_mask.unsqueeze(-2) * input).sum(dim=-1) / valid_mask.sum(dim=-1, keepdim=True)

def masked_global_max_pool1d(input: torch.FloatTensor, valid_mask: torch.BoolTensor) -> torch.FloatTensor:
    """
    Apply masked global max pooling along temporal dimension.

    Parameters
    ----------
    input : FloatTensor
        Input.
        Shape is (*, ch, time).
    valid_mask : BoolTensor
        Mask of valid times.
        True for valid and False for invalid.
        Shape is (*, time).

    output : FloatTensor
        Maximum output.
        Shape is (*, ch).
    """

    return input.masked_fill(~valid_mask.unsqueeze(-2), -torch.inf).max(dim=-1).values

def masked_global_soft_pool1d(input: torch.FloatTensor, valid_mask: torch.BoolTensor, temp: float | nn.Parameter = 1) -> torch.FloatTensor:
    """
    Apply masked global soft pooling along temporal dimension.

    Parameters
    ----------
    input : FloatTensor
        Input.
        Shape is (*, ch, time).
    valid_mask : BoolTensor
        Mask of valid times.
        True for valid and False for invalid.
        Shape is (*, time).
    temp : float | Parameter
        Temperature parameter.
        Standard if 1.

    Returns
    -------
    output : FloatTensor
        Soft maximum output.
        Shape is (*, ch).
    """

    return (F.softmax((input / temp).masked_fill(~valid_mask.unsqueeze(-2), -torch.inf), dim=-1) * input).sum(dim=-1)

class MaskedGlobalAttnPool1d(nn.Module):
    def __init__(self, d_model: int, nhead: int) -> None:
        if d_model % nhead != 0:
            raise ValueError("dimension must be divisible by number of heads")

        super().__init__()

        self.nhead = nhead
        self.proj = nn.Linear(d_model, self.nhead)

    def forward(self, input: torch.FloatTensor, valid_mask: torch.BoolTensor) -> torch.FloatTensor:  # (*, dim, seq), (*, seq) -> (*, dim)
        score: torch.FloatTensor
        score  = self.proj(input.transpose(-2, -1))
        weight = F.softmax(score.masked_fill(~valid_mask.unsqueeze(-1), -torch.inf), dim=-2)
        weight = einops.rearrange(weight, "... s nh -> ... s nh 1")
        input  = einops.rearrange(input, "... (nh dh) s -> ... s nh dh", nh=self.nhead)
        output = (weight * input).sum(dim=-3).flatten(start_dim=-2)

        return output
