import torch
from torch import nn


class MaskedGlobalAttnPool1d(nn.Module):
    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("dimension must be divisible by number of heads")

        self.d_head, self.nhead = d_model // nhead, nhead
        self.proj = nn.Linear(d_model, nhead)

    def forward(self, input: torch.FloatTensor, valid_mask: torch.BoolTensor) -> torch.FloatTensor:    # (*, dim, time), (*, time) -> (*, dim)
        score: torch.FloatTensor = self.proj(input.transpose(-2, -1))    # (*, d_model, time) -> (*, time, nhead)
        score = score.masked_fill(~valid_mask.unsqueeze(-1), -torch.inf)
        output = (score.softmax(-2).unsqueeze(-1) * input.transpose(-2, -1).view(*input.shape[:-2], input.shape[-1], self.nhead, self.d_head)).sum(dim=-3)    # (*, time, nhead), (*, d_model, time) -> (*, nhead, d_head)
        output = output.flatten(start_dim=-2)
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

    return ((input / temp).masked_fill(~valid_mask.unsqueeze(-2), -torch.inf).softmax(-1) * input).sum(dim=-1)
