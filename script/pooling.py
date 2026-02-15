import torch
from torch import nn


class MaskedAttnPool1d(nn.Module):
    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        self.d_head, self.nhead = d_model // nhead, nhead
        self.proj = nn.Linear(d_model, nhead)

    def forward(self, input: torch.FloatTensor, valid_mask: torch.BoolTensor) -> torch.FloatTensor:    # (batch, channel, time), (batch, time) -> (batch, channel)
        score: torch.Tensor = self.proj(input.transpose(1, 2))    # (batch, d_model, time) -> (batch, time, nhead)
        score = score.masked_fill(~valid_mask.unsqueeze(2), -torch.inf)
        output = (score.softmax(1).unsqueeze(3) * input.transpose(1, 2).view(len(input), input.shape[2], self.nhead, self.d_head)).sum(dim=1).flatten(start_dim=1)    # (batch, time, nhead), (batch, d_model, time) -> (batch, nhead, d_head) -> (batch, d_model)
        return output

def masked_global_avg_pool1d(input: torch.FloatTensor, valid_mask: torch.BoolTensor) -> torch.FloatTensor:
    """
    Apply masked global average pooling along time dimension.

    Parameters
    ----------
    input : Tensor[float32]
        Input.
        Shape is (batch, channel, time).
    valid_mask : Tensor[bool]
        Mask of valid times.
        It takes 1 for valid and 0 for invalid.
        Shape is (batch, time).

    output : Tensor[float32]
        Averaged output.
        Shape is (batch, channel).
    """

    return (valid_mask.unsqueeze(1) * input).sum(dim=2) / valid_mask.sum(dim=1, keepdim=True)

def masked_global_max_pool1d(input: torch.FloatTensor, valid_mask: torch.BoolTensor) -> torch.FloatTensor:
    """
    Apply masked global max pooling along time dimension.

    Parameters
    ----------
    input : Tensor[float32]
        Input.
        Shape is (batch, channel, time).
    valid_mask : Tensor[bool]
        Mask of valid times.
        It takes True for valid and False for invalid.
        Shape is (batch, time).

    output : Tensor[float32]
        Maximum output.
        Shape is (batch, channel).
    """

    return input.masked_fill(~valid_mask.unsqueeze(1), -torch.inf).max(dim=2).values

def masked_global_softmax_pool1d(input: torch.FloatTensor, valid_mask: torch.BoolTensor, temp: float = 1) -> torch.FloatTensor:
    """
    Apply masked global softmax pooling along time dimension.

    Parameters
    ----------
    input : Tensor[float32]
        Input.
        Shape is (batch, channel, time).
    valid_mask : Tensor[bool]
        Mask of valid times.
        It takes True for valid and False for invalid.
        Shape is (batch, time).
    temp : float
        Temperature parameter.
        Standard if 1.

    Returns
    -------
    output : Tensor[float32]
        Soft maximum output.
        Shape is (batch, channel).
    """

    return ((input / temp).masked_fill(~valid_mask.unsqueeze(1), -torch.inf).softmax(2) * input).sum(dim=2)
