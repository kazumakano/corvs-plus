import math
from typing import Callable, Literal, Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.types import Device
from corvs.normalization import MaskedBatchNorm1d


class DualCNN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ks_s: int, act: nn.Module | Callable[[torch.FloatTensor], torch.FloatTensor] = F.silu) -> None:
        if out_ch % 2 != 0:
            raise ValueError(f"number of out channels must be even, but got {out_ch}")

        super().__init__()

        self.act = act
        half_out_ch = out_ch // 2

        self.conv_1   = nn.Conv1d(in_ch, half_out_ch, ks_s, bias=False)
        self.bn_1     = MaskedBatchNorm1d(half_out_ch)
        self.conv_2_s = nn.Conv1d(half_out_ch, half_out_ch, ks_s, bias=False)
        self.bn_2_s   = MaskedBatchNorm1d(half_out_ch)
        self.conv_3_s = nn.Conv1d(half_out_ch, half_out_ch, ks_s, bias=False)
        self.bn_3_s   = MaskedBatchNorm1d(half_out_ch)
        self.conv_2_l = nn.Conv1d(half_out_ch, half_out_ch, 2 * ks_s - 1, bias=False)
        self.bn_2_l   = MaskedBatchNorm1d(half_out_ch)
        self.conv_3_l = nn.Conv1d(half_out_ch, half_out_ch, 2 * ks_s - 1, bias=False)
        self.bn_3_l   = MaskedBatchNorm1d(half_out_ch)

    def forward(self, input: torch.FloatTensor, valid_mask: torch.BoolTensor | torch.FloatTensor | torch.IntTensor) -> torch.FloatTensor:  # (batch, ch, time), (batch, time) -> (batch, ch, time)
        hidden:   torch.FloatTensor
        hidden_s: torch.FloatTensor
        hidden_l: torch.FloatTensor

        hidden   = self.conv_1(input)
        hidden   = self.act(self.bn_1(hidden, valid_mask[:, -hidden.shape[2]:]))
        hidden_s = self.conv_2_s(hidden)
        hidden_s = self.act(self.bn_2_s(hidden_s, valid_mask[:, -hidden_s.shape[2]:]))
        hidden_s = self.conv_3_s(hidden_s)
        hidden_s = self.act(self.bn_3_s(hidden_s, valid_mask[:, -hidden_s.shape[2]:]))
        hidden_l = self.conv_2_l(hidden)
        hidden_l = self.act(self.bn_2_l(hidden_l, valid_mask[:, -hidden_l.shape[2]:]))
        hidden_l = self.conv_3_l(hidden_l)
        hidden_l = self.act(self.bn_3_l(hidden_l, valid_mask[:, -hidden_l.shape[2]:]))

        head_len = math.floor((hidden_s.shape[2] - hidden_l.shape[2]) / 2)
        tail_len = math.ceil((hidden_s.shape[2] - hidden_l.shape[2]) / 2)
        output = torch.cat((hidden_s[:, :, head_len:-tail_len], hidden_l), dim=1)

        return output

    @property
    def recept_field(self) -> int:
        return self.conv_1.kernel_size[0] + self.conv_2_l.kernel_size[0] + self.conv_3_l.kernel_size[0] - 2

class SeparableConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple[int],
            expansion: int = 1,
            stride: int | tuple[int] = 1,
            padding: int | tuple[int] | Literal["valid", "same"] = 0,
            dilation: int | tuple[int] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
            device: Device = None,
            dtype: Optional[torch.dtype] = None
        ) -> None:

        super().__init__()

        self.d = nn.Conv1d(
            in_channels,
            expansion * in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        self.p = nn.Conv1d(
            expansion * in_channels,
            out_channels,
            1,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype
        )

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:  # (..., ch, time) -> (..., ch, time)
        hidden = self.d(input)
        output = self.p(hidden)
        return output

    @property
    def kernel_size(self) -> tuple[int]:
        return self.d.kernel_size

    @property
    def stride(self) -> tuple[int]:
        return self.d.stride

    @property
    def padding(self) -> tuple[int] | Literal["valid", "same"]:
        return self.d.padding

    @property
    def dilation(self) -> tuple[int]:
        return self.d.dilation

class SeparableDualCNN(DualCNN):
    def __init__(self, in_ch: int, out_ch: int, ks_s: int, ex: int = 1, act: nn.Module | Callable[[torch.FloatTensor], torch.FloatTensor] = F.silu) -> None:
        super().__init__(in_ch, out_ch, ks_s, act)

        half_out_ch = out_ch // 2

        self.conv_2_s = SeparableConv1d(half_out_ch, half_out_ch, ks_s, ex, bias=False)
        self.conv_3_s = SeparableConv1d(half_out_ch, half_out_ch, ks_s, ex, bias=False)
        self.conv_2_l = SeparableConv1d(half_out_ch, half_out_ch, 2 * ks_s - 1, ex, bias=False)
        self.conv_3_l = SeparableConv1d(half_out_ch, half_out_ch, 2 * ks_s - 1, ex, bias=False)
