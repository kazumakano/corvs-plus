import math
from typing import Callable, Literal
import torch
from torch import nn
from torch.nn import functional as F
from corvs.normalization import MaskedBatchNorm1d


class DualCNN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ks_s: int, act: Callable[[torch.FloatTensor], torch.FloatTensor] = F.silu) -> None:
        super().__init__()

        if out_ch % 2 != 0:
            raise ValueError("number of out channels must be even")

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

    def forward(self, input: torch.FloatTensor, valid_mask: torch.BoolTensor | torch.FloatTensor | torch.IntTensor) -> torch.FloatTensor:    # (batch, channel, time), (batch, time) -> (batch, channel, time)
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
            in_ch: int,
            out_ch: int,
            ks: int | tuple[int],
            fn: int = 1,
            st: int | tuple[int] = 1,
            pad: int | tuple[int] | Literal["same", "valid"] = 0,
            dil: int | tuple[int] = 1,
            grps: int = 1,
            bias: bool = True,
            pad_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros"
        ) -> None:
        super().__init__()

        self.d = nn.Conv1d(in_ch, fn * in_ch, ks, stride=st, padding=pad, dilation=dil, groups=in_ch, bias=False, padding_mode=pad_mode)
        self.p = nn.Conv1d(fn * in_ch, out_ch, 1, groups=grps, bias=bias)

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:    # (*, channel, time) -> (*, channel, time)
        hidden = self.d(input)
        output = self.p(hidden)
        return output

    @property
    def kernel_size(self) -> tuple[int]:
        return self.d.kernel_size

class SeparableDualCNN(DualCNN):
    def __init__(self, in_ch: int, out_ch: int, ks_s: int, fn: int = 1, act: Callable[[torch.FloatTensor], torch.FloatTensor] = F.silu) -> None:
        super().__init__(in_ch, out_ch, ks_s, act)

        half_out_ch = out_ch // 2

        self.conv_2_s = SeparableConv1d(half_out_ch, half_out_ch, ks_s, fn, bias=False)
        self.conv_3_s = SeparableConv1d(half_out_ch, half_out_ch, ks_s, fn, bias=False)
        self.conv_2_l = SeparableConv1d(half_out_ch, half_out_ch, 2 * ks_s - 1, fn, bias=False)
        self.conv_3_l = SeparableConv1d(half_out_ch, half_out_ch, 2 * ks_s - 1, fn, bias=False)
