from typing import Callable, Literal
import torch
from torch import nn
from torch.nn import functional as F
from corvs.normalization import MaskedBatchNorm1d


class DualCNN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ks_s: int, act_func: Callable[[torch.FloatTensor], torch.FloatTensor] = F.silu) -> None:
        super().__init__()

        if out_ch % 2 != 0:
            raise ValueError("number of out channels must be even")

        self.act_func = act_func
        half_out_ch = out_ch // 2

        self.conv_1 = nn.Conv1d(in_ch, half_out_ch, ks_s, bias=False)
        self.bn_1 = MaskedBatchNorm1d(half_out_ch)
        self.conv_2_s = nn.Conv1d(half_out_ch, half_out_ch, ks_s, bias=False)
        self.bn_2_s = MaskedBatchNorm1d(half_out_ch)
        self.conv_3_s = nn.Conv1d(half_out_ch, half_out_ch, ks_s, bias=False)
        self.bn_3_s = MaskedBatchNorm1d(half_out_ch)
        self.conv_2_l = nn.Conv1d(half_out_ch, half_out_ch, 2 * ks_s - 1, bias=False)
        self.bn_2_l = MaskedBatchNorm1d(half_out_ch)
        self.conv_3_l = nn.Conv1d(half_out_ch, half_out_ch, 2 * ks_s - 1, bias=False)
        self.bn_3_l = MaskedBatchNorm1d(half_out_ch)

    def forward(self, input: torch.FloatTensor, valid_mask: torch.BoolTensor | torch.FloatTensor | torch.IntTensor) -> torch.FloatTensor:    # (batch, channel, time), (batch, time) -> (batch, channel, time)
        hidden: torch.FloatTensor = self.conv_1(input)
        hidden = self.act_func(self.bn_1(hidden, valid_mask[:, -hidden.shape[2]:]))
        hidden_s: torch.FloatTensor = self.conv_2_s(hidden)
        hidden_s = self.act_func(self.bn_2_s(hidden_s, valid_mask[:, -hidden_s.shape[2]:]))
        hidden_s = self.conv_3_s(hidden_s)
        hidden_s = self.act_func(self.bn_3_s(hidden_s, valid_mask[:, -hidden_s.shape[2]:]))
        hidden_l: torch.FloatTensor = self.conv_2_l(hidden)
        hidden_l = self.act_func(self.bn_2_l(hidden_l, valid_mask[:, -hidden_l.shape[2]:]))
        hidden_l = self.conv_3_l(hidden_l)
        hidden_l = self.act_func(self.bn_3_l(hidden_l, valid_mask[:, -hidden_l.shape[2]:]))

        head_len = (hidden_s.shape[2] - hidden_l.shape[2]) // 2
        tail_len = hidden_s.shape[2] - hidden_l.shape[2] - head_len
        output = torch.cat((hidden_s[:, :, head_len:-tail_len], hidden_l), dim=1)

        return output

class SeparableConv1d(nn.Module):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            ks: int,
            fn: int = 1,
            st: int = 1,
            pad: int = 0,
            dil: int = 1,
            grps: int = 1,
            bias: bool = True,
            pad_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros"
        ) -> None:
        super().__init__()

        self.conv_d = nn.Conv1d(in_ch, fn * in_ch, ks, stride=st, padding=pad, dilation=dil, groups=in_ch, bias=False, padding_mode=pad_mode)
        self.conv_p = nn.Conv1d(fn * in_ch, out_ch, 1, groups=grps, bias=bias)

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:    # (*, channel, time) -> (*, channel, time)
        hidden = self.conv_d(input)
        output = self.conv_p(hidden)
        return output

class SeparableDualCNN(DualCNN):
    def __init__(self, in_ch: int, out_ch: int, ks_s: int, fn: int = 1, act_func: Callable[[torch.FloatTensor], torch.FloatTensor] = F.silu) -> None:
        super().__init__(in_ch, out_ch, ks_s, act_func)

        half_out_ch = out_ch // 2

        self.conv_1 = SeparableConv1d(in_ch, half_out_ch, ks_s, fn, bias=False)
        self.conv_2_s = SeparableConv1d(half_out_ch, half_out_ch, ks_s, fn, bias=False)
        self.conv_3_s = SeparableConv1d(half_out_ch, half_out_ch, ks_s, fn, bias=False)
        self.conv_2_l = SeparableConv1d(half_out_ch, half_out_ch, 2 * ks_s - 1, fn, bias=False)
        self.conv_3_l = SeparableConv1d(half_out_ch, half_out_ch, 2 * ks_s - 1, fn, bias=False)
