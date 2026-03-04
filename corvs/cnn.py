from typing import Callable
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

class SeparableDualCNN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, fn: int, ks_s: int, act_func: Callable[[torch.FloatTensor], torch.FloatTensor] = F.silu) -> None:
        super().__init__()

        if out_ch % 2 != 0:
            raise ValueError("number of out channels must be even")

        self.act_func = act_func
        half_out_ch = out_ch // 2

        self.conv_1_d = nn.Conv1d(1, fn, ks_s, bias=False)
        self.conv_1_p = nn.Conv1d(in_ch * fn, half_out_ch, 1, bias=False)
        self.bn_1 = MaskedBatchNorm1d(half_out_ch)
        self.conv_2_s_d = nn.Conv1d(1, fn, ks_s, bias=False)
        self.conv_2_s_p = nn.Conv1d(half_out_ch * fn, half_out_ch, 1, bias=False)
        self.bn_2_s = MaskedBatchNorm1d(half_out_ch)
        self.conv_3_s_d = nn.Conv1d(1, fn, ks_s, bias=False)
        self.conv_3_s_p = nn.Conv1d(half_out_ch * fn, half_out_ch, 1, bias=False)
        self.bn_3_s = MaskedBatchNorm1d(half_out_ch)
        self.conv_2_l_d = nn.Conv1d(1, fn, 2 * ks_s - 1, bias=False)
        self.conv_2_l_p = nn.Conv1d(half_out_ch * fn, half_out_ch, 1, bias=False)
        self.bn_2_l = MaskedBatchNorm1d(half_out_ch)
        self.conv_3_l_d = nn.Conv1d(1, fn, 2 * ks_s - 1, bias=False)
        self.conv_3_l_p = nn.Conv1d(half_out_ch * fn, half_out_ch, 1, bias=False)
        self.bn_3_l = MaskedBatchNorm1d(half_out_ch)

    def forward(self, input: torch.FloatTensor, valid_mask: torch.BoolTensor | torch.FloatTensor | torch.IntTensor) -> torch.FloatTensor:    # (batch, channel, time), (batch, time) -> (batch, channel, time)
        batch_size = len(input)

        hidden: torch.FloatTensor = self.conv_1_d(input.view(batch_size * input.shape[1], 1, input.shape[2]))    # (batch, in_ch, time) -> (batch * in_ch, 1, time) -> (batch * in_ch, fn, time)
        hidden = self.conv_1_p(hidden.view(batch_size, -1, hidden.shape[2]))    # (batch * in_ch, fn, time) -> (batch, in_ch * fn, time) -> (batch, half_out_ch, time)
        hidden = self.act_func(self.bn_1(hidden, valid_mask[:, -hidden.shape[2]:]))

        hidden_s: torch.FloatTensor = self.conv_2_s_d(hidden.view(batch_size * hidden.shape[1], 1, hidden.shape[2]))
        hidden_s = self.conv_2_s_p(hidden_s.view(batch_size, -1, hidden_s.shape[2]))
        hidden_s = self.act_func(self.bn_2_s(hidden_s, valid_mask[:, -hidden_s.shape[2]:]))
        hidden_s = self.conv_3_s_d(hidden_s.view(batch_size * hidden_s.shape[1], 1, hidden_s.shape[2]))
        hidden_s = self.conv_3_s_p(hidden_s.view(batch_size, -1, hidden_s.shape[2]))
        hidden_s = self.act_func(self.bn_3_s(hidden_s, valid_mask[:, -hidden_s.shape[2]:]))

        hidden_l: torch.FloatTensor = self.conv_2_l_d(hidden.view(batch_size * hidden.shape[1], 1, hidden.shape[2]))
        hidden_l = self.conv_2_l_p(hidden_l.view(batch_size, -1, hidden_l.shape[2]))
        hidden_l = self.act_func(self.bn_2_l(hidden_l, valid_mask[:, -hidden_l.shape[2]:]))
        hidden_l = self.conv_3_l_d(hidden_l.view(batch_size * hidden_l.shape[1], 1, hidden_l.shape[2]))
        hidden_l = self.conv_3_l_p(hidden_l.view(batch_size, -1, hidden_l.shape[2]))
        hidden_l = self.act_func(self.bn_3_l(hidden_l, valid_mask[:, -hidden_l.shape[2]:]))

        head_len = (hidden_s.shape[2] - hidden_l.shape[2]) // 2
        tail_len = hidden_s.shape[2] - hidden_l.shape[2] - head_len
        output = torch.cat((hidden_s[:, :, head_len:-tail_len], hidden_l), dim=1)

        return output
