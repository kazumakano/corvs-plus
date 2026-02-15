from typing import Any, Optional
import torch
from lightning import pytorch as L
from torch import nn
from torch.nn import functional as F
from .pooling import MaskedGlobalAttnPool1d
from .transformer import RoFormerEncoderLayer, create_sin_pos_embed


class MaskedBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.FloatTensor, valid_mask: torch.BoolTensor) -> torch.FloatTensor:    # (batch, channel, time), (batch, time) -> (batch, channel, time)
        """
        Modified from `BatchNorm1d.forward()`.
        This method supports masking.

        Parameters
        ----------
        input : Tensor[float32]
            Input.
            Shape is (batch, channel, time).
        valid_mask : Tensor[bool]
            Mask of valid times.
            It takes 'True' for valid and 'False' for invalid.
            Shape is (batch, time).

        Returns
        -------
        output : Tensor[float32]
            Normalized output.
            Shape is (batch, channel, time).
        """

        self._check_input_dim(input)
        self._check_mask(valid_mask)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return self.batch_norm(input, valid_mask, self.running_mean if not self.training or self.track_running_stats else None, self.running_var if not self.training or self.track_running_stats else None, self.weight, self.bias, bn_training, exponential_average_factor, self.eps)

    @staticmethod
    def batch_norm(input: torch.Tensor, valid_mask: torch.Tensor, running_mean: torch.Tensor | None, running_var: torch.Tensor | None, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None, training: bool = False, momentum: float = 0.1, eps: float = 1e-5) -> torch.Tensor:
        """
        Modified from `batch_norm()`.
        """

        if training:
            F._verify_batch_size(input.size())

        if eps <= 0.0:
            raise ValueError(f"batch_norm eps must be positive, but got {eps}")

        if training:
            cnt = valid_mask.count_nonzero()
            mean = (valid_mask.unsqueeze(1) * input).sum(dim=(0, 2)) / cnt    # (channel, )    
            var = (valid_mask.unsqueeze(1) * (input - mean.unsqueeze(0).unsqueeze(2)) ** 2).sum(dim=(0, 2)) / cnt    # (channel, )
            if running_mean is not None and running_var is not None:
                running_mean.copy_((1 - momentum) * running_mean + momentum * mean)
                running_var.copy_((1 - momentum) * running_var + momentum * cnt / (cnt - 1) * var)
        else:
            mean = running_mean
            var = running_var

        assert mean is not None and var is not None
        output = (input - mean.unsqueeze(0).unsqueeze(2)) / (var + eps).sqrt().unsqueeze(0).unsqueeze(2)
        if weight is not None and bias is not None:
            output = weight.unsqueeze(0).unsqueeze(2) * output + bias.unsqueeze(0).unsqueeze(2)

        return output

    def _check_input_dim(self, input: torch.Tensor) -> None:
        if input.dim() != 3:
            raise ValueError(f"expected 3D input (got {input.dim()}D input)")

    def _check_mask(self, mask: torch.Tensor) -> None:
        if mask.dim() != 2:
            raise ValueError(f"expected 2D mask (got {mask.dim()}D mask)")
        if mask.count_nonzero() == 0:
            raise ValueError("expected non-zero mask")

class DualCNN(nn.Module):
    def __init__(self, ch: int, ks_s: int) -> None:
        super().__init__()

        self.conv_1 = nn.Conv1d(9, ch, ks_s, bias=False)
        self.bn_1 = MaskedBatchNorm1d(ch)
        self.conv_2_s = nn.Conv1d(ch, ch, ks_s, bias=False)
        self.bn_2_s = MaskedBatchNorm1d(ch)
        self.conv_3_s = nn.Conv1d(ch, ch, ks_s, bias=False)
        self.bn_3_s = MaskedBatchNorm1d(ch)
        self.conv_2_l = nn.Conv1d(ch, ch, 2 * ks_s - 1, bias=False)
        self.bn_2_l = MaskedBatchNorm1d(ch)
        self.conv_3_l = nn.Conv1d(ch, ch, 2 * ks_s - 1, bias=False)
        self.bn_3_l = MaskedBatchNorm1d(ch)

    def forward(self, input: torch.FloatTensor, valid_mask: torch.BoolTensor) -> torch.FloatTensor:    # (batch, channel, time), (batch, time) -> (batch, channel, time)
        hidden: torch.Tensor = self.conv_1(input)
        hidden = F.silu(self.bn_1(hidden, valid_mask[:, -hidden.shape[2]:]))
        hidden_s: torch.Tensor = self.conv_2_s(hidden)
        hidden_s = F.silu(self.bn_2_s(hidden_s, valid_mask[:, -hidden_s.shape[2]:]))
        hidden_s = self.conv_3_s(hidden_s)
        hidden_s = F.silu(self.bn_3_s(hidden_s, valid_mask[:, -hidden_s.shape[2]:]))
        hidden_l: torch.Tensor = self.conv_2_l(hidden)
        hidden_l = F.silu(self.bn_2_l(hidden_l, valid_mask[:, -hidden_l.shape[2]:]))
        hidden_l = self.conv_3_l(hidden_l)
        hidden_l = F.silu(self.bn_3_l(hidden_l, valid_mask[:, -hidden_l.shape[2]:]))

        head_len = (hidden_s.shape[2] - hidden_l.shape[2]) // 2
        tail_len = hidden_s.shape[2] - hidden_l.shape[2] - head_len
        output = torch.cat((hidden_s[:, :, head_len:-tail_len], hidden_l), dim=1)

        return output

class CorVSNet(L.LightningModule):
    def __init__(self, hparams: dict[str, Any]) -> None:
        if hparams["min_input_len"] < 5 * hparams["cnn_ks_s"] - 4:
            raise ValueError("input cannot be shorter than receptive field of CNN backbone")
        if hparams["xformer_d_model"] % 2 != 0:
            raise ValueError("dimension of Transformer encoder must be even")
        if hparams["time_agg"] == "cls_token" and not hparams["use_cls_token"]:
            raise ValueError("time aggregation with CLS token needs to enable CLS token")

        super().__init__()
        self.save_hyperparameters(hparams)

        self.bn = MaskedBatchNorm1d(9, affine=False)
        self.cnn = DualCNN(hparams["xformer_d_model"] // 2, hparams["cnn_ks_s"])

        xformer_time_len = hparams["win_len"] - 5 * hparams["cnn_ks_s"] + 5
        if hparams["use_cls_token"]:
            self.cls_token = nn.Parameter(data=torch.empty(1, 1, hparams["xformer_d_model"], dtype=torch.float32))
            xformer_time_len += 1
        match hparams["xformer_pos_enc"]:
            case "learnable":
                self.pos_embed = nn.Parameter(data=torch.randn(xformer_time_len, 1, hparams["xformer_d_model"], dtype=torch.float32))
                xformer_layer = nn.TransformerEncoderLayer(hparams["xformer_d_model"], hparams["xformer_nhead"], dim_feedforward=hparams["xformer_d_ff"], activation=F.gelu, norm_first=True)
            case "sinusoidal":
                self.register_buffer("pos_embed", create_sin_pos_embed(hparams["xformer_d_model"], xformer_time_len).unsqueeze(1))
                xformer_layer = nn.TransformerEncoderLayer(hparams["xformer_d_model"], hparams["xformer_nhead"], dim_feedforward=hparams["xformer_d_ff"], activation=F.gelu, norm_first=True)
            case "rope":
                xformer_layer = RoFormerEncoderLayer(hparams["xformer_d_model"], hparams["xformer_nhead"], xformer_time_len, dim_feedforward=hparams["xformer_d_ff"], activation=F.gelu, norm_first=True)
            case _:
                raise ValueError(f"unknown positional encoding {hparams['xformer_pos_enc']} was specified")
        self.xformer = nn.TransformerEncoder(xformer_layer, hparams["xformer_n_layers"], norm=nn.LayerNorm(hparams["xformer_d_model"]))

        if hparams["time_agg"] == "attn_pool":
            self.pool = MaskedGlobalAttnPool1d(hparams["xformer_d_model"], 1)

        self.mlp = nn.Sequential(
            nn.Linear(hparams["xformer_d_model"], hparams["xformer_d_model"] // 4),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hparams["xformer_d_model"] // 4, 1)
        )

        self.apply(self._init_params)

    @classmethod
    def _init_params(cls, mod: nn.Module) -> None:
        if isinstance(mod, cls):
            if mod.hparams["use_cls_token"]:
                nn.init.xavier_uniform_(mod.cls_token)
            if mod.hparams["xformer_pos_enc"] == "learnable":
                nn.init.xavier_uniform_(mod.pos_embed)

        elif isinstance(mod, nn.BatchNorm1d):
            mod.reset_parameters()

        elif isinstance(mod, nn.Conv1d):
            nn.init.kaiming_normal_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

        elif isinstance(mod, nn.LayerNorm):
            mod.reset_parameters()

        elif isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

        elif isinstance(mod, nn.MultiheadAttention):
            mod._reset_parameters()

    def forward(self, traj_input: torch.FloatTensor, sensor_input: torch.FloatTensor, valid_mask: Optional[torch.BoolTensor] = None) -> torch.FloatTensor:    # (batch, time, channel), (batch, time, channel), (batch, time) -> (batch, 1)
        if valid_mask is None:
            valid_mask = torch.ones(traj_input.shape[:2], dtype=torch.bool, device=traj_input.device)

        hidden: torch.Tensor = self.bn(torch.cat((traj_input, sensor_input), dim=2).transpose(1, 2), valid_mask)
        hidden = self.cnn(hidden, valid_mask)
        valid_mask = valid_mask[:, -hidden.shape[2]:]
        hidden = self.xformer(hidden.permute(2, 0, 1), src_key_padding_mask=~valid_mask)
        hidden = self.pool(hidden.permute(1, 2, 0), valid_mask)
        output = self.mlp(hidden)

        return output
