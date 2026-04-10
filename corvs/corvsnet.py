import warnings
from argparse import Namespace
from typing import Any, Callable, Literal, Optional, overload
import einops
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from corvs.base import BaseDataset, BaseFitDataset, BaseFitModule, BaseModule, BasePredModule, Modality, SensorMet, TrajMet
from corvs.cnn import DualCNN, SeparableDualCNN
from corvs.embedding import create_sin_pos_emb
from corvs.normalization import MaskedBatchNorm1d
from corvs.pooling import MaskedGlobalAttnPool1d, masked_global_avg_pool1d, masked_global_max_pool1d, masked_global_soft_pool1d
from corvs.transformer import RoFormerEncoderLayer, TransformerEncoderLayer


@overload
def str_to_mod(act: Literal["relu", "leaky_relu", "gelu", "silu"], func: Literal[False], **kwargs: Any) -> nn.ReLU | nn.LeakyReLU | nn.GELU | nn.SiLU:
    ...

@overload
def str_to_mod(act: Literal["relu", "leaky_relu", "gelu", "silu"], func: Literal[True]) -> Callable[[torch.FloatTensor], torch.FloatTensor]:
    ...

@overload
def str_to_mod(act: Literal["relu", "leaky_relu", "gelu", "silu"], **kwargs: Any) -> nn.ReLU | nn.LeakyReLU | nn.GELU | nn.SiLU:
    ...

def str_to_mod(act: Literal["relu", "leaky_relu", "gelu", "silu"], func: bool = False, **kwargs: Any) -> nn.ReLU | nn.LeakyReLU | nn.GELU | nn.SiLU | Callable[[torch.FloatTensor], torch.FloatTensor]:
    if func and len(kwargs) > 0:
        warnings.warn(UserWarning("keyword arguments are ignored when functional"))

    match act:
        case "relu":
            return F.relu if func else nn.ReLU(**kwargs)
        case "leaky_relu":
            return F.leaky_relu if func else nn.LeakyReLU(**kwargs)
        case "gelu":
            return F.gelu if func else nn.GELU(**kwargs)
        case "silu":
            return F.silu if func else nn.SiLU(**kwargs)
        case _:
            raise ValueError(f"unknown activation function {act} was specified")

class CorVSNet(BaseModule):
    def __init__(self, hparams: dict[str, Any] | Namespace | DictConfig, ds_cls: type[BaseDataset]) -> None:
        super().__init__(hparams, ds_cls)

        if self.hparams["time_agg"] == "cls_tok" and not self.hparams["cls_tok"]:
            raise ValueError("time aggregation with CLS token needs to enable CLS token")

        self.bn = MaskedBatchNorm1d(len(self.in_mets), affine=False)
        cnn_act = str_to_mod(self.hparams["cnn_act"], True)
        if self.hparams["cnn_sep"]:
            self.cnn = SeparableDualCNN(len(self.in_mets), self.hparams["xfmr_d_model"], self.hparams["cnn_ks_s"], self.hparams["cnn_fn"], cnn_act)
        else:
            self.cnn = DualCNN(len(self.in_mets), self.hparams["xfmr_d_model"], self.hparams["cnn_ks_s"], cnn_act)

        if self.hparams["min_in_len"] < self.cnn.recept_field:
            raise ValueError("input cannot be shorter than receptive field of CNN backbone")

        xfmr_time_len = self.hparams["win_len"] - self.cnn.recept_field + 1
        if self.hparams["cls_tok"]:
            self.cls_tok = nn.Parameter(data=torch.empty(1, 1, self.hparams["xfmr_d_model"], dtype=torch.float32))
            xfmr_time_len += 1

        match self.hparams["xfmr_pos_enc"]:
            case "sinusoidal":
                self.register_buffer("pos_emb", create_sin_pos_emb(self.hparams["xfmr_d_model"], xfmr_time_len).unsqueeze(1), persistent=False)
                xfmr_layer = TransformerEncoderLayer(
                    self.hparams["xfmr_d_model"],
                    self.hparams["xfmr_nhead"],
                    self.hparams["xfmr_d_ff"],
                    self.hparams["xfmr_dr"],
                    self.hparams["xfmr_act"],
                    self.hparams["xfmr_norm"],
                    norm_first=True
                )
            case "learnable":
                self.pos_emb = nn.Parameter(data=torch.empty(xfmr_time_len, 1, self.hparams["xfmr_d_model"], dtype=torch.float32))
                xfmr_layer = TransformerEncoderLayer(
                    self.hparams["xfmr_d_model"],
                    self.hparams["xfmr_nhead"],
                    self.hparams["xfmr_d_ff"],
                    self.hparams["xfmr_dr"],
                    self.hparams["xfmr_act"],
                    self.hparams["xfmr_norm"],
                    norm_first=True
                )
            case "rope":
                xfmr_layer = RoFormerEncoderLayer(
                    self.hparams["xfmr_d_model"],
                    self.hparams["xfmr_nhead"],
                    self.hparams["xfmr_d_ff"],
                    xfmr_time_len,
                    self.hparams["xfmr_dr"],
                    self.hparams["xfmr_act"],
                    self.hparams["xfmr_norm"],
                    norm_first=True
                )
            case _:
                raise ValueError(f"unknown positional encoding {self.hparams['xfmr_pos_enc']} was specified")

        match self.hparams["xfmr_norm"]:
            case "layer":
                xfmr_norm = nn.LayerNorm(self.hparams["xfmr_d_model"])
            case "rms":
                xfmr_norm = nn.RMSNorm(self.hparams["xfmr_d_model"])
            case _:
                raise ValueError(f"unknown normalization {self.hparams['xfmr_norm']} was specified")

        self.xfmr = nn.TransformerEncoder(xfmr_layer, self.hparams["xfmr_n_layers"], norm=xfmr_norm, enable_nested_tensor=False)

        if self.hparams["time_agg"] == "attn_pool":
            self.pool = MaskedGlobalAttnPool1d(self.hparams["xfmr_d_model"], 1)

        self.mlp = nn.Sequential(
            nn.Linear(self.hparams["xfmr_d_model"], self.hparams["xfmr_d_model"] // 4),
            str_to_mod(self.hparams["mlp_act"]),
            nn.Dropout(p=self.hparams["mlp_dr"]),
            nn.Linear(self.hparams["xfmr_d_model"] // 4, 1)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.RMSNorm)):
                m.reset_parameters()

            elif isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

            elif isinstance(m, nn.MultiheadAttention):
                m._reset_parameters()

            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

        if self.hparams["cls_tok"]:
            init.xavier_uniform_(self.cls_tok)
        if self.hparams["xfmr_pos_enc"] == "learnable":
            init.xavier_uniform_(self.pos_emb)

    def forward(
            self,
            traj_input:   torch.FloatTensor,                  # (batch, time, ch)
            sensor_input: torch.FloatTensor,                  # (batch, time, ch)
            valid_mask:   Optional[torch.BoolTensor] = None,  # (batch, time)
            visible_mask: Optional[torch.BoolTensor] = None   # (batch, time)
        ) -> torch.FloatTensor:  # (batch, 1)

        if valid_mask is None:
            valid_mask = torch.ones(traj_input.shape[:2], dtype=torch.bool, device=traj_input.device)

        hidden: torch.FloatTensor
        hidden = self.bn(torch.cat((traj_input, sensor_input), dim=2).transpose(1, 2), valid_mask)
        hidden = self.cnn(hidden, valid_mask)

        hidden = einops.rearrange(hidden, "b d t -> t b d")
        if self.hparams["cls_tok"]:
            cls_tok = einops.repeat(self.cls_tok, "1 1 d -> 1 b d", b=hidden.shape[1])
            hidden = torch.cat((cls_tok, hidden))
        if self.hparams["xfmr_pos_enc"] in ("sinusoidal", "learnable"):
            pos_emb = einops.repeat(self.pos_emb, "t 1 d -> t b d", b=hidden.shape[1])
            hidden += pos_emb

        valid_mask = valid_mask[:, -hidden.shape[0]:]
        if visible_mask is not None:
            visible_mask = ~self.mask_contract(~visible_mask, hidden.shape[0])
            valid_mask = valid_mask & visible_mask

        hidden = self.xfmr(hidden, src_key_padding_mask=~valid_mask)

        hidden = einops.rearrange(hidden, "t b d -> b d t")
        match self.hparams["time_agg"]:
            case "avg_pool":
                hidden = masked_global_avg_pool1d(hidden, valid_mask)
            case "max_pool":
                hidden = masked_global_max_pool1d(hidden, valid_mask)
            case "soft_pool":
                hidden = masked_global_soft_pool1d(hidden, valid_mask)
            case "attn_pool":
                hidden = self.pool(hidden, valid_mask)
            case "cls_tok":
                hidden = hidden[:, :, 0]
            case _:
                raise ValueError(f"unknown time aggregation method {self.hparams['time_agg']} was specified")

        output = self.mlp(hidden)

        return output

    @staticmethod
    def mask_contract(mask: torch.BoolTensor, tgt_len: int) -> torch.BoolTensor:    # (batch, time) -> (batch, time)
        diff = mask.diff(prepend=torch.zeros(mask.shape[0], 1, dtype=torch.bool, device=mask.device))
        diff_cnt = diff.count_nonzero(dim=1)
        if not (diff_cnt < 3).all().item():
            raise ValueError("non-contiguous masks cannnot be contracted")

        time_idx = torch.arange(mask.shape[1], dtype=torch.int32, device=mask.device)  # (time, )
        min_idx = torch.where(mask, time_idx, torch.inf).min(dim=1).values             # (batch, )
        max_idx = torch.where(mask, time_idx, -torch.inf).max(dim=1).values            # (batch, )
        mask = (min_idx.unsqueeze(1) - 0.5 < time_idx[:tgt_len].unsqueeze(0)) & (time_idx[-tgt_len:].unsqueeze(0) < max_idx.unsqueeze(1) + 0.5)

        return mask

class CorVSNetFitter(CorVSNet, BaseFitModule):
    def __init__(self, hparams: dict[str, Any] | Namespace | DictConfig, ds_cls: type[BaseFitDataset]) -> None:
        super().__init__(hparams, ds_cls)

        self.example_input_array = (
            torch.empty(1, self.hparams["win_len"], len(self.traj_mets), dtype=torch.float32),
            torch.empty(1, self.hparams["win_len"], len(self.sensor_mets), dtype=torch.float32),
            torch.ones(1, self.hparams["win_len"], dtype=torch.bool),
            torch.ones(1, self.hparams["win_len"], dtype=torch.bool)
        )

    def training_step(self, batch: list[torch.FloatTensor | torch.BoolTensor], _: int) -> torch.FloatTensor:
        traj_feat = batch[self.modalities.index(Modality.TRAJ_FEAT)]
        sensor_feat = batch[self.modalities.index(Modality.SENSOR_FEAT)]
        valid_mask = batch[self.modalities.index(Modality.VALID_MASK)]
        visible_mask = batch[self.modalities.index(Modality.VISIBLE_MASK)]
        logit = self(traj_feat, sensor_feat, valid_mask, visible_mask)

        label = batch[self.modalities.index(Modality.LABEL)]
        loss = self.train_crit(logit, label)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch: list[torch.FloatTensor | torch.BoolTensor], _: int) -> torch.FloatTensor:
        traj_feat = batch[self.modalities.index(Modality.TRAJ_FEAT)]
        sensor_feat = batch[self.modalities.index(Modality.SENSOR_FEAT)]
        valid_mask = batch[self.modalities.index(Modality.VALID_MASK)]
        visible_mask = batch[self.modalities.index(Modality.VISIBLE_MASK)]
        logit = self(traj_feat, sensor_feat, valid_mask, visible_mask)

        label = batch[self.modalities.index(Modality.LABEL)]
        loss = self.val_crit(logit, label)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return loss

class CorVSNetPredictor(CorVSNet, BasePredModule):
    def forward(self, traj_input: torch.FloatTensor, sensor_input: torch.FloatTensor, valid_mask: Optional[torch.BoolTensor] = None) -> tuple[torch.FloatTensor, torch.FloatTensor]:  # (batch, time, ch), (batch, time, ch), (batch, time) -> (batch, 1), (batch, 1)
        prob = F.sigmoid(super().forward(traj_input, sensor_input, valid_mask))
        spd = traj_input[:, :, self.traj_mets.index(TrajMet.SPD)]
        linacc = sensor_input[:, :, self.sensor_mets.index(SensorMet.LINACC_NORM)]
        rel = self.rel_estim(spd, linacc, valid_mask)

        return prob, rel

    def rel_estim(self, spd: torch.FloatTensor, linacc: torch.FloatTensor, valid_mask: torch.BoolTensor | torch.FloatTensor | torch.IntTensor, eps: float = 1e-5) -> torch.FloatTensor:  # (batch, time), (batch, time), (batch, time) -> (batch, 1)
        cnt = valid_mask.count_nonzero(dim=1)
        spd_mean = (valid_mask * spd).sum(dim=1) / cnt
        spd_var = (valid_mask * (spd - spd_mean.unsqueeze(1)) ** 2).sum(dim=1) / cnt
        linacc_mean = (valid_mask * linacc).sum(dim=1) / cnt
        linacc_var = (valid_mask * (linacc - linacc_mean.unsqueeze(1)) ** 2).sum(dim=1) / cnt

        spd_run_var = self.bn.running_var[self.in_mets.index(TrajMet.SPD)]
        linacc_run_var = self.bn.running_var[self.in_mets.index(SensorMet.LINACC_NORM)]
        rel = 1 / (1 + torch.min(spd_run_var / (spd_var + eps), linacc_run_var / (linacc_var + eps))).unsqueeze(1)

        return rel

    def predict_step(self, batch: list[torch.DoubleTensor | torch.FloatTensor | torch.BoolTensor], _: int) -> tuple[torch.DoubleTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        traj_feat = batch[self.modalities.index(Modality.TRAJ_FEAT)]
        sensor_feat = batch[self.modalities.index(Modality.SENSOR_FEAT)]
        valid_mask = batch[self.modalities.index(Modality.VALID_MASK)]
        prob, rel = self(traj_feat, sensor_feat, valid_mask)

        time = batch[self.modalities.index(Modality.TIME)]
        label = batch[self.modalities.index(Modality.LABEL)]

        return time, prob, rel, label
