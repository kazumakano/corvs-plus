import logging
from argparse import Namespace
from typing import Annotated, Any, Optional
import einops
import torch
from annotated_types import Lt
from omegaconf import DictConfig
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from torch import nn
from torch.nn import functional as F
from corvs.enums import Modality, SensorMet, TrajMet
from corvs.model.activation import get_act
from corvs.model.base import BaseDataset, BaseModelHParams, BaseModule, BasePredModule
from corvs.model.cnn import DualCNN, SeparableDualCNN
from corvs.model.embedding import create_sin_pos_emb
from corvs.model.normalization import MaskedBatchNorm1d
from corvs.model.pooling import MaskedGlobalAttnPool1d, masked_global_avg_pool1d, masked_global_max_pool1d, masked_global_soft_pool1d
from corvs.model.transformer import RoFormerEncoderLayer, TransformerEncoderLayer


class CorVSNetHParams(BaseModelHParams):
    win_len:       PositiveInt
    min_in_len:    NonNegativeInt
    cnn_sep:       bool
    cnn_ks_s:      PositiveInt
    cnn_ex:        PositiveFloat | None
    cnn_act:       str
    cls_tok:       bool
    xfmr_pos_enc:  str
    xfmr_n_layers: PositiveInt
    xfmr_d_model:  PositiveInt
    xfmr_nhead:    PositiveInt
    xfmr_d_ff:     PositiveInt
    xfmr_dr:       Annotated[NonNegativeFloat, Lt(1)]
    xfmr_act:      str
    xfmr_norm:     str
    time_agg:      str
    mlp_dr:        Annotated[NonNegativeFloat, Lt(1)]
    mlp_act:       str

class CorVSNet(BaseModule[CorVSNetHParams]):
    traj_mets = TrajMet.SPD, TrajMet.TURN_RATE
    sensor_mets = SensorMet.LINACC_NORM, SensorMet.ACC_X, SensorMet.ACC_Y, SensorMet.ACC_Z, SensorMet.GYRO_X, SensorMet.GYRO_Y, SensorMet.GYRO_Z

    def __init__(self, hparams: dict[str, Any] | Namespace | DictConfig, ds_cls: type[BaseDataset]) -> None:
        super().__init__(hparams, ds_cls)

        if self.hparams.time_agg == "cls_tok" and not self.hparams.cls_tok:
            raise ValueError("Time aggregation by CLS token needs to enable CLS token.")

        self.bn = MaskedBatchNorm1d(len(self.mets), affine=False)
        cnn_act = get_act(self.hparams.cnn_act, True)
        if self.hparams.cnn_sep:
            self.cnn = SeparableDualCNN(len(self.mets), self.hparams.xfmr_d_model, self.hparams.cnn_ks_s, self.hparams.cnn_ex, cnn_act)
        else:
            if self.hparams.cnn_ex is not None:
                logger = logging.getLogger(name=__name__)
                logger.warning("Parameter 'cnn_ex' is ignored when not using depthwise separable CNN.")
            self.cnn = DualCNN(len(self.mets), self.hparams.xfmr_d_model, self.hparams.cnn_ks_s, cnn_act)

        if self.hparams.min_in_len < self.cnn.recept_field:
            raise ValueError("Inputs cannot be shorter than receptive field of CNN backbone.")

        xfmr_time_len = self.hparams.win_len - self.cnn.recept_field + 1
        if self.hparams.cls_tok:
            self.cls_tok = nn.Parameter(data=torch.empty(self.hparams.xfmr_d_model, dtype=torch.float32))
            xfmr_time_len += 1

        match self.hparams.xfmr_pos_enc:
            case "sinusoidal":
                self.register_buffer("pos_emb", create_sin_pos_emb(xfmr_time_len, self.hparams.xfmr_d_model), persistent=False)
                xfmr_layer = TransformerEncoderLayer(
                    self.hparams.xfmr_d_model,
                    self.hparams.xfmr_nhead,
                    self.hparams.xfmr_d_ff,
                    self.hparams.xfmr_dr,
                    self.hparams.xfmr_act,
                    self.hparams.xfmr_norm,
                    norm_first=True
                )
            case "learnable":
                self.pos_emb = nn.Parameter(data=torch.empty(xfmr_time_len, self.hparams.xfmr_d_model, dtype=torch.float32))
                xfmr_layer = TransformerEncoderLayer(
                    self.hparams.xfmr_d_model,
                    self.hparams.xfmr_nhead,
                    self.hparams.xfmr_d_ff,
                    self.hparams.xfmr_dr,
                    self.hparams.xfmr_act,
                    self.hparams.xfmr_norm,
                    norm_first=True
                )
            case "rope":
                xfmr_layer = RoFormerEncoderLayer(
                    self.hparams.xfmr_d_model,
                    self.hparams.xfmr_nhead,
                    self.hparams.xfmr_d_ff,
                    xfmr_time_len,
                    self.hparams.xfmr_dr,
                    self.hparams.xfmr_act,
                    self.hparams.xfmr_norm,
                    norm_first=True
                )
            case _:
                raise ValueError(f"Unknown positional encoding {self.hparams.xfmr_pos_enc} was specified.")

        match self.hparams.xfmr_norm:
            case "layer":
                xfmr_norm = nn.LayerNorm(self.hparams.xfmr_d_model)
            case "rms":
                xfmr_norm = nn.RMSNorm(self.hparams.xfmr_d_model)
            case _:
                raise ValueError(f"Unknown normalization {self.hparams.xfmr_norm} was specified.")

        self.xfmr = nn.TransformerEncoder(xfmr_layer, self.hparams.xfmr_n_layers, norm=xfmr_norm, enable_nested_tensor=False)

        if self.hparams.time_agg == "attn_pool":
            self.pool = MaskedGlobalAttnPool1d(self.hparams.xfmr_d_model, 1)

        self.mlp = nn.Sequential(
            nn.Linear(self.hparams.xfmr_d_model, self.hparams.xfmr_d_model // 4),
            get_act(self.hparams.mlp_act),
            nn.Dropout(p=self.hparams.mlp_dr),
            nn.Linear(self.hparams.xfmr_d_model // 4, 1)
        )

    @property
    def example_input_array(self) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.BoolTensor, torch.BoolTensor]:
        return (
            torch.empty(1, self.hparams.win_len, len(self.traj_mets), dtype=self.dtype),
            torch.empty(1, self.hparams.win_len, len(self.sensor_mets), dtype=self.dtype),
            torch.ones(1, self.hparams.win_len, dtype=torch.bool),
            torch.ones(1, self.hparams.win_len, dtype=torch.bool)
        )

    def forward(
            self,
            traj_input: torch.FloatTensor,                   # (batch, time, ch)
            sensor_input: torch.FloatTensor,                 # (batch, time, ch)
            valid_mask: Optional[torch.BoolTensor] = None,   # (batch, time)
            visible_mask: Optional[torch.BoolTensor] = None  # (batch, time)
        ) -> torch.FloatTensor:  # (batch, 1)

        if valid_mask is None:
            valid_mask = torch.ones(traj_input.shape[:2], dtype=torch.bool, device=traj_input.device)

        hidden: torch.FloatTensor
        hidden = self.bn(torch.cat((traj_input, sensor_input), dim=2).transpose(1, 2), valid_mask)
        hidden = self.cnn(hidden, valid_mask)

        hidden = einops.rearrange(hidden, "b d t -> t b d")
        if self.hparams.cls_tok:
            cls_tok = einops.repeat(self.cls_tok, "d -> 1 b d", b=hidden.shape[1])
            hidden = torch.cat((cls_tok, hidden))
        if self.hparams.xfmr_pos_enc in ("sinusoidal", "learnable"):
            pos_emb = einops.rearrange(self.pos_emb, "t d -> t 1 d")
            hidden += pos_emb

        valid_mask = valid_mask[:, -hidden.shape[0]:]
        if visible_mask is not None:
            visible_mask = ~self.mask_contract(~visible_mask, hidden.shape[0])
            valid_mask = valid_mask & visible_mask

        hidden = self.xfmr(hidden, src_key_padding_mask=~valid_mask)

        hidden = einops.rearrange(hidden, "t b d -> b d t")
        match self.hparams.time_agg:
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
                raise ValueError(f"Unknown time aggregation method {self.hparams.time_agg} was specified.")

        output = self.mlp(hidden)

        return output

    @staticmethod
    def mask_contract(mask: torch.BoolTensor, tgt_len: int) -> torch.BoolTensor:  # (batch, time) -> (batch, time)
        diff = mask.diff(prepend=torch.zeros(mask.shape[0], 1, dtype=torch.bool, device=mask.device))
        diff_cnt = diff.count_nonzero(dim=1)
        if (diff_cnt > 2).any().item():
            raise ValueError("Non-contiguous masks are not supported.")

        step = torch.arange(mask.shape[1], dtype=torch.int32, device=mask.device)  # (time, )
        min_step = torch.where(mask, step, torch.inf).min(dim=1).values            # (batch, )
        max_step = torch.where(mask, step, -torch.inf).max(dim=1).values           # (batch, )
        mask = (min_step.unsqueeze(1) - 0.5 < step[:tgt_len].unsqueeze(0)) & (step[-tgt_len:].unsqueeze(0) < max_step.unsqueeze(1) + 0.5)

        return mask


class CorVSNetPredictor(CorVSNet, BasePredModule[BaseModelHParams]):
    def forward(self, traj_input: torch.FloatTensor, sensor_input: torch.FloatTensor, valid_mask: Optional[torch.BoolTensor] = None) -> tuple[torch.FloatTensor, torch.FloatTensor]:  # (batch, time, ch), (batch, time, ch), (batch, time) -> (batch, 1), (batch, 1)
        prob = F.sigmoid(super().forward(traj_input, sensor_input, valid_mask))
        spd = traj_input[:, :, self.traj_mets.index(TrajMet.SPD)]
        linacc = sensor_input[:, :, self.sensor_mets.index(SensorMet.LINACC_NORM)]
        rel = self.rel_estim(spd, linacc, valid_mask)

        return prob, rel

    def rel_estim(self, spd: torch.FloatTensor, linacc: torch.FloatTensor, valid_mask: torch.BoolTensor | torch.FloatTensor | torch.IntTensor, eps: float = 1e-5) -> torch.FloatTensor:  # (batch, time), (batch, time), (batch, time) -> (batch, 1)
        valid_cnt = valid_mask.count_nonzero(dim=1)
        spd_mean = (valid_mask * spd).sum(dim=1) / valid_cnt
        spd_var = (valid_mask * (spd - spd_mean.unsqueeze(1)) ** 2).sum(dim=1) / valid_cnt
        linacc_mean = (valid_mask * linacc).sum(dim=1) / valid_cnt
        linacc_var = (valid_mask * (linacc - linacc_mean.unsqueeze(1)) ** 2).sum(dim=1) / valid_cnt

        spd_run_var = self.bn.running_var[self.mets.index(TrajMet.SPD)]
        linacc_run_var = self.bn.running_var[self.mets.index(SensorMet.LINACC_NORM)]
        rel = 1 / (1 + torch.min(spd_run_var / (spd_var + eps), linacc_run_var / (linacc_var + eps))).unsqueeze(1)

        return rel

    def predict_step(self, batch: dict[Modality, torch.Tensor], _: int) -> tuple[torch.DoubleTensor, torch.IntTensor, torch.IntTensor, torch.FloatTensor, torch.FloatTensor]:
        time = batch[Modality.TIME]
        track_id = batch[Modality.TRACK_ID]
        worker_id = batch[Modality.WORKER_ID]

        traj_feat = batch[Modality.TRAJ_FEAT]
        sensor_feat = batch[Modality.SENSOR_FEAT]
        valid_mask = batch[Modality.VALID_MASK]
        prob, rel = self(traj_feat, sensor_feat, valid_mask)

        return time, track_id, worker_id, prob, rel
