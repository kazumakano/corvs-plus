from typing import Any, Optional
import einops
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from corvs import utils
from corvs.base import BaseDataset, BaseFitDataset, BaseFitModule, BaseModule, BasePredictModule, Modality, SensorMet, TrajMet
from corvs.cnn import DualCNN, SeparableDualCNN
from corvs.embedding import create_sin_pos_emb
from corvs.normalization import MaskedBatchNorm1d
from corvs.pooling import MaskedGlobalAttnPool1d, masked_global_avg_pool1d, masked_global_max_pool1d, masked_global_softmax_pool1d
from corvs.transformer import RoFormerEncoderLayer, TransformerEncoderLayer


class CorVSNet(BaseModule):
    def __init__(self, hparams: dict[str, Any] | DictConfig, ds_cls: type[BaseDataset]) -> None:
        super().__init__(hparams, ds_cls)

        if self.hparams["time_agg"] == "cls_tok" and not self.hparams["cls_tok"]:
            raise ValueError("time aggregation with CLS token needs to enable CLS token")

        self.bn = MaskedBatchNorm1d(len(self.in_mets), affine=False)
        if self.hparams["cnn_sep"]:
            self.cnn = SeparableDualCNN(
                len(self.in_mets),
                self.hparams["xformer_d_model"],
                self.hparams["cnn_ks_s"],
                self.hparams["cnn_fn"],
                utils.str_to_mod(self.hparams["cnn_act"], True)
            )
        else:
            self.cnn = DualCNN(
                len(self.in_mets),
                self.hparams["xformer_d_model"],
                self.hparams["cnn_ks_s"],
                utils.str_to_mod(self.hparams["cnn_act"], True)
            )

        if self.hparams["min_in_len"] < self.cnn.recept_field:
            raise ValueError("input cannot be shorter than receptive field of CNN backbone")

        xformer_time_len = self.hparams["win_len"] - self.cnn.recept_field + 1
        if self.hparams["cls_tok"]:
            self.cls_tok = nn.Parameter(data=torch.empty(1, 1, self.hparams["xformer_d_model"], dtype=torch.float32))
            xformer_time_len += 1
        match self.hparams["xformer_pos_enc"]:
            case "sinusoidal":
                self.register_buffer("pos_emb", create_sin_pos_emb(self.hparams["xformer_d_model"], xformer_time_len).unsqueeze(1), persistent=False)
                xformer_layer = TransformerEncoderLayer(
                    self.hparams["xformer_d_model"],
                    self.hparams["xformer_nhead"],
                    self.hparams["xformer_d_ff"],
                    self.hparams["xformer_dr"],
                    self.hparams["xformer_act"],
                    self.hparams["xformer_norm"],
                    norm_first=True
                )
            case "learnable":
                self.pos_emb = nn.Parameter(data=torch.empty(xformer_time_len, 1, self.hparams["xformer_d_model"], dtype=torch.float32))
                xformer_layer = TransformerEncoderLayer(
                    self.hparams["xformer_d_model"],
                    self.hparams["xformer_nhead"],
                    self.hparams["xformer_d_ff"],
                    self.hparams["xformer_dr"],
                    self.hparams["xformer_act"],
                    self.hparams["xformer_norm"],
                    norm_first=True
                )
            case "rope":
                xformer_layer = RoFormerEncoderLayer(
                    self.hparams["xformer_d_model"],
                    self.hparams["xformer_nhead"],
                    xformer_time_len,
                    self.hparams["xformer_d_ff"],
                    self.hparams["xformer_dr"],
                    self.hparams["xformer_act"],
                    self.hparams["xformer_norm"],
                    norm_first=True
                )
            case _:
                raise ValueError(f"unknown positional encoding {self.hparams['xformer_pos_enc']} was specified")
        self.xformer = nn.TransformerEncoder(
            xformer_layer,
            self.hparams["xformer_n_layers"],
            norm=nn.LayerNorm(self.hparams["xformer_d_model"]),
            enable_nested_tensor=False
        )

        if self.hparams["time_agg"] == "attn_pool":
            self.pool = MaskedGlobalAttnPool1d(self.hparams["xformer_d_model"], 1)

        self.mlp = nn.Sequential(
            nn.Linear(self.hparams["xformer_d_model"], self.hparams["xformer_d_model"] // 4),
            utils.str_to_mod(self.hparams["mlp_act"]),
            nn.Dropout(p=self.hparams["mlp_dr"]),
            nn.Linear(self.hparams["xformer_d_model"] // 4, 1)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                m.reset_parameters()

            elif isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

            elif isinstance(m, nn.MultiheadAttention):
                m._reset_parameters()

        if self.hparams["cls_tok"]:
            init.xavier_uniform_(self.cls_tok)
        if self.hparams["xformer_pos_enc"] == "learnable":
            init.xavier_uniform_(self.pos_emb)

    def forward(self, traj_input: torch.FloatTensor, sensor_input: torch.FloatTensor, valid_mask: Optional[torch.BoolTensor] = None, visible_mask: Optional[torch.BoolTensor] = None) -> torch.FloatTensor:    # (batch, time, channel), (batch, time, channel), (batch, time), (batch, time) -> (batch, 1)
        if valid_mask is None:
            valid_mask = torch.ones(traj_input.shape[:2], dtype=torch.bool, device=traj_input.device)

        hidden: torch.FloatTensor

        hidden = self.bn(torch.cat((traj_input, sensor_input), dim=2).transpose(1, 2), valid_mask)
        hidden = self.cnn(hidden, valid_mask)

        hidden = einops.rearrange(hidden, "b d t -> t b d")
        if self.hparams["cls_tok"]:
            cls_tok = einops.repeat(self.cls_tok, "1 1 d -> 1 b d", b=hidden.shape[1])
            hidden = torch.cat((cls_tok, hidden))
        if self.hparams["xformer_pos_enc"] in ("sinusoidal", "learnable"):
            pos_emb = einops.repeat(self.pos_emb, "t 1 d -> t b d", b=hidden.shape[1])
            hidden += pos_emb

        valid_mask = valid_mask[:, -hidden.shape[0]:]
        if visible_mask is not None:
            if not self._mask_is_contig(~visible_mask):
                raise ValueError("invisible region must be contiguous")
            time_idx = torch.arange(visible_mask.shape[1], dtype=torch.int32, device=visible_mask.device)    # (time, )
            invisible_min_idx = torch.where(visible_mask, torch.inf, time_idx).min(dim=1).values             # (batch, )
            invisible_max_idx = torch.where(visible_mask, -torch.inf, time_idx).max(dim=1).values            # (batch, )
            visible_mask = (time_idx[:hidden.shape[0]].unsqueeze(0) < invisible_min_idx.unsqueeze(1) - 0.5) | (invisible_max_idx.unsqueeze(1) + 0.5 < time_idx[-hidden.shape[0]:].unsqueeze(0))    # (batch, time)
            valid_mask = valid_mask & visible_mask

        hidden = self.xformer(hidden, src_key_padding_mask=~valid_mask)

        hidden = einops.rearrange(hidden, "t b d -> b d t")
        match self.hparams["time_agg"]:
            case "avg_pool":
                hidden = masked_global_avg_pool1d(hidden, valid_mask)
            case "max_pool":
                hidden = masked_global_max_pool1d(hidden, valid_mask)
            case "softmax_pool":
                hidden = masked_global_softmax_pool1d(hidden, valid_mask)
            case "attn_pool":
                hidden = self.pool(hidden, valid_mask)
            case "cls_tok":
                hidden = hidden[:, :, 0]
            case _:
                raise ValueError(f"unknown time aggregation {self.hparams['time_agg']} was specified")

        output = self.mlp(hidden)

        return output

    @staticmethod
    def _mask_is_contig(mask: torch.BoolTensor) -> bool:
        diff = mask.diff(prepend=torch.zeros(mask.shape[0], 1, dtype=torch.bool, device=mask.device))
        diff_cnt = diff.count_nonzero(dim=1)
        return (diff_cnt < 3).all().item()

class CorVSNetFitter(CorVSNet, BaseFitModule):
    def __init__(self, hparams: dict[str, Any] | DictConfig, ds_cls: type[BaseFitDataset]) -> None:
        super().__init__(hparams, ds_cls)

        self.example_input_array = (
            torch.empty(1, self.hparams["win_len"], len(self.traj_mets), dtype=torch.float32),
            torch.empty(1, self.hparams["win_len"], len(self.sensor_mets), dtype=torch.float32),
            torch.ones(1, self.hparams["win_len"], dtype=torch.bool),
            torch.ones(1, self.hparams["win_len"], dtype=torch.bool)
        )

    def training_step(self, batch: list[torch.FloatTensor | torch.BoolTensor], _: int) -> torch.FloatTensor:
        logit = self(batch[self.modalities.index(Modality.TRAJ_FEAT)], batch[self.modalities.index(Modality.SENSOR_FEAT)], batch[self.modalities.index(Modality.VALID_MASK)], batch[self.modalities.index(Modality.VISIBLE_MASK)])
        loss  = self.train_crit(logit, batch[self.modalities.index(Modality.LABEL)])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: list[torch.FloatTensor | torch.BoolTensor], _: int) -> torch.FloatTensor:
        logit = self(batch[self.modalities.index(Modality.TRAJ_FEAT)], batch[self.modalities.index(Modality.SENSOR_FEAT)], batch[self.modalities.index(Modality.VALID_MASK)], batch[self.modalities.index(Modality.VISIBLE_MASK)])
        loss  = self.val_crit(logit, batch[self.modalities.index(Modality.LABEL)])
        self.log("val_loss", loss, prog_bar=True)
        return loss

class CorVSNetPredictor(CorVSNet, BasePredictModule):
    def forward(self, traj_input: torch.FloatTensor, sensor_input: torch.FloatTensor, valid_mask: Optional[torch.BoolTensor] = None) -> tuple[torch.FloatTensor, torch.FloatTensor]:    # (batch, time, channel), (batch, time, channel), (batch, time) -> (batch, 1), (batch, 1)
        prob = F.sigmoid(super().forward(traj_input, sensor_input, valid_mask))
        spd = traj_input[:, :, self.traj_mets.index(TrajMet.SPD)]
        linacc = sensor_input[:, :, self.sensor_mets.index(SensorMet.LINACC_NORM)]
        rel = self.rel_estim(spd, linacc, valid_mask)
        return prob, rel

    def rel_estim(self, spd: torch.FloatTensor, linacc: torch.FloatTensor, valid_mask: torch.BoolTensor | torch.FloatTensor | torch.IntTensor, eps: float = 1e-5) -> torch.FloatTensor:    # (batch, time), (batch, time), (batch, time) -> (batch, 1)
        cnt = valid_mask.count_nonzero(dim=1)
        spd_mean = (valid_mask * spd).sum(dim=1) / cnt
        spd_var  = (valid_mask * (spd - spd_mean.unsqueeze(1)) ** 2).sum(dim=1) / cnt
        linacc_mean = (valid_mask * linacc).sum(dim=1) / cnt
        linacc_var  = (valid_mask * (linacc - linacc_mean.unsqueeze(1)) ** 2).sum(dim=1) / cnt

        spd_run_var    = self.bn.running_var[self.in_mets.index(TrajMet.SPD)]
        linacc_run_var = self.bn.running_var[self.in_mets.index(SensorMet.LINACC_NORM)]

        output = 1 / (1 + torch.min(spd_run_var / (spd_var + eps), linacc_run_var / (linacc_var + eps))).unsqueeze(1)

        return output

    def predict_step(self, batch: list[torch.DoubleTensor | torch.FloatTensor | torch.BoolTensor], _: int) -> tuple[torch.DoubleTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        time = batch[self.modalities.index(Modality.TIME)]
        prob, rel = self(batch[self.modalities.index(Modality.TRAJ_FEAT)], batch[self.modalities.index(Modality.SENSOR_FEAT)], batch[self.modalities.index(Modality.VALID_MASK)])
        label = batch[self.modalities.index(Modality.LABEL)]
        return time, prob, rel, label
