from typing import Any, Optional
import einops
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from corvs.base import BaseDataset, BaseFitModule, BaseModule, BasePredictModule, DataItem
from corvs.cnn import DualCNN, SeparableDualCNN
from corvs.embedding import create_sin_pos_emb
from corvs.normalization import MaskedBatchNorm1d
from corvs.pooling import MaskedGlobalAttnPool1d, masked_global_avg_pool1d, masked_global_max_pool1d, masked_global_softmax_pool1d
from corvs.transformer import RoFormerEncoderLayer, TransformerEncoderLayer


class CorVSNet(BaseModule):
    def __init__(self, hparams: dict[str, Any] | DictConfig, dataset_cls: type[BaseDataset]) -> None:
        super().__init__(hparams, dataset_cls)

        if self.hparams["time_agg"] == "cls_tok" and not self.hparams["cls_tok"]:
            raise ValueError("time aggregation with CLS token needs to enable CLS token")

        self.bn = MaskedBatchNorm1d(9, affine=False)
        if self.hparams["cnn_sep"]:
            self.cnn = SeparableDualCNN(9, self.hparams["xformer_d_model"], self.hparams["cnn_ks_s"], self.hparams["cnn_fn"])
        else:
            self.cnn = DualCNN(9, self.hparams["xformer_d_model"], self.hparams["cnn_ks_s"])

        if self.hparams["min_input_len"] < self.cnn.recept_field:
            raise ValueError("input cannot be shorter than receptive field of CNN backbone")

        xformer_time_len = self.hparams["win_len"] - self.cnn.recept_field + 1
        if self.hparams["cls_tok"]:
            self.cls_tok = nn.Parameter(data=torch.empty(1, 1, self.hparams["xformer_d_model"], dtype=torch.float32))
            xformer_time_len += 1
        match self.hparams["xformer_pos_enc"]:
            case "learnable":
                self.pos_emb = nn.Parameter(data=torch.empty(xformer_time_len, 1, self.hparams["xformer_d_model"], dtype=torch.float32))
                xformer_layer = TransformerEncoderLayer(self.hparams["xformer_d_model"], self.hparams["xformer_nhead"], self.hparams["xformer_d_ff"], activation=self.hparams["xformer_act"], norm_first=True)
            case "sinusoidal":
                self.register_buffer("pos_emb", create_sin_pos_emb(self.hparams["xformer_d_model"], xformer_time_len).unsqueeze(1), persistent=False)
                xformer_layer = TransformerEncoderLayer(self.hparams["xformer_d_model"], self.hparams["xformer_nhead"], self.hparams["xformer_d_ff"], activation=self.hparams["xformer_act"], norm_first=True)
            case "rope":
                xformer_layer = RoFormerEncoderLayer(self.hparams["xformer_d_model"], self.hparams["xformer_nhead"], xformer_time_len, self.hparams["xformer_d_ff"], activation=self.hparams["xformer_act"], norm_first=True)
            case _:
                raise ValueError(f"unknown positional encoding {self.hparams['xformer_pos_enc']} was specified")
        self.xformer = nn.TransformerEncoder(xformer_layer, self.hparams["xformer_n_layers"], norm=nn.LayerNorm(self.hparams["xformer_d_model"]), enable_nested_tensor=False)

        if self.hparams["time_agg"] == "attn_pool":
            self.pool = MaskedGlobalAttnPool1d(self.hparams["xformer_d_model"], 1)

        self.mlp = nn.Sequential(
            nn.Linear(self.hparams["xformer_d_model"], self.hparams["xformer_d_model"] // 4),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.hparams["xformer_d_model"] // 4, 1)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
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
            valid_mask = torch.ones(traj_input.shape[:2], dtype=torch.bool, device=self.device)

        hidden: torch.FloatTensor = self.bn(torch.cat((traj_input, sensor_input), dim=2).transpose(1, 2), valid_mask)
        hidden = self.cnn(hidden, valid_mask)

        hidden = einops.rearrange(hidden, "b d t -> t b d")
        if self.hparams["cls_tok"]:
            cls_tok = einops.repeat(self.cls_tok, "1 1 d -> 1 b d", b=hidden.shape[1])
            hidden = torch.cat((cls_tok, hidden))
        if self.hparams["xformer_pos_enc"] in ("learnable", "sinusoidal"):
            pos_emb = einops.repeat(self.pos_emb, "t 1 d -> t b d", b=hidden.shape[1])
            hidden += pos_emb

        valid_mask = valid_mask[:, -len(hidden):]
        if visible_mask is not None:
            if not self._mask_is_contig(~visible_mask):
                raise ValueError("invisible region must be contiguous")
            time_idx = torch.arange(visible_mask.shape[1], dtype=torch.int32, device=self.device)
            invisible_min_idx = torch.where(~visible_mask, time_idx, torch.inf).min(dim=1).values    # (batch, )
            invisible_max_idx = torch.where(~visible_mask, time_idx, -torch.inf).max(dim=1).values    # (batch, )
            valid_mask &= (time_idx[:len(hidden)].unsqueeze(0) < invisible_min_idx.unsqueeze(1)) | (invisible_max_idx.unsqueeze(1) < time_idx[-len(hidden):].unsqueeze(0))

        hidden = self.xformer(hidden, src_key_padding_mask=~valid_mask)

        hidden = einops.rearrange(hidden, "t b d -> b d t")
        match self.hparams["time_agg"]:
            case "attn_pool":
                hidden = self.pool(hidden, valid_mask)
            case "avg_pool":
                hidden = masked_global_avg_pool1d(hidden, valid_mask)
            case "cls_tok":
                hidden = hidden[:, :, 0]
            case "max_pool":
                hidden = masked_global_max_pool1d(hidden, valid_mask)
            case "softmax_pool":
                hidden = masked_global_softmax_pool1d(hidden, valid_mask)
            case _:
                raise ValueError(f"unknown time aggregation {self.hparams['time_agg']} was specified")

        output = self.mlp(hidden)

        return output

    def _mask_is_contig(self, mask: torch.BoolTensor) -> bool:
        diff = mask.diff(prepend=torch.zeros((len(mask), 1), dtype=torch.bool, device=self.device))
        diff_cnt = diff.count_nonzero(dim=1)
        return (diff_cnt < 3).all().item()

class CorVSNetFitter(CorVSNet, BaseFitModule):
    def training_step(self, batch: list[torch.FloatTensor | torch.BoolTensor], _: int) -> torch.FloatTensor:
        logit = self(batch[self.data_item_idx[DataItem.TRAJ_FEAT]], batch[self.data_item_idx[DataItem.SENOSR_FEAT]], batch[self.data_item_idx[DataItem.VALID_MASK]], batch[self.data_item_idx[DataItem.VISIBLE_MASK]])
        loss = self.train_criterion(logit, batch[self.data_item_idx[DataItem.LABEL]])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: list[torch.FloatTensor | torch.BoolTensor], _: int) -> torch.FloatTensor:
        logit = self(batch[self.data_item_idx[DataItem.TRAJ_FEAT]], batch[self.data_item_idx[DataItem.SENOSR_FEAT]], batch[self.data_item_idx[DataItem.VALID_MASK]], batch[self.data_item_idx[DataItem.VISIBLE_MASK]])
        loss = self.val_criterion(logit, batch[self.data_item_idx[DataItem.LABEL]])
        self.log("val_loss", loss, prog_bar=True)
        return loss

class CorVSNetPredictor(CorVSNet, BasePredictModule):
    def forward(self, traj_input: torch.FloatTensor, sensor_input: torch.FloatTensor, valid_mask: Optional[torch.BoolTensor] = None) -> tuple[torch.FloatTensor, torch.FloatTensor]:    # (batch, time, channel), (batch, time, channel), (batch, time) -> (batch, 1), (batch, 1)
        prob = F.sigmoid(super().forward(traj_input, sensor_input, valid_mask))
        rel = self.rel_estim(traj_input[:, :, 0], sensor_input[:, :, 0], valid_mask)
        return prob, rel

    def rel_estim(self, spd: torch.FloatTensor, linacc: torch.FloatTensor, valid_mask: torch.BoolTensor | torch.FloatTensor | torch.IntTensor, eps: float = 1e-5) -> torch.FloatTensor:    # (batch, time), (batch, time), (batch, time) -> (batch, 1)
        cnt = valid_mask.count_nonzero(dim=1)
        spd_mean = (valid_mask * spd).sum(dim=1) / cnt
        spd_var = (valid_mask * (spd - spd_mean.unsqueeze(1)) ** 2).sum(dim=1) / cnt
        linacc_mean = (valid_mask * linacc).sum(dim=1) / cnt
        linacc_var = (valid_mask * (linacc - linacc_mean.unsqueeze(1)) ** 2).sum(dim=1) / cnt
        output = 1 / (1 + torch.min(self.bn.running_var[0] / (spd_var + eps), self.bn.running_var[2] / (linacc_var + eps))).unsqueeze(1)

        return output

    def predict_step(self, batch: list[torch.DoubleTensor | torch.FloatTensor | torch.BoolTensor], _: int) -> tuple[torch.DoubleTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        time = batch[self.data_item_idx[DataItem.TIME]]
        prob, rel = self(batch[self.data_item_idx[DataItem.TRAJ_FEAT]], batch[self.data_item_idx[DataItem.SENOSR_FEAT]], batch[self.data_item_idx[DataItem.VALID_MASK]])
        label = batch[self.data_item_idx[DataItem.LABEL]]
        return time, prob, rel, label
