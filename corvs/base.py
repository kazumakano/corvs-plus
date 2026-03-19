from os import PathLike
from typing import Any, Optional, Self
import pytorch_optimizer as optim
import torch
from lightning import pytorch as L
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from omegaconf import DictConfig
from safetensors import torch as safetensors
from torch import nn
from torch.optim import Optimizer
from torch.types import Device
from torch.utils import data
from torchtune import training
from corvs.loss import FocalWithLogitsLoss


class BaseDataset(data.Dataset):
    item_idx: dict[str, int]

class BaseModule(L.LightningModule):
    def __init__(self, hparams: dict[str, Any] | DictConfig, dataset_cls: type[BaseDataset], loss_pos_weight: Optional[float] = None) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_item_idx = dataset_cls.item_idx

        match self.hparams["loss"]:
            case "bce":
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=None if loss_pos_weight is None else torch.tensor(loss_pos_weight, dtype=torch.float32))
            case "focal":
                self.criterion = FocalWithLogitsLoss()
            case _:
                raise ValueError(f"unknown loss function {self.hparams['loss']} was specified")

    def configure_optimizers(self) -> Optimizer | OptimizerLRSchedulerConfig:
        match self.hparams["opt"]:
            case "adam":
                opt = optim.Adam(self.parameters(), lr=self.hparams["lr"])
            case "adamw":
                if self.hparams["sched"] == "free":
                    return optim.ScheduleFreeAdamW(self.parameters(), lr=self.hparams["lr"])
                else:
                    opt = optim.AdamW(self.parameters(), lr=self.hparams["lr"])
            case "sgd":
                if self.hparams["sched"] == "free":
                    return optim.ScheduleFreeSGD(self.parameters(), lr=self.hparams["lr"])
                else:
                    opt = optim.SGD(self.parameters(), lr=self.hparams["lr"])
            case "soap":
                opt = optim.SOAP(self.parameters(), lr=self.hparams["lr"])
            case _:
                raise ValueError(f"unknown optimizer {self.hparams['opt']} was specified")

        match self.hparams["sched"]:
            case "free":
                raise ValueError(f"free scheduler is not supported for optimizer {self.hparams['opt']}")
            case "warm_cos":
                return {
                    "optimizer": opt,
                    "lr_scheduler": {
                        "scheduler": training.get_cosine_schedule_with_warmup(opt, self.hparams["warm_step"], self.trainer.estimated_stepping_batches),
                        "interval": "step"
                    }
                }
            case None:
                return opt
            case _:
                raise ValueError(f"unknown scheduler {self.hparams['sched']} was specified")

    def on_train_epoch_start(self) -> None:
        if self.hparams["sched"] == "free":
            self.optimizers().optimizer.train()

    def on_validation_epoch_start(self) -> None:
        if self.hparams["sched"] == "free":
            self.optimizers().optimizer.eval()

    def on_test_start(self) -> None:
        if self.hparams["sched"] == "free":
            self.optimizers().optimizer.eval()

    def to_safetensors(self, path: PathLike, metadata: Optional[dict[str, str]] = None) -> None:
        safetensors.save_model(self, path, metadata=metadata)

class BasePredictModule(L.LightningModule):
    def on_predict_start(self) -> None:
        if self.hparams["sched"] == "free":
            self.optimizers().optimizer.eval()

    @classmethod
    def load_from_safetensors(cls, path: PathLike, hparams: dict[str, Any] | DictConfig, dataset_cls: type[BaseDataset], device: Device = None, **kwargs: Any) -> Self:
        self = cls(hparams=hparams, dataset_cls=dataset_cls, **kwargs)
        safetensors.load_model(self, path)    # device argument does not work with lightning
        self = self.to(device=device).eval()
        return self
