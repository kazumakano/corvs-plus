import enum
import abc
from os import PathLike
from typing import Any, Literal, Optional, Self
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


class DataItem(enum.Enum):
    TIME = enum.auto()
    TRAJ_FEAT = enum.auto()
    SENOSR_FEAT = enum.auto()
    VALID_MASK = enum.auto()
    VISIBLE_MASK = enum.auto()
    LABEL = enum.auto()

class BaseDataset(data.Dataset, abc.ABC):
    @property
    @abc.abstractmethod
    def item_idx(self) -> dict[DataItem, int]:
        ...

class BaseFitDataset(BaseDataset):
    @property
    @abc.abstractmethod
    def neg_ratio(self) -> float:
        ...

class BaseModule(L.LightningModule):
    def __init__(self, hparams: dict[str, Any] | DictConfig, dataset_cls: type[BaseDataset]) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_item_idx = dataset_cls.item_idx

class BaseFitModule(BaseModule):
    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        if stage == "fit":
            match self.hparams["loss"]:
                case "bce":
                    self.train_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.trainer.datamodule.datasets["train"].neg_ratio, dtype=torch.float32))
                case "focal":
                    self.train_criterion = FocalWithLogitsLoss()
                case _:
                    raise ValueError(f"unknown loss function {self.hparams['loss']} was specified")
            self.val_criterion = nn.BCEWithLogitsLoss()

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

class BasePredictModule(BaseModule):
    def on_predict_start(self) -> None:
        if self.hparams["sched"] == "free":
            self.optimizers().optimizer.eval()

    @classmethod
    def load_from_safetensors(cls, path: PathLike, hparams: dict[str, Any] | DictConfig, dataset_cls: type[BaseDataset], device: Device = None, **kwargs: Any) -> Self:
        self = cls(hparams=hparams, dataset_cls=dataset_cls, **kwargs)
        safetensors.load_model(self, path)    # device argument does not work with lightning
        self = self.to(device=device).eval()
        return self
