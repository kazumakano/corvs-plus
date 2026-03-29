import abc
import enum
from argparse import Namespace
from os import PathLike
from typing import Any, BinaryIO, ClassVar, Generic, Literal, Optional, Self, Sequence, TypeVar
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
from corvs import utils
from corvs.loss import FocalWithLogitsLoss


class Modality(enum.Enum):
    TIME         = enum.auto()
    TRAJ_FEAT    = enum.auto()
    SENSOR_FEAT  = enum.auto()
    VALID_MASK   = enum.auto()
    VISIBLE_MASK = enum.auto()
    LABEL        = enum.auto()

class TrajMet(enum.Enum):
    SPD       = enum.auto()
    TURN_RATE = enum.auto()

class SensorMet(enum.Enum):
    LINACC_NORM = enum.auto()
    ACC_X       = enum.auto()
    ACC_Y       = enum.auto()
    ACC_Z       = enum.auto()
    GYRO_X      = enum.auto()
    GYRO_Y      = enum.auto()
    GYRO_Z      = enum.auto()

class BaseDataset(data.Dataset):
    modalities:  ClassVar[Sequence[Modality]]
    traj_mets:   ClassVar[Sequence[TrajMet]]
    sensor_mets: ClassVar[Sequence[SensorMet]]

class BaseFitDataset(BaseDataset, abc.ABC):
    @property
    @abc.abstractmethod
    def neg_ratio(self) -> float:
        ...

ModeT = TypeVar("ModeT", bound=Literal["train", "val", "test", "pred"])
DatasetT = TypeVar("DatasetT", bound=BaseDataset)

class BaseDataModule(L.LightningDataModule, Generic[ModeT, DatasetT]):
    def __init__(self, hparams: dict[str, Any] | Namespace | DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.datasets: dict[ModeT, DatasetT] = {}

FitDatasetT = TypeVar("FitDatasetT", bound=BaseFitDataset)

class BaseFitDataModule(BaseDataModule[Literal["train", "val", "test"], FitDatasetT]):
    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.datasets["train"],
            batch_size=self.hparams["bsz"],
            shuffle=True,
            num_workers=self.hparams["n_workers"],
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.datasets["val"],
            batch_size=self.hparams["bsz"],
            num_workers=self.hparams["n_workers"],
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.datasets["test"],
            batch_size=self.hparams["bsz"],
            num_workers=self.hparams["n_workers"],
            pin_memory=True
        )

class BasePredDataModule(BaseDataModule[Literal["pred"], DatasetT]):
    def predict_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.datasets["pred"],
            batch_size=self.hparams["bsz"],
            num_workers=self.hparams["n_workers"],
            pin_memory=True
        )

class BaseModule(L.LightningModule):
    def __init__(self, hparams: dict[str, Any] | Namespace | DictConfig, ds_cls: type[BaseDataset] | tuple[str, str]) -> None:
        super().__init__()
        self.save_hyperparameters(hparams, ignore="ds_cls")

        if isinstance(ds_cls, tuple):
            ds_cls = utils.import_by_str(*ds_cls)
        self.modalities  = tuple(ds_cls.modalities)
        self.traj_mets   = tuple(ds_cls.traj_mets)
        self.sensor_mets = tuple(ds_cls.sensor_mets)

    @classmethod
    def load_from_checkpoint(
            cls,
            checkpoint_path: PathLike | BinaryIO,
            ds_cls: type[BaseDataset],
            map_location: Optional[torch.device | str | dict[str, str]] = None,
            hparams_file: Optional[PathLike] = None,
            **kwargs: Any
        ) -> Self:

        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=False,
            weights_only=False,
            ds_cls=(ds_cls.__module__, ds_cls.__qualname__),    # only primitive types are allowed for extra keyword arguments
            **kwargs
        )

    @property
    def in_mets(self) -> tuple[TrajMet | SensorMet, ...]:
        return self.traj_mets + self.sensor_mets

class BaseFitModule(BaseModule):
    def setup(self, stage: Literal["fit", "validate", "test"]) -> None:
        if stage == "fit":
            match self.hparams["loss"]:
                case "bce":
                    self.train_crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.trainer.datamodule.datasets["train"].neg_ratio, dtype=torch.float32))
                case "focal":
                    self.train_crit = FocalWithLogitsLoss()
                case _:
                    raise ValueError(f"unknown loss function {self.hparams['loss']} was specified")
            self.val_crit = nn.BCEWithLogitsLoss()

    def configure_optimizers(self) -> Optimizer | OptimizerLRSchedulerConfig:
        match self.hparams["opt"]:
            case "sgd":
                if self.hparams["sched"] == "free":
                    return optim.ScheduleFreeSGD(self.parameters(), lr=self.hparams["lr"])
                else:
                    opt = optim.SGD(self.parameters(), lr=self.hparams["lr"])
            case "adam":
                opt = optim.Adam(self.parameters(), lr=self.hparams["lr"])
            case "adamw":
                if self.hparams["sched"] == "free":
                    return optim.ScheduleFreeAdamW(self.parameters(), lr=self.hparams["lr"])
                else:
                    opt = optim.AdamW(self.parameters(), lr=self.hparams["lr"])
            case "soap":
                opt = optim.SOAP(self.parameters(), lr=self.hparams["lr"])
            case _:
                raise ValueError(f"unknown optimizer {self.hparams['opt']} was specified")

        match self.hparams["sched"]:
            case "warm_cos":
                return {
                    "optimizer": opt,
                    "lr_scheduler": {
                        "scheduler": training.get_cosine_schedule_with_warmup(opt, self.hparams["warm_step"], self.trainer.estimated_stepping_batches),
                        "interval": "step"
                    }
                }
            case "free":
                raise ValueError(f"free scheduler is not supported for optimizer {self.hparams['opt']}")
            case None:
                return opt
            case _:
                raise ValueError(f"unknown learning rate scheduler {self.hparams['sched']} was specified")

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

class BasePredModule(BaseModule):
    def on_predict_start(self) -> None:
        if self.hparams["sched"] == "free":
            self.optimizers().optimizer.eval()

    @classmethod
    def load_from_safetensors(cls, path: PathLike, hparams: dict[str, Any] | Namespace | DictConfig, ds_cls: type[BaseDataset], device: Device = None, **kwargs: Any) -> Self:
        self = cls(hparams=hparams, ds_cls=ds_cls, **kwargs)
        safetensors.load_model(self, path)    # device argument does not work with lightning
        self = self.to(device=device).eval()
        return self
