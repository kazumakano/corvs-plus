import abc
import enum
import warnings
from argparse import Namespace
from os import PathLike
from typing import Any, ClassVar, Generic, Literal, Optional, Self, Sequence, TypeVar
import pytorch_optimizer as optim
import torch
from lightning import pytorch as L
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from omegaconf import DictConfig
from safetensors import torch as safetensors
from torch.optim import Optimizer
from torch.types import Device, FileLike
from torch.utils import data
from torchtune import training
from corvs import utils
from corvs.loss import BCEWithLogitsLoss, FocalWithLogitsLoss


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

class BaseDataset(data.Dataset[Sequence[torch.Tensor]]):
    modalities:  ClassVar[Sequence[Modality]]
    traj_mets:   ClassVar[Sequence[TrajMet]]
    sensor_mets: ClassVar[Sequence[SensorMet]]

class BaseFitDataset(BaseDataset, abc.ABC):
    def __len__(self) -> int:
        return len(self.label)

    @property
    @abc.abstractmethod
    def label(self) -> torch.FloatTensor:
        ...

    @property
    def neg_ratio(self) -> float:
        pos_cnt = self.label.count_nonzero().item()
        return (len(self) - pos_cnt) / pos_cnt

ModeT = TypeVar("ModeT", bound=Literal["train", "val", "test", "pred"])
DatasetT = TypeVar("DatasetT", bound=BaseDataset)

class BaseDataModule(L.LightningDataModule, Generic[ModeT, DatasetT]):
    def __init__(self, hparams: dict[str, Any] | Namespace | DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.datasets: dict[ModeT, DatasetT] = {}

FitDatasetT = TypeVar("FitDatasetT", bound=BaseFitDataset)

class BaseFitDataModule(BaseDataModule[Literal["train", "val", "test"], FitDatasetT]):
    def __init__(self, hparams: dict[str, Any] | Namespace | DictConfig, seed: Optional[int] = None) -> None:
        super().__init__(hparams)

        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def train_dataloader(self) -> data.DataLoader[Sequence[torch.Tensor]]:
        if self.hparams["balance"] == "sample":
            if len(self.datasets["train"]) > 2 ** 24:
                raise RuntimeError("weighted sampler is not supported for datasets with lengths over 2^24")

            weight = torch.where(self.datasets["train"].label.squeeze(1).bool(), self.datasets["train"].neg_ratio, 1)
            sampler = data.WeightedRandomSampler(weight, len(self.datasets["train"]), generator=self.rng)

            return data.DataLoader(
                self.datasets["train"],
                batch_size=self.hparams["bsz"],
                sampler=sampler,
                num_workers=self.hparams["n_workers"],
                pin_memory=True,
                drop_last=True,
                persistent_workers=True
            )
        else:
            return data.DataLoader(
                self.datasets["train"],
                batch_size=self.hparams["bsz"],
                shuffle=True,
                num_workers=self.hparams["n_workers"],
                pin_memory=True,
                drop_last=True,
                generator=self.rng,
                persistent_workers=True
            )

    def val_dataloader(self) -> data.DataLoader[Sequence[torch.Tensor]]:
        return data.DataLoader(
            self.datasets["val"],
            batch_size=self.hparams["bsz"],
            num_workers=self.hparams["n_workers"],
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self) -> data.DataLoader[Sequence[torch.Tensor]]:
        return data.DataLoader(
            self.datasets["test"],
            batch_size=self.hparams["bsz"],
            num_workers=self.hparams["n_workers"],
            pin_memory=True
        )

    @property
    def seed(self) -> int:
        return self.rng.initial_seed()

class BasePredDataModule(BaseDataModule[Literal["pred"], DatasetT]):
    def predict_dataloader(self) -> data.DataLoader[Sequence[torch.Tensor]]:
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
            checkpoint_path: FileLike,
            ds_cls: type[BaseDataset],
            map_location: Optional[torch.device | str | dict[str, str]] = None,
            hparams_file: Optional[str | PathLike[str]] = None,
            **kwargs: Any
        ) -> Self:

        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=False,
            weights_only=False,
            ds_cls=(ds_cls.__module__, ds_cls.__qualname__),  # only serializable types are allowed for extra keyword arguments
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
                    self.train_crit = BCEWithLogitsLoss(
                        pos_weight=torch.tensor(self.trainer.datamodule.datasets["train"].neg_ratio, dtype=torch.float32) if self.hparams["balance"] == "loss" else None,
                        label_smoothing=self.hparams["label_smooth"]
                    )
                case "focal":
                    if self.hparams["label_smooth"] is not None:
                        warnings.warn("parameter 'label_smooth' is ignored when using focal loss")
                    self.train_crit = FocalWithLogitsLoss()
                case _:
                    raise ValueError(f"unknown loss function {self.hparams['loss']} was specified")

        if stage in ("fit", "validate"):
            self.val_crit = BCEWithLogitsLoss()

    def configure_optimizers(self) -> Optimizer | OptimizerLRSchedulerConfig:
        if self.hparams["sched"] != "warm_cos" and self.hparams["warm_step"] is not None:
            warnings.warn(UserWarning("parameter 'warm_step' is ignored when not using warmup scheduler"))

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
            self.optimizers().train()

    def on_validation_epoch_start(self) -> None:
        if self.hparams["sched"] == "free" and self.optimizers() != []:
            self.optimizers().eval()

    def on_test_start(self) -> None:
        if self.hparams["sched"] == "free" and self.optimizers() != []:
            self.optimizers().eval()

    def to_safetensors(self, path: str | PathLike[str], metadata: Optional[dict[str, str]] = None) -> None:
        safetensors.save_model(self, path, metadata=metadata)

class BasePredModule(BaseModule):
    @classmethod
    def load_from_safetensors(cls, path: str | PathLike[str], hparams: dict[str, Any] | Namespace | DictConfig, ds_cls: type[BaseDataset], device: Device = None, **kwargs: Any) -> Self:
        self = cls(hparams=hparams, ds_cls=ds_cls, **kwargs)
        safetensors.load_model(self, path)  # device argument does not work with lightning
        self = self.to(device=device).eval()
        return self
