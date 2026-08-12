import typing
from argparse import Namespace
from typing import Any, ClassVar, Generic, Literal, Sequence
import torch
from lightning import pytorch as L
from lightning.fabric.utilities import AttributeDict
from omegaconf import DictConfig
from pydantic import BaseModel, NonNegativeInt, PositiveInt
from torch.utils import data
from typing_extensions import TypeVar
from corvs import utils
from corvs.enums import Modality, Mode, SensorMet, TrajMet


class BaseDataset(data.Dataset[dict[Modality, torch.Tensor]]):
    traj_mets: ClassVar[Sequence[TrajMet]]
    sensor_mets: ClassVar[Sequence[SensorMet]]

class BaseDataHParams(BaseModel):
    bsz:       PositiveInt
    n_workers: NonNegativeInt
    pin_mem:   bool

ModeT = TypeVar("ModeT", bound=Mode, default=Mode)
DatasetT = TypeVar("DatasetT", bound=BaseDataset, default=BaseDataset)
DataHParamsT = TypeVar("DataHParamsT", bound=BaseDataHParams, default=BaseDataHParams)

class BaseDataModule(L.LightningDataModule, Generic[ModeT, DatasetT, DataHParamsT]):
    if typing.TYPE_CHECKING:
        hparams: AttributeDict | DictConfig | DataHParamsT

    def __init__(self, hparams: dict[str, Any] | Namespace | DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.datasets: dict[ModeT, DatasetT] = {}

        t: type[DataHParamsT]
        for t in utils.resol_type_var(type(self), DataHParamsT):
            t.model_validate(self.hparams)

    @property
    def ds_cls(self) -> type[DatasetT]:
        return utils.resol_type_var(type(self), DatasetT).pop()


class BasePredDataModule(BaseDataModule[Literal[Mode.PRED], DatasetT, DataHParamsT]):
    def predict_dataloader(self) -> data.DataLoader[dict[Modality, torch.Tensor]]:
        return data.DataLoader(
            self.datasets[Mode.PRED],
            batch_size=self.hparams.bsz,
            num_workers=self.hparams.n_workers,
            pin_memory=self.hparams.pin_mem
        )
