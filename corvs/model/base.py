import typing
from argparse import Namespace
from os import PathLike
from typing import Any, ClassVar, Generic, Optional, Self, Sequence
import torch
from lightning import pytorch as L
from lightning.fabric.utilities import AttributeDict
from omegaconf import DictConfig
from pydantic import BaseModel
from safetensors import torch as safetensors
from torch.types import Device, FileLike
from typing_extensions import TypeVar
from corvs import utils
from corvs.data.base import BaseDataset
from corvs.enums import Modality, SensorMet, TrajMet


class BaseModelHParams(BaseModel):
    arch: str

ModelHParamsT = TypeVar("ModelHParamsT", bound=BaseModelHParams, default=BaseModelHParams)

class BaseModule(L.LightningModule, Generic[ModelHParamsT]):
    traj_mets: ClassVar[Sequence[TrajMet]]
    sensor_mets: ClassVar[Sequence[SensorMet]]
    if typing.TYPE_CHECKING:
        hparams: AttributeDict | DictConfig | ModelHParamsT

    def __init__(self, hparams: dict[str, Any] | Namespace | DictConfig, ds_cls: type[BaseDataset] | tuple[str, str]) -> None:
        super().__init__()
        self.save_hyperparameters(hparams, ignore="ds_cls")

        t: type[ModelHParamsT]
        for t in utils.resol_type_var(type(self), ModelHParamsT):
            t.model_validate(self.hparams)

        if isinstance(ds_cls, Sequence):
            ds_cls = utils.import_by_str(*ds_cls)
        try:
            self.traj_map: list[int] = [ds_cls.traj_mets.index(tm) for tm in self.traj_mets]
            self.sensor_map: list[int] = [ds_cls.sensor_mets.index(sm) for sm in self.sensor_mets]
        except ValueError:
            raise ValueError("Datasets must contain all input metrics.") from None

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

    def on_before_batch_transfer(self, batch: dict[Modality, torch.Tensor] | Any, dataloader_idx: int) -> dict[Modality, torch.Tensor] | Any:
        if not getattr(self.trainer, "summarizing", False):  # skip example inputs during model summary
            batch[Modality.TRAJ_FEAT] = batch[Modality.TRAJ_FEAT][:, :, self.traj_map]
            batch[Modality.SENSOR_FEAT] = batch[Modality.SENSOR_FEAT][:, :, self.sensor_map]
        return super().on_before_batch_transfer(batch, dataloader_idx)

    @property
    def mets(self) -> Sequence[TrajMet | SensorMet]:
        return self.traj_mets + self.sensor_mets


class BasePredModule(BaseModule[ModelHParamsT]):
    @classmethod
    def load_from_safetensors(cls, path: str | PathLike[str], hparams: dict[str, Any] | Namespace | DictConfig, ds_cls: type[BaseDataset], device: Device = None, **kwargs: Any) -> Self:
        self = cls(hparams=hparams, ds_cls=ds_cls, **kwargs)
        safetensors.load_model(self, path)  # device argument does not work with Lightning
        self = self.to(device=device).eval()
        return self
