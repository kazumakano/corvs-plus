from os import PathLike
from typing import Any, Optional, Self
import torch
from lightning import pytorch as L
from omegaconf import DictConfig
from safetensors import torch as safetensors
from torch import optim
from torchtune import training


class BaseModule(L.LightningModule):
    def __init__(self, hparams: dict[str, Any] | DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

    def configure_optimizers(self) -> None:
        match self.hparams["opt"]:
            case "adam":
                opt = optim.Adam(self.parameters(), lr=self.hparams["lr"])
            case "adamw":
                opt = optim.AdamW(self.parameters(), lr=self.hparams["lr"])
            case "sgd":
                opt = optim.SGD(self.parameters(), lr=self.hparams["lr"])
        match self.hparams["sch"]:
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

    def to_safetensors(self, file: str | PathLike, metadata: Optional[dict[str, str]] = None) -> None:
        safetensors.save_model(self, file, metadata)

class BasePredictor(L.LightningModule):
    @classmethod
    def load_from_safetensors(cls, file: str | PathLike, hparams: dict[str, Any] | DictConfig, device: int | str | torch.device = "cpu", **kwargs: Any) -> Self:
        self = cls(hparams, **kwargs)
        safetensors.load_model(self, file)    # device argument does not work with lightning
        self = self.to(device=device).eval()
        return self
