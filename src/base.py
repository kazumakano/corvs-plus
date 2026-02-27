from os import PathLike
from typing import Any, Optional, Self
import pytorch_optimizer as optim
import torch
from lightning import pytorch as L
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from omegaconf import DictConfig
from safetensors import torch as safetensors
from torchtune import training


class BaseModule(L.LightningModule):
    def __init__(self, hparams: dict[str, Any] | DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

    def configure_optimizers(self) -> torch.optim.Optimizer | OptimizerLRSchedulerConfig:
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
                raise ValueError(f"free scheduler with optimizer {self.hparams['opt']} is not supported")
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

    def on_predict_start(self) -> None:
        if self.hparams["sched"] == "free":
            self.optimizers().optimizer.eval()

    def to_safetensors(self, path: PathLike, metadata: Optional[dict[str, str]] = None) -> None:
        safetensors.save_model(self, path, metadata=metadata)

class BasePredictor(L.LightningModule):
    @classmethod
    def load_from_safetensors(cls, path: PathLike, hparams: dict[str, Any] | DictConfig, device: int | str | torch.device = "cpu", **kwargs: Any) -> Self:
        self = cls(hparams=hparams, **kwargs)
        safetensors.load_model(self, path)    # device argument does not work with lightning
        self = self.to(device=device).eval()
        return self
