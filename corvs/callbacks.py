import math
import traceback
from pathlib import Path
from typing import Any, Literal
from lightning import pytorch as L
from omegaconf import OmegaConf
from corvs import utils


class ArgsWriter(L.Callback):
    def __init__(self, **kwargs: Any) -> None:
        self.args = kwargs

    def on_fit_end(self, trainer: L.Trainer, _: L.LightningModule) -> None:
        OmegaConf.save(self.args, Path(trainer.log_dir) / "args.yaml")

    def on_exception(self, trainer: L.Trainer, _: L.LightningModule, __: BaseException) -> None:
        OmegaConf.save(self.args, Path(trainer.log_dir) / "args.yaml")

class BestMetricsWriter(L.Callback):
    def __init__(self, monitor: str, mode: Literal["min", "max"] = "min") -> None:
        self.monitor = monitor
        self.mode    = mode

        self.best_metrics: dict[str, float] = {}
        match self.mode:
            case "min":
                self.best_metrics[self.monitor] = math.inf
            case "max":
                self.best_metrics[self.monitor] = -math.inf
            case _:
                raise ValueError("mode must be 'min' or 'max'")

    def on_validation_epoch_end(self, trainer: L.Trainer, _: L.LightningModule) -> None:
        metrics = {n: v.item() for n, v in trainer.callback_metrics.items()}
        if self.mode == "min" and metrics[self.monitor] < self.best_metrics[self.monitor] or self.mode == "max" and self.best_metrics[self.monitor] < metrics[self.monitor]:
            self.best_metrics = metrics | {"epoch": trainer.current_epoch, "step": trainer.global_step}
            trainer.logger.log_metrics({"hp_metric": self.best_metrics[self.monitor]}, step=trainer.global_step)

    def on_fit_end(self, trainer: L.Trainer, _: L.LightningModule) -> None:
        OmegaConf.save(self.best_metrics, Path(trainer.log_dir) / "metrics.yaml")

    def on_exception(self, trainer: L.Trainer, _: L.LightningModule, __: BaseException) -> None:
        OmegaConf.save(self.best_metrics, Path(trainer.log_dir) / "metrics.yaml")

class ErrWriter(L.Callback):
    def on_exception(self, trainer: L.Trainer, _: L.LightningModule, __: BaseException) -> None:
        utils.save_txt(traceback.format_exc(), Path(trainer.log_dir) / "error.txt")
