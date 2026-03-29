import math
import traceback
from pathlib import Path
from typing import Any, Literal, overload
from lightning import pytorch as L
from omegaconf import DictConfig, OmegaConf
from corvs import utils


class ArgsWriter(L.Callback):
    @overload
    def __init__(self, args: dict[str, Any] | DictConfig, /) -> None:
        ...

    @overload
    def __init__(self, **kwargs: Any) -> None:
        ...

    def __init__(self, *args: dict[str, Any] | DictConfig, **kwargs: Any) -> None:
        if len(args) == 1 and isinstance(args[0], (dict, DictConfig)) and len(kwargs) == 0:
            self.args = args[0]
        elif len(args) == 0:
            self.args = kwargs
        else:
            raise ValueError("arguments must be passed as one dictionary or keyword arguments")

    def on_fit_start(self, trainer: L.Trainer, _: L.LightningModule) -> None:
        OmegaConf.save(self.args, Path(trainer.log_dir) / "args.yaml")

    def on_test_start(self, trainer: L.Trainer, _: L.LightningModule, __: BaseException) -> None:
        OmegaConf.save(self.args, Path(trainer.log_dir) / "args.yaml")

    def on_predict_start(self, trainer: L.Trainer, _: L.LightningModule, __: BaseException) -> None:
        OmegaConf.save(self.args, Path(trainer.log_dir) / "args.yaml")

class BestMetricsWriter(L.Callback):
    def __init__(self, monitor: str, mode: Literal["min", "max"] = "min") -> None:
        self.monitor = monitor
        self.mode = mode

        self.best_mets: dict[str, float] = {}
        match self.mode:
            case "min":
                self.best_mets[self.monitor] = math.inf
            case "max":
                self.best_mets[self.monitor] = -math.inf
            case _:
                raise ValueError("mode must be 'min' or 'max'")

    def on_validation_epoch_end(self, trainer: L.Trainer, _: L.LightningModule) -> None:
        mets = {n: v.item() for n, v in trainer.callback_metrics.items()}
        if self.mode == "min" and mets[self.monitor] < self.best_mets[self.monitor] or self.mode == "max" and self.best_mets[self.monitor] < mets[self.monitor]:
            self.best_mets = mets | {"epoch": trainer.current_epoch, "step": trainer.global_step}
            trainer.logger.log_metrics({"hp_metric": self.best_mets[self.monitor]}, step=trainer.global_step)

    def on_fit_end(self, trainer: L.Trainer, _: L.LightningModule) -> None:
        OmegaConf.save(self.best_mets, Path(trainer.log_dir) / "metrics.yaml")

    def on_exception(self, trainer: L.Trainer, _: L.LightningModule, __: BaseException) -> None:
        OmegaConf.save(self.best_mets, Path(trainer.log_dir) / "metrics.yaml")

class ErrWriter(L.Callback):
    def on_exception(self, trainer: L.Trainer, _: L.LightningModule, __: BaseException) -> None:
        utils.save_txt(traceback.format_exc(), Path(trainer.log_dir) / "error.txt")
