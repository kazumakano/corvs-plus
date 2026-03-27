from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Optional
import torch
from lightning import pytorch as L
from lightning.pytorch import callbacks, loggers
from omegaconf import OmegaConf
from corvs import CorVSFitDataModule, CorVSFitDataset, CorVSNetFitter
from corvs.callbacks import ArgsWriter, BestMetricsWriter, ErrWriter


def train(
        data_path: PathLike,
        split_path: PathLike,
        param_path: PathLike,
        exp_name: Optional[str] = None,
        start: Optional[float | str | datetime] = None,
        end: Optional[float | str | datetime] = None,
        seed: Optional[float] = None
    ) -> None:

    torch.set_float32_matmul_precision("high")
    split = OmegaConf.load(split_path)
    hparams = OmegaConf.load(param_path)

    datamodule = CorVSFitDataModule(data_path, hparams, (split["train"], split["val"], split["test"]), start=start, end=end, seed=seed)
    model = CorVSNetFitter(hparams, CorVSFitDataset)

    cbs = [
        ArgsWriter(data_path=data_path, split_path=split_path, param_path=param_path, start=start, end=end, seed=seed),
        BestMetricsWriter("val_loss"),
        ErrWriter(),
        callbacks.LearningRateMonitor(),
        callbacks.ModelCheckpoint(monitor="val_loss"),
        callbacks.ModelCheckpoint(filename="last")
    ]
    if hparams["patience"] is not None:
        cbs.append(callbacks.EarlyStopping("val_loss", patience=hparams["patience"]))
    trainer = L.Trainer(
        devices=1,
        logger=loggers.TensorBoardLogger(Path(__file__).parent / "runs", name=exp_name),
        callbacks=cbs,
        max_steps=hparams["max_step"],
        val_check_interval=hparams["val_step"],
        check_val_every_n_epoch=None
    )

    trainer.fit(model, datamodule=datamodule)

    datamodule.save_split()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", default="dataset/", help="path to dataset root directory")
    parser.add_argument("-s", "--split_path", default="configs/split.yaml", help="path to split file")
    parser.add_argument("-p", "--param_path", default="configs/hparams.yaml", help="path to hyperparameter file")
    parser.add_argument("-e", "--exp_name", help="experiment name", metavar="NAME")
    parser.add_argument("--from", help="start datetime in JST", metavar="DATETIME", dest="from_")
    parser.add_argument("--to", help="end datetime in JST", metavar="DATETIME")
    parser.add_argument("--seed", type=int, help="random seed")
    args = parser.parse_args()

    train(args.data_path, args.split_path, args.param_path, args.exp_name, args.from_, args.to, args.seed)
