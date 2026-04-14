from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Optional
import torch
from lightning import pytorch as L
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from corvs import CorVSNetPredictor, CorVSPredDataModule, CorVSPredDataset


def show_summary(traj_track_id: int, sensor_worker_id: int, dataset: CorVSPredDataset, prob: torch.FloatTensor, rel: torch.FloatTensor) -> None:
    tbl = Table("Track ID", "Worker ID", "Time (sec)", "Prob Avg", "Rel Avg", "Label")
    tbl.add_row(
        str(traj_track_id),
        str(sensor_worker_id),
        str(round(dataset.tot_time_in_sec)),
        format(prob.mean().item(), ".3f"),
        format(rel.mean().item(), ".3f"),
        str(round(dataset.label.item()))
    )
    Console().print(tbl)

def infer(
        data_path: str | PathLike[str],
        param_path: str | PathLike[str],
        weight_path: str | PathLike[str],
        traj_track_id: int,
        sensor_worker_id: int,
        start: Optional[float | str | datetime] = None,
        end: Optional[float | str | datetime] = None,
        devices: int | str | list[int] = 1
    ) -> None:

    torch.set_float32_matmul_precision("high")
    hparams = OmegaConf.load(param_path)

    trainer = L.Trainer(devices=devices, logger=False)

    datamodule = CorVSPredDataModule(data_path, traj_track_id, sensor_worker_id, hparams, start, end)
    match Path(weight_path).suffix:
        case ".ckpt":
            model = CorVSNetPredictor.load_from_checkpoint(weight_path, CorVSPredDataset, torch.device(trainer.device_ids[0]), param_path)
        case ".safetensors":
            model = CorVSNetPredictor.load_from_safetensors(weight_path, hparams, CorVSPredDataset)
        case _:
            raise ValueError("only checkpoint and safetensors are supported")

    result = trainer.predict(model, datamodule=datamodule)

    if trainer.is_global_zero and result is not None:
        prob = torch.cat([r[1] for r in result])
        rel = torch.cat([r[2] for r in result])
        show_summary(traj_track_id, sensor_worker_id, datamodule.datasets["pred"], prob, rel)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", default="dataset/", help="path to dataset root directory")
    parser.add_argument("-p", "--param_path", default="configs/hparams.yaml", help="path to hyperparameter file")
    parser.add_argument("-w", "--weight_path", default="dataset/model.safetensors", help="path to model weight file")
    parser.add_argument("-t", "--traj_track_id", type=int, required=True, help="track ID of trajectory", metavar="TRACK_ID")
    parser.add_argument("-s", "--sensor_worker_id", type=int, required=True, help="worker ID of sensor measurements", metavar="WORKER_ID")
    parser.add_argument("--from", help="start datetime in JST", metavar="DATETIME", dest="from_")
    parser.add_argument("--to", help="end datetime in JST", metavar="DATETIME")
    parser.add_argument("--devices", nargs="+", default=[0], type=int, help="computation device indices", metavar="IDX")
    args = parser.parse_args()

    infer(args.data_path, args.param_path, args.weight_path, args.traj_track_id, args.sensor_worker_id, args.from_, args.to, args.devices)
