import pathlib
from datetime import datetime
from os import PathLike
from typing import Optional
import torch
from lightning import pytorch as L
from omegaconf import OmegaConf
from rich import console, table
from corvs import CorVSDemoDataModule, CorVSDemoDataset, CorVSNetPredictor, log
from corvs.enums import Mode


def show_summary(dataset: CorVSDemoDataset, prob: torch.FloatTensor, rel: torch.FloatTensor) -> None:
    tbl = table.Table("Track ID", "Worker ID", "Time (sec)", "Prob Avg", "Rel Avg", "Label")
    tbl.add_row(
        str(dataset.track_id.item()),
        str(dataset.worker_id.item()),
        str(round(dataset.tot_time_in_sec)),
        format(prob.mean().item(), ".3f"),
        format(rel.mean().item(), ".3f"),
        str(round(dataset.label.item()))
    )
    console.Console().print(tbl)

def demo(
        data_path: str | PathLike[str],
        param_path: str | PathLike[str],
        weight_path: str | PathLike[str],
        traj_track_id: int,
        sensor_worker_id: int,
        start: Optional[float | str | datetime] = None,
        end: Optional[float | str | datetime] = None,
        devices: int | str | list[int] = 1
    ) -> None:

    log.init_all_loggers()
    torch.set_float32_matmul_precision("high")
    hparams = OmegaConf.load(param_path)

    trainer = L.Trainer(devices=devices, logger=False)

    datamodule = CorVSDemoDataModule(data_path, traj_track_id, sensor_worker_id, hparams, start, end)
    match pathlib.Path(weight_path).suffix:
        case ".ckpt":
            model = CorVSNetPredictor.load_from_checkpoint(weight_path, datamodule.ds_cls, torch.device(trainer.device_ids[0]), param_path)
        case ".safetensors":
            model = CorVSNetPredictor.load_from_safetensors(weight_path, hparams, datamodule.ds_cls)
        case _:
            raise ValueError("Only checkpoint and safetensors files are supported.")

    result = trainer.predict(model=model, datamodule=datamodule)

    if trainer.is_global_zero and result is not None:
        prob = torch.cat([r[3] for r in result]).squeeze(1)
        rel = torch.cat([r[4] for r in result]).squeeze(1)
        show_summary(datamodule.datasets[Mode.PRED], prob, rel)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", default="data/", help="Path to dataset root directory")
    parser.add_argument("-p", "--param_path", default="configs/hparams.yaml", help="Path to hyperparameter file")
    parser.add_argument("-w", "--weight_path", default="models/model.safetensors", help="Path to model weight file")
    parser.add_argument("-t", "--traj_track_id", type=int, required=True, help="Track ID of trajectory", metavar="TRACK_ID")
    parser.add_argument("-s", "--sensor_worker_id", type=int, required=True, help="Worker ID of sensor measurements", metavar="WORKER_ID")
    parser.add_argument("--from", help="Start datetime in JST", metavar="DATETIME", dest="from_")
    parser.add_argument("--to", help="End datetime in JST", metavar="DATETIME")
    parser.add_argument("--devices", nargs="+", default=[0], type=int, help="Computation device indices", metavar="IDX")
    args = parser.parse_args()

    demo(args.data_path, args.param_path, args.weight_path, args.traj_track_id, args.sensor_worker_id, args.from_, args.to, args.devices)
