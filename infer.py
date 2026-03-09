from os import PathLike
from typing import Optional
import torch
from lightning import pytorch as L
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from corvs import CorVSNetPredictor, CorVSPredictDataModule
from corvs.data import CorVSPredictDataset


def show_summary(traj_track_id: int, sensor_worker_id: int, dataset: CorVSPredictDataset, prob: torch.FloatTensor, rel: torch.FloatTensor) -> None:
    tbl = Table("Track ID", "Worker ID", "Time (sec)", "Prob Avg", "Rel Avg")
    tbl.add_row(str(traj_track_id), str(sensor_worker_id), str(round(dataset.time_len)), format(prob.mean().item(), ".3f"), format(rel.mean().item(), ".3f"))
    Console().print(tbl)

def infer(data_path: PathLike, param_path: PathLike, weight_path: PathLike, traj_track_id: int, sensor_worker_id: int, start: Optional[str] = None, stop: Optional[str] = None) -> None:
    torch.set_float32_matmul_precision("high")
    hparams = OmegaConf.load(param_path)

    datamodule = CorVSPredictDataModule(data_path, traj_track_id, sensor_worker_id, hparams, start, stop)
    model = CorVSNetPredictor.load_from_safetensors(weight_path, hparams)
    trainer = L.Trainer(devices=1, logger=False)

    result = trainer.predict(model, datamodule=datamodule)

    if result is not None:
        prob = torch.cat([r[1] for r in result])
        rel = torch.cat([r[2] for r in result])
        show_summary(traj_track_id, sensor_worker_id, datamodule.datasets["pred"], prob, rel)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", default="dataset/", help="path to dataset root directory")
    parser.add_argument("-p", "--param_path", default="configs/hparams.yaml", help="path to hyperparameter file")
    parser.add_argument("-w", "--weight_path", default="dataset/model.safetensors", help="path to model weight file")
    parser.add_argument("-t", "--traj_track_id", type=int, required=True, help="trajectory track ID")
    parser.add_argument("-s", "--sensor_worker_id", type=int, required=True, help="sensor worker ID")
    parser.add_argument("--start", help="start time in JST (e.g., 2024-10-03 11:30:00)")
    parser.add_argument("--stop", help="stop time in JST (e.g., 2024-10-03 12:00:00)")
    args = parser.parse_args()

    infer(args.data_path, args.param_path, args.weight_path, args.traj_track_id, args.sensor_worker_id, args.start, args.stop)
