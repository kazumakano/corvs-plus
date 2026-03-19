from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Optional
import numpy as np
import pandas as pd
import torch
from lightning import pytorch as L
from numpy import linalg
from omegaconf import DictConfig
from scipy import ndimage
from scipy.interpolate import interp1d
from torch.utils import data
from corvs import preprocess, utils
from corvs.base import BaseDataset, DataItem

TRAJ_FREQ = 2.5
TRAJ_RESOL = 0.01
SENSOR_FREQ = 100

class CorVSDataset(BaseDataset):
    ...

class CorVSDataModule(L.LightningDataModule):
    ...

class CorVSPredictDataset(BaseDataset):
    item_idx = {DataItem.TIME: 0, DataItem.TRAJ_FEAT: 1, DataItem.SENOSR_FEAT: 2, DataItem.VALID_MASK: 3}

    def __init__(
            self,
            path: PathLike,
            traj_track_id: int,
            sensor_worker_id: int,
            freq_in_hz: float,
            smooth_in_sec: float,
            min_input_len: int,
            win_len: int,
            win_stride: int,
            start: Optional[float] = None,
            stop: Optional[float] = None
        ) -> None:
        self.freq = freq_in_hz
        self.win_len, self.win_stride = win_len, win_stride

        traj_data = self._load_traj_data(Path(path) / "trajectory", traj_track_id, start, stop)
        sensor_data = self._load_sensor_data(Path(path) / "sensor", sensor_worker_id, start, stop)

        self.time: list[torch.DoubleTensor] = []
        self.traj_feat: list[torch.FloatTensor] = []
        self.sensor_feat: list[torch.FloatTensor] = []
        self.map: list[tuple[int, int, int]] = []
        if len(sensor_data) / SENSOR_FREQ > min_input_len / self.freq:
            meas = ndimage.gaussian_filter1d(np.column_stack((linalg.norm(sensor_data[["linacc_x", "linacc_y", "linacc_z"]], axis=1), sensor_data[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]])), smooth_in_sec * SENSOR_FREQ, axis=0)

            for i, td in preprocess.seg_by_timeout(traj_data, 5):
                traj_time = np.arange(td.iloc[0]["time"], td.iloc[-1]["time"], step=1 / TRAJ_FREQ, dtype=np.float64)

                if (len(traj_time) - 2) / TRAJ_FREQ > min_input_len / self.freq:
                    loc = interp1d(td["time"], td[["x", "y"]], axis=0, copy=False, fill_value="extrapolate", assume_sorted=True)(traj_time)
                    spd = ndimage.gaussian_filter1d(preprocess.loc_to_spd(loc, TRAJ_FREQ, TRAJ_RESOL), smooth_in_sec * TRAJ_FREQ)
                    ang_vel = ndimage.gaussian_filter1d(preprocess.loc_to_ang_vel(loc, TRAJ_FREQ), smooth_in_sec * TRAJ_FREQ)

                    synced_time, synced_spd, synced_ang_vel, synced_meas = preprocess.sync(traj_time[:-1] + 0.5 / TRAJ_FREQ, spd, traj_time[1:-1], ang_vel, sensor_data["time"], meas, self.freq)
                    self.traj_feat.append(torch.from_numpy(np.column_stack((synced_spd.astype(np.float32), synced_ang_vel.astype(np.float32)))))
                    self.sensor_feat.append(torch.from_numpy(synced_meas.astype(np.float32)))

                    valid_len = min(self.win_len, len(synced_time))
                    win_num = max(1, (len(synced_time) - self.win_len) // self.win_stride + 1)
                    self.time.append(torch.empty(win_num, dtype=torch.float64))
                    for j in range(win_num):
                        self.time[-1][j] = synced_time[j * self.win_stride]
                        self.map.append((i, j, valid_len))

    @staticmethod
    def _load_traj_data(path: Path, traj_track_id: int, start: float | None, stop: float | None) -> pd.DataFrame:
        traj_data_list = []
        for f in sorted(path.glob("trajectory_????????_??_??.csv")):
            traj_data_list.append(pd.read_csv(f, usecols=("time", "track", "x", "y"), engine="pyarrow"))
        if len(traj_data_list) > 0:
            traj_data = pd.concat(traj_data_list, ignore_index=True)
        else:
            traj_data = pd.DataFrame(columns=("time", "track", "x", "y"))
        traj_data = traj_data[traj_data["track"] == traj_track_id]
        if start is not None:
            traj_data = traj_data[traj_data["time"] >= start]
        if stop is not None:
            traj_data = traj_data[traj_data["time"] < stop]

        return traj_data

    @staticmethod
    def _load_sensor_data(path: Path, sensor_worker_id: int, start: float | None, stop: float | None) -> pd.DataFrame:
        sensor_data_list = []
        for f in sorted(path.glob(f"sensor_????????_??_??_{sensor_worker_id:02d}_??.csv")):
            sensor_data_list.append(pd.read_csv(f, usecols=("time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "linacc_x", "linacc_y", "linacc_z"), engine="pyarrow"))
        if len(sensor_data_list) > 0:
            sensor_data = pd.concat(sensor_data_list, ignore_index=True)
        else:
            sensor_data = pd.DataFrame(columns=("time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "linacc_x", "linacc_y", "linacc_z"))
        if start is not None:
            sensor_data = sensor_data[sensor_data["time"] >= start]
        if stop is not None:
            sensor_data = sensor_data[sensor_data["time"] < stop]

        return sensor_data

    def __getitem__(self, idx: int) -> tuple[torch.DoubleTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
        return (
            self.time[self.map[idx][0]][self.map[idx][1]].unsqueeze(0),
            preprocess.pad(self.traj_feat[self.map[idx][0]][self.map[idx][1] * self.win_stride:self.map[idx][1] * self.win_stride + self.win_len].unsqueeze(0), self.win_len).squeeze(dim=1),
            preprocess.pad(self.sensor_feat[self.map[idx][0]][self.map[idx][1] * self.win_stride:self.map[idx][1] * self.win_stride + self.win_len].unsqueeze(0), self.win_len).squeeze(dim=1),
            torch.arange(self.win_len, dtype=torch.int32) < self.map[idx][2]
        )

    def __len__(self) -> int:
        return len(self.map)

    @property
    def tot_time_in_sec(self) -> float:
        tot_time = 0
        for f in self.traj_feat:
            tot_time += len(f) / self.freq
        return tot_time

class CorVSPredictDataModule(L.LightningDataModule):
    def __init__(self, path: PathLike, traj_track_id: int, sensor_worker_id: int, hparams: dict[str, Any] | DictConfig, start: Optional[float | str | datetime] = None, stop: Optional[float | str | datetime] = None) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.datasets: dict[Literal["pred"], CorVSPredictDataset] = {}
        self.root_path = path
        self.traj_track_id, self.sensor_worker_id = traj_track_id, sensor_worker_id

        if isinstance(start, str):
            start = utils.str_to_datetime(start, utils.jst)
        if isinstance(start, datetime):
            start = start.timestamp()
        if isinstance(stop, str):
            stop = utils.str_to_datetime(stop, utils.jst)
        if isinstance(stop, datetime):
            stop = stop.timestamp()
        self.start, self.stop = start, stop

    def setup(self, stage: Literal["predict"]) -> None:
        match stage:
            case "predict":
                if "pred" not in self.datasets.keys():
                    self.datasets["pred"] = CorVSPredictDataset(
                        self.root_path,
                        self.traj_track_id,
                        self.sensor_worker_id,
                        self.hparams["freq"],
                        self.hparams["smooth"],
                        self.hparams["min_input_len"],
                        self.hparams["win_len"],
                        1,
                        self.start,
                        self.stop
                    )

    def predict_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.datasets["pred"], batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"], pin_memory=True)
