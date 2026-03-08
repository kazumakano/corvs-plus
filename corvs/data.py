from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import torch
from lightning import pytorch as L
from numpy import linalg
from omegaconf import DictConfig
from scipy import ndimage
from scipy.interpolate import interp1d
from torch.utils import data
from corvs import preprocess

TRAJ_FREQ = 2.5
TRAJ_RESOL = 0.01
SENSOR_FREQ = 100

class CorVSDataset(data.Dataset):
    ...

class CorVSDataModule(L.LightningDataModule):
    ...

class CorVSPredictDataset(data.Dataset):
    def __init__(self, path: PathLike, traj_track_id: int, sensor_worker_id: int, hparams: dict[str, Any] | DictConfig, start: Optional[float | datetime] = None, stop: Optional[float | datetime] = None) -> None:
        self.win_len, self.win_stride = hparams["win_len"], hparams["win_stride"]

        if isinstance(start, datetime):
            start = start.timestamp()
        if isinstance(stop, datetime):
            stop = stop.timestamp()

        traj_data = self._load_traj_data(Path(path) / "trajectory", traj_track_id, start, stop)
        sensor_data = self._load_sensor_data(Path(path) / "sensor", sensor_worker_id, start, stop)

        self.time: list[torch.LongTensor] = []
        self.traj_feat: list[torch.FloatTensor] = []
        self.sensor_feat: list[torch.FloatTensor] = []
        self.map: list[tuple[int, int, int]] = []
        if len(sensor_data) / SENSOR_FREQ > hparams["min_input_len"] / hparams["freq"]:
            meas = ndimage.gaussian_filter1d(np.column_stack((linalg.norm(sensor_data[["linacc_x", "linacc_y", "linacc_z"]], axis=1), sensor_data[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]])), hparams["smooth_sd"] * SENSOR_FREQ, axis=0)

            for i, td in preprocess.seg_by_timeout(traj_data, 5):
                traj_time = np.arange(td.iloc[0]["time"], td.iloc[-1]["time"], step=1 / TRAJ_FREQ, dtype=np.float64)

                if (len(traj_time) - 2) / TRAJ_FREQ > hparams["min_input_len"] / hparams["freq"]:
                    loc = interp1d(td["time"], td[["x", "y"]], axis=0, copy=False, assume_sorted=True)(traj_time)
                    spd = ndimage.gaussian_filter1d(preprocess.loc_to_spd(loc, TRAJ_FREQ, TRAJ_RESOL), hparams["smooth_sd"] * TRAJ_FREQ)
                    ang_vel = ndimage.gaussian_filter1d(preprocess.loc_to_ang_vel(loc, TRAJ_FREQ), hparams["smooth_sd"] * TRAJ_FREQ)

                    time, spd, ang_vel, meas = preprocess.sync(traj_time[:-1] + 0.5 / TRAJ_FREQ, spd, traj_time[1:-1], ang_vel, sensor_data["time"], meas, hparams["freq"])
                    self.traj_feat.append(torch.from_numpy(np.column_stack((spd.astype(np.float32), ang_vel.astype(np.float32)))))
                    self.sensor_feat.append(torch.from_numpy(meas.astype(np.float32)))

                    valid_len = min(self.win_len, len(time))
                    win_num = max(1, (len(time) - self.win_len) // self.win_stride + 1)
                    self.time.append(torch.empty(win_num, dtype=torch.float64))
                    for j in range(win_num):
                        self.time[-1][j] = time[j * self.win_stride]
                        self.map.append((i, j, valid_len))

    @staticmethod
    def _load_traj_data(path: Path, traj_track_id: int, start: float | None, stop: float | None) -> pd.DataFrame:
        traj_data_list = []
        for f in sorted(path.glob("trajectory_????????_??_??.csv")):
            traj_data_list.append(pd.read_csv(f, usecols=("time", "track", "x", "y")))
        traj_data = pd.concat(traj_data_list, ignore_index=True)
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
            sensor_data_list.append(pd.read_csv(f, usecols=("time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "linacc_x", "linacc_y", "linacc_z")))
        sensor_data = pd.concat(sensor_data_list, ignore_index=True)
        if start is not None:
            sensor_data = sensor_data[sensor_data["time"] >= start]
        if stop is not None:
            sensor_data = sensor_data[sensor_data["time"] < stop]

        return sensor_data

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
        return (
            self.time[self.map[idx][0]][self.map[idx][1]],
            preprocess.pad(self.traj_feat[self.map[idx][0]][self.map[idx][1] * self.win_stride:self.map[idx][1] * self.win_stride + self.win_len].unsqueeze(0), self.win_len).squeeze(dim=1),
            preprocess.pad(self.sensor_feat[self.map[idx][0]][self.map[idx][1] * self.win_stride:self.map[idx][1] * self.win_stride + self.win_len].unsqueeze(0), self.win_len).squeeze(dim=1),
            torch.arange(self.win_len, dtype=torch.int32) < self.map[idx][2]
        )

    def __len__(self) -> int:
        return len(self.map)
