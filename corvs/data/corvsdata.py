import pathlib
from argparse import Namespace
from datetime import datetime
from os import PathLike
from typing import Any, Literal, Optional, Self
import numpy as np
import torch
from numpy import linalg
from omegaconf import DictConfig
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from rich import progress
from scipy import ndimage
from scipy.interpolate import interp1d
from torch.types import FileLike
from corvs import utils
from corvs.data import load
from corvs.data import preprocess as preproc
from corvs.data.base import BaseDataHParams, BaseDataset, BasePredDataModule
from corvs.enums import Modality, Mode, SensorMet, TrajMet

TIME_ZONE    = "Asia/Tokyo"
TRAJ_FREQ    = 2.5
TRAJ_RESOL   = 0.01
TRAJ_TIMEOUT = 5
SENSOR_FREQ  = 100

class CorVSDataset(BaseDataset):
    traj_mets = TrajMet.SPD, TrajMet.TURN_RATE

class CorVSDataHParams(BaseDataHParams):
    freq:       PositiveFloat
    denoise:    NonNegativeFloat
    win_len:    PositiveInt
    win_st:     PositiveInt
    min_in_len: NonNegativeInt


class CorVSPredDataset(CorVSDataset):
    sensor_mets = SensorMet.ACC_X, SensorMet.ACC_Y, SensorMet.ACC_Z, SensorMet.GYRO_X, SensorMet.GYRO_Y, SensorMet.GYRO_Z, SensorMet.LINACC_NORM

    def __init__(
        self,
        root_path: str | PathLike[str],
        freq_in_hz: float,
        denoise_in_sec: float,
        win_len: int,
        win_st: int = 1,
        min_valid_len: int = 0,
        start: Optional[float] = None,
        end: Optional[float] = None,
        pt_path: Optional[FileLike] = None
    ) -> None:

        self.win_len, self.win_st = win_len, win_st

        root_path = pathlib.Path(root_path)
        all_traj_data = load.load_traj_data(root_path / "trajectory/", start=start, end=end)
        all_sensor_data = load.iter_all_sensor_data(
            root_path / "sensor/",
            ("time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "linacc_x", "linacc_y", "linacc_z"),
            all_traj_data["time"].iat[0] - 1 / SENSOR_FREQ,
            all_traj_data["time"].iat[-1] + 1 / SENSOR_FREQ
        )

        self.time: list[torch.DoubleTensor] = []
        self.traj_feat: list[torch.FloatTensor] = []
        self.sensor_feat: list[torch.FloatTensor] = []
        track_id, worker_id = [], []
        map = []
        max_win_num = 0
        i = 0
        for wi, sd in progress.track(all_sensor_data, description="Loading and preprocessing data"):
            if min_valid_len / freq_in_hz < (len(sd) - 1) / SENSOR_FREQ:
                meas = np.column_stack((sd[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]], linalg.norm(sd[["linacc_x", "linacc_y", "linacc_z"]], axis=1)))
                meas = ndimage.gaussian_filter1d(meas, denoise_in_sec * SENSOR_FREQ, axis=0)

                for ti in all_traj_data["track"].unique():
                    traj_data = all_traj_data[all_traj_data["track"] == ti]

                    for _, td in preproc.seg_by_timeout(traj_data, TRAJ_TIMEOUT):
                        traj_time = np.arange(td["time"].iat[0], td["time"].iat[-1], step=1 / TRAJ_FREQ, dtype=np.float64)

                        if min_valid_len / freq_in_hz < (len(traj_time) - 3) / TRAJ_FREQ:
                            loc = interp1d(td["time"], td[["x", "y"]], axis=0, copy=False, fill_value="extrapolate", assume_sorted=True)(traj_time)
                            spd = ndimage.gaussian_filter1d(preproc.loc_to_spd(loc, TRAJ_FREQ, TRAJ_RESOL), denoise_in_sec * TRAJ_FREQ)
                            turn_rate = ndimage.gaussian_filter1d(preproc.loc_to_turn_rate(loc, TRAJ_FREQ), denoise_in_sec * TRAJ_FREQ)

                            synced_time, synced_spd, synced_turn_rate, synced_meas = preproc.sync(traj_time[:-1] + 0.5 / TRAJ_FREQ, spd, traj_time[1:-1], turn_rate, sd["time"], meas, freq_in_hz)

                            win_num = max(1, (len(synced_time) - self.win_len) // self.win_st + 1)
                            self.time.append(torch.from_numpy(synced_time[:win_num * self.win_st:self.win_st]))
                            self.traj_feat.append(torch.from_numpy(np.column_stack((synced_spd.astype(np.float32), synced_turn_rate.astype(np.float32)))))
                            self.sensor_feat.append(torch.from_numpy(synced_meas.astype(np.float32)))

                            track_id.append(ti)
                            worker_id.append(wi)

                            valid_len = min(self.win_len, len(synced_time))
                            for j in range(win_num):
                                map.append((i, j, valid_len))

                            max_win_num = max(win_num, max_win_num)
                            i += 1

        self.track_id: torch.IntTensor = torch.tensor(track_id, dtype=torch.int32)
        self.worker_id: torch.IntTensor = torch.tensor(worker_id, dtype=torch.int32)

        self.map: torch.CharTensor | torch.ShortTensor | torch.IntTensor | torch.LongTensor
        if len(map) == 0:
            self.map = torch.empty(0, 3, dtype=torch.int32)
        else:
            self.map = torch.tensor(map, dtype=utils.get_min_int_dtype(max(len(self.time), max_win_num, self.win_len)))

        if pt_path is not None:
            self.save(pt_path)
            self.load(pt_path)

    def save(self, path: FileLike) -> None:
        torch.save((self.time, self.track_id, self.worker_id, self.traj_feat, self.sensor_feat, self.map, self.win_len, self.win_st), path)

    def load(self, path: FileLike, mmap: bool = True) -> None:
        self.time, self.track_id, self.worker_id, self.traj_feat, self.sensor_feat, self.map, self.win_len, self.win_st = torch.load(path, mmap=mmap)

    def __getitem__(self, idx: int) -> dict[Modality, torch.DoubleTensor | torch.IntTensor | torch.FloatTensor | torch.BoolTensor]:
        return {
            Modality.TIME: self.time[self.map[idx][0]][self.map[idx][1]].unsqueeze(0),
            Modality.TRACK_ID: self.track_id[self.map[idx][0]].unsqueeze(0),
            Modality.WORKER_ID: self.worker_id[self.map[idx][0]].unsqueeze(0),
            Modality.TRAJ_FEAT: preproc.pad(self.traj_feat[self.map[idx][0]][self.map[idx][1] * self.win_st:self.map[idx][1] * self.win_st + self.win_len].unsqueeze(0), self.win_len).squeeze(1),
            Modality.SENSOR_FEAT: preproc.pad(self.sensor_feat[self.map[idx][0]][self.map[idx][1] * self.win_st:self.map[idx][1] * self.win_st + self.win_len].unsqueeze(0), self.win_len).squeeze(1),
            Modality.VALID_MASK: torch.arange(self.win_len, dtype=torch.int32) < self.map[idx][2]
        }

    def __len__(self) -> int:
        return len(self.map)

    @classmethod
    def from_pt(cls, path: FileLike) -> Self:
        self = cls.__new__(cls)
        self.load(path)
        return self

class CorVSPredDataModule(BasePredDataModule[CorVSPredDataset, CorVSDataHParams]):
    def __init__(
            self,
            root_path: str | PathLike[str],
            hparams: dict[str, Any] | Namespace | DictConfig,
            start: Optional[float | str | datetime] = None,
            end: Optional[float | str | datetime] = None,
            pts_path: Optional[str | PathLike[str]] = None
        ) -> None:

        super().__init__(hparams)

        self.root_path = pathlib.Path(root_path)
        self.start = None if start is None else utils.to_unix(start, TIME_ZONE)
        self.end = None if end is None else utils.to_unix(end, TIME_ZONE)
        self.pts_path = None if pts_path is None else pathlib.Path(pts_path)

    def prepare_data(self) -> None:
        """
        Build datasets and save them to pt files.
        This method will be called only in zero-rank process.
        """

        if self.pts_path is not None:
            self.pts_path.mkdir(parents=True, exist_ok=True)

        if self.pts_path is None or not (self.pts_path / "pred_data.pt").exists():
            self.datasets[Mode.PRED] = CorVSPredDataset(
                self.root_path,
                self.hparams.freq,
                self.hparams.denoise,
                self.hparams.win_len,
                min_valid_len=self.hparams.min_in_len,
                start=self.start,
                end=self.end,
                pt_path=self.pts_path / "pred_data.pt"
            )

    def setup(self, stage: Literal["predict"]) -> None:
        """
        Load datasets from pt files.
        This method will be called in every process.

        Parameters
        ----------
        stage : 'predict'
            Stage to setup.
        """

        if Mode.PRED not in self.datasets:
            self.datasets[Mode.PRED] = CorVSPredDataset.from_pt(self.pts_path / "pred_data.pt")


class CorVSDemoDataset(CorVSDataset):
    sensor_mets = SensorMet.ACC_X, SensorMet.ACC_Y, SensorMet.ACC_Z, SensorMet.GYRO_X, SensorMet.GYRO_Y, SensorMet.GYRO_Z, SensorMet.LINACC_NORM

    def __init__(
            self,
            root_path: str | PathLike[str],
            traj_track_id: int,
            sensor_worker_id: int,
            freq_in_hz: float,
            denoise_in_sec: float,
            win_len: int,
            win_st: int = 1,
            min_valid_len: int = 0,
            start: Optional[float] = None,
            end: Optional[float] = None
        ) -> None:

        self.track_id: torch.IntTensor = torch.tensor((traj_track_id, ), dtype=torch.int32)
        self.worker_id: torch.IntTensor = torch.tensor((sensor_worker_id, ), dtype=torch.int32)

        self.freq = freq_in_hz
        self.win_len, self.win_st = win_len, win_st

        root_path = pathlib.Path(root_path)
        traj_data = load.load_traj_data(root_path / "trajectory/", (self.track_id.item(), ), start=start, end=end)
        sensor_data = load.load_sensor_data(
            root_path / "sensor/",
            self.worker_id.item(),
            ("time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "linacc_x", "linacc_y", "linacc_z"),
            start,
            end
        )

        self.time: list[torch.DoubleTensor] = []
        self.traj_feat: list[torch.FloatTensor] = []
        self.sensor_feat: list[torch.FloatTensor] = []
        map = []
        max_win_num = 0
        if min_valid_len / self.freq < (len(sensor_data) - 1) / SENSOR_FREQ:
            meas = np.column_stack((sensor_data[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]], linalg.norm(sensor_data[["linacc_x", "linacc_y", "linacc_z"]], axis=1)))
            meas = ndimage.gaussian_filter1d(meas, denoise_in_sec * SENSOR_FREQ, axis=0)

            i = 0
            for _, td in preproc.seg_by_timeout(traj_data, TRAJ_TIMEOUT):
                traj_time = np.arange(td["time"].iat[0], td["time"].iat[-1], step=1 / TRAJ_FREQ, dtype=np.float64)

                if min_valid_len / self.freq < (len(traj_time) - 3) / TRAJ_FREQ:
                    loc = interp1d(td["time"], td[["x", "y"]], axis=0, copy=False, fill_value="extrapolate", assume_sorted=True)(traj_time)
                    spd = ndimage.gaussian_filter1d(preproc.loc_to_spd(loc, TRAJ_FREQ, TRAJ_RESOL), denoise_in_sec * TRAJ_FREQ)
                    turn_rate = ndimage.gaussian_filter1d(preproc.loc_to_turn_rate(loc, TRAJ_FREQ), denoise_in_sec * TRAJ_FREQ)

                    synced_time, synced_spd, synced_turn_rate, synced_meas = preproc.sync(traj_time[:-1] + 0.5 / TRAJ_FREQ, spd, traj_time[1:-1], turn_rate, sensor_data["time"], meas, self.freq)

                    win_num = max(1, (len(synced_time) - self.win_len) // self.win_st + 1)
                    self.time.append(torch.from_numpy(synced_time[:win_num * self.win_st:self.win_st]))
                    self.traj_feat.append(torch.from_numpy(np.column_stack((synced_spd.astype(np.float32), synced_turn_rate.astype(np.float32)))))
                    self.sensor_feat.append(torch.from_numpy(synced_meas.astype(np.float32)))

                    valid_len = min(self.win_len, len(synced_time))
                    for j in range(win_num):
                        map.append((i, j, valid_len))

                    max_win_num = max(win_num, max_win_num)
                    i += 1

        self.map: torch.CharTensor | torch.ShortTensor | torch.IntTensor | torch.LongTensor
        if len(map) == 0:
            self.map = torch.empty(0, 3, dtype=torch.int32)
        else:
            self.map = torch.tensor(map, dtype=utils.get_min_int_dtype(max(len(self.time), max_win_num, self.win_len)))

        self.label: torch.FloatTensor | None
        if len(traj_data) == 0:
            self.label = None
        else:
            self.label = (self.worker_id == traj_data["label"].iat[0].item()).float()

    def __getitem__(self, idx: int) -> dict[Modality, torch.DoubleTensor | torch.IntTensor | torch.FloatTensor | torch.BoolTensor]:
        return {
            Modality.TIME: self.time[self.map[idx][0]][self.map[idx][1]].unsqueeze(0),
            Modality.TRACK_ID: self.track_id,
            Modality.WORKER_ID: self.worker_id,
            Modality.TRAJ_FEAT: preproc.pad(self.traj_feat[self.map[idx][0]][self.map[idx][1] * self.win_st:self.map[idx][1] * self.win_st + self.win_len].unsqueeze(0), self.win_len).squeeze(1),
            Modality.SENSOR_FEAT: preproc.pad(self.sensor_feat[self.map[idx][0]][self.map[idx][1] * self.win_st:self.map[idx][1] * self.win_st + self.win_len].unsqueeze(0), self.win_len).squeeze(1),
            Modality.VALID_MASK: torch.arange(self.win_len, dtype=torch.int32) < self.map[idx][2],
            Modality.LABEL: self.label
        }

    def __len__(self) -> int:
        return len(self.map)

    @property
    def tot_time_in_sec(self) -> float:
        tot_time = 0
        for tf in self.traj_feat:
            tot_time += len(tf) / self.freq
        return tot_time

class CorVSDemoDataModule(BasePredDataModule[CorVSDemoDataset, CorVSDataHParams]):
    def __init__(
            self,
            root_path: str | PathLike[str],
            traj_track_id: int,
            sensor_worker_id: int,
            hparams: dict[str, Any] | Namespace | DictConfig,
            start: Optional[float | str | datetime] = None,
            end: Optional[float | str | datetime] = None
        ) -> None:

        super().__init__(hparams)

        self.root_path = pathlib.Path(root_path)
        self.track_id, self.worker_id = traj_track_id, sensor_worker_id
        self.start = None if start is None else utils.to_unix(start, TIME_ZONE)
        self.end = None if end is None else utils.to_unix(end, TIME_ZONE)

    def setup(self, stage: Literal["predict"]) -> None:
        if Mode.PRED not in self.datasets:
            self.datasets[Mode.PRED] = CorVSDemoDataset(
                self.root_path,
                self.track_id,
                self.worker_id,
                self.hparams.freq,
                self.hparams.denoise,
                self.hparams.win_len,
                min_valid_len=self.hparams.min_in_len,
                start=self.start,
                end=self.end
            )
