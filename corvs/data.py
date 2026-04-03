from argparse import Namespace
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Collection, Iterable, Literal, Optional, Self
import numpy as np
import pandas as pd
import torch
import tqdm
from numpy import linalg, random
from numpy.typing import ArrayLike, NDArray
from omegaconf import DictConfig, OmegaConf
from scipy import ndimage
from scipy.interpolate import interp1d
from torch.types import FileLike
from corvs import preprocess as preproc
from corvs import utils
from corvs.base import BaseDataset, BaseFitDataModule, BaseFitDataset, BasePredDataModule, Modality, SensorMet, TrajMet

TRAJ_FREQ = 2.5
TRAJ_RESOL = 0.01
TRAJ_TIMEOUT = 5
SENSOR_FREQ = 100

def load_traj_data(
        path: PathLike,
        track_ids: Optional[Iterable[int]] = None,
        label_ids: Optional[Iterable[int]] = None,
        start: Optional[float] = None,
        end: Optional[float] = None
    ) -> pd.DataFrame:

    all_data = []
    for p in sorted(Path(path).glob("trajectory_????????_??_??.csv")):
        data = pd.read_csv(p, usecols=("time", "track", "x", "y", "label"), dtype={"track": np.uint32, "label": np.uint32}, engine="pyarrow")
        if track_ids is not None:
            data = data[data["track"].isin(track_ids)]
        if label_ids is not None:
            data = data[data["label"].isin(label_ids)]
        if start is not None:
            data = data[data["time"] >= start]
        if end is not None:
            data = data[data["time"] < end]
        all_data.append(data)

    if len(all_data) == 0:
        all_data = pd.DataFrame(columns=("time", "track", "x", "y", "label"))
    else:
        all_data = pd.concat(all_data, ignore_index=True)

    return all_data

def load_sensor_data(path: PathLike, worker_id: int, start: Optional[float] = None, end: Optional[float] = None) -> pd.DataFrame:
    all_data = []
    for p in sorted(Path(path).glob(f"sensor_????????_??_??_{worker_id:02d}_??.csv")):
        data = pd.read_csv(p, usecols=("time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "linacc_x", "linacc_y", "linacc_z"), engine="pyarrow")
        if start is not None:
            data = data[data["time"] >= start]
        if end is not None:
            data = data[data["time"] < end]
        all_data.append(data)

    if len(all_data) == 0:
        all_data = pd.DataFrame(columns=("time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "linacc_x", "linacc_y", "linacc_z"))
    else:
        all_data = pd.concat(all_data, ignore_index=True)

    return all_data

class CorVSDataset(BaseDataset):
    traj_mets   = TrajMet.SPD, TrajMet.TURN_RATE
    sensor_mets = SensorMet.LINACC_NORM, SensorMet.ACC_X, SensorMet.ACC_Y, SensorMet.ACC_Z, SensorMet.GYRO_X, SensorMet.GYRO_Y, SensorMet.GYRO_Z

class CorVSFitDataset(CorVSDataset, BaseFitDataset):
    modalities = Modality.TRAJ_FEAT, Modality.SENSOR_FEAT, Modality.VALID_MASK, Modality.VISIBLE_MASK, Modality.LABEL

    def __init__(
            self,
            path: PathLike,
            track_ids: Collection[int],
            freq_in_hz: float,
            smooth_in_sec: float,
            win_len: int,
            win_st: int = 1,
            min_valid_len: int = 0,
            pos_factor: int = 1,
            pos_mask: Optional[float] = None,
            pos_shift_in_sec: Optional[float] = None,
            neg_ratio: int = 1,
            start: Optional[float] = None,
            end: Optional[float] = None,
            pt_path: Optional[FileLike] = None,
            seed: Optional[int] = None
        ) -> None:

        self.win_len, self.win_st = win_len, win_st

        root_path = Path(path)
        all_traj_data = load_traj_data(root_path / "trajectory", track_ids, start=start, end=end)

        self.traj_feat: list[torch.FloatTensor] = []
        self.sensor_feat: list[torch.FloatTensor] = []
        for ti in tqdm.tqdm(track_ids, desc="loading and preprocessing data", disable=len(track_ids) == 0):
            traj_data = all_traj_data[all_traj_data["track"] == ti]
            sensor_data = load_sensor_data(root_path / "sensor", traj_data["label"].iat[0], traj_data["time"].iat[0] - 1 / SENSOR_FREQ, traj_data["time"].iat[-1] + 1 / SENSOR_FREQ)

            if min_valid_len / freq_in_hz < len(sensor_data) / SENSOR_FREQ:
                meas = np.column_stack((linalg.norm(sensor_data[["linacc_x", "linacc_y", "linacc_z"]], axis=1), sensor_data[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]]))
                meas = ndimage.gaussian_filter1d(meas, smooth_in_sec * SENSOR_FREQ, axis=0)

                for _, td in preproc.seg_by_timeout(traj_data, TRAJ_TIMEOUT):
                    traj_time = np.arange(td["time"].iat[0], td["time"].iat[-1], step=1 / TRAJ_FREQ, dtype=np.float64)

                    if min_valid_len / freq_in_hz < (len(traj_time) - 2) / TRAJ_FREQ:
                        loc = interp1d(td["time"], td[["x", "y"]], axis=0, copy=False, fill_value="extrapolate", assume_sorted=True)(traj_time)
                        spd = ndimage.gaussian_filter1d(preproc.loc_to_spd(loc, TRAJ_FREQ, TRAJ_RESOL), smooth_in_sec * TRAJ_FREQ)
                        turn_rate = ndimage.gaussian_filter1d(preproc.loc_to_turn_rate(loc, TRAJ_FREQ), smooth_in_sec * TRAJ_FREQ)

                        synced_spd, synced_turn_rate, synced_meas = preproc.sync(traj_time[:-1] + 0.5 / TRAJ_FREQ, spd, traj_time[1:-1], turn_rate, sensor_data["time"], meas, freq_in_hz)[1:]
                        self.traj_feat.append(torch.from_numpy(np.column_stack((synced_spd.astype(np.float32), synced_turn_rate.astype(np.float32)))))
                        self.sensor_feat.append(torch.from_numpy(synced_meas.astype(np.float32)))

        rng = random.default_rng(seed=seed)

        pos_map = []
        max_win_num = 0
        for i, tf in enumerate(tqdm.tqdm(self.traj_feat, desc="building positive pairs", disable=len(self.traj_feat) == 0)):
            valid_len = min(self.win_len, len(tf))
            mask_len = 0 if pos_mask is None else max(0, round(pos_mask * self.win_len) - self.win_len + valid_len)
            win_num = max(1, (len(tf) - self.win_len) // self.win_st + 1)
            max_win_num = max(win_num, max_win_num)
            for j in range(win_num):
                pos_map.append((i, j, valid_len, 0, 0, 0))
                for _ in range(pos_factor - 1):
                    mask_pos = rng.integers(valid_len - mask_len, dtype=np.int32, endpoint=True)
                    if pos_shift_in_sec is None or valid_len < self.win_len:
                        shift_len = 0
                    else:
                        shift_len = round(rng.normal(scale=pos_shift_in_sec * freq_in_hz))
                        shift_len = max(-j * self.win_st, shift_len)
                        shift_len = min(shift_len, len(tf) - j * self.win_st - self.win_len)
                    pos_map.append((i, j, valid_len, mask_pos, mask_len, shift_len))
        if len(pos_map) == 0:
            self.pos_map = torch.empty(0, 6, dtype=torch.int32)
        else:
            self.pos_map: torch.CharTensor | torch.ShortTensor | torch.IntTensor | torch.LongTensor = torch.tensor(pos_map, dtype=utils.get_min_int_dtype(max_win_num))

        neg_map = []
        for i_1, j_1, vl_1 in tqdm.tqdm(self.pos_map[::pos_factor, :3], desc="building negative pairs", disable=len(self.pos_map) == 0):
            cnt = 0
            for i_2, j_2, vl_2 in rng.permutation(self.pos_map[::pos_factor, :3]):
                if i_1 != i_2 or min(vl_1, vl_2) / self.win_st < abs(j_1 - j_2):
                    neg_map.append((i_1, j_1, i_2, j_2, min(vl_1, vl_2)))
                    cnt += 1
                    if cnt >= neg_ratio:
                        break
        if len(neg_map) == 0:
            self.neg_map = torch.empty(0, 5, dtype=torch.int32)
        else:
            self.neg_map: torch.CharTensor | torch.ShortTensor | torch.IntTensor | torch.LongTensor = torch.tensor(neg_map, dtype=self.pos_map.dtype)

        if pt_path is not None:
            self.save(pt_path)
            self.load(pt_path)

    def load(self, path: FileLike, mmap: bool = True) -> None:
        self.traj_feat, self.sensor_feat, self.pos_map, self.neg_map, self.win_len, self.win_st = torch.load(path, mmap=mmap)

    def save(self, path: FileLike) -> None:
        torch.save((self.traj_feat, self.sensor_feat, self.pos_map, self.neg_map, self.win_len, self.win_st), path)

    def __getitem__(self, idx: int) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.BoolTensor, torch.BoolTensor, torch.FloatTensor]:
        time_idx = torch.arange(self.win_len, dtype=torch.int32)
        if idx < len(self.pos_map):
            return (
                preproc.pad(self.traj_feat[self.pos_map[idx, 0]][self.pos_map[idx, 1] * self.win_st:self.pos_map[idx, 1] * self.win_st + self.win_len].unsqueeze(0), self.win_len).squeeze(1),
                preproc.pad(self.sensor_feat[self.pos_map[idx, 0]][self.pos_map[idx, 1] * self.win_st + self.pos_map[idx, 5]:self.pos_map[idx, 1] * self.win_st + self.win_len + self.pos_map[idx, 5]].unsqueeze(0), self.win_len).squeeze(1),
                time_idx < self.pos_map[idx, 2],
                (time_idx < self.pos_map[idx, 3]) | (self.pos_map[idx, 3] + self.pos_map[idx, 4] <= time_idx),
                torch.ones(1, dtype=torch.float32)
            )
        else:
            idx -= len(self.pos_map)
            return (
                preproc.pad(self.traj_feat[self.neg_map[idx, 0]][self.neg_map[idx, 1] * self.win_st:self.neg_map[idx, 1] * self.win_st + self.win_len].unsqueeze(0), self.win_len).squeeze(1),
                preproc.pad(self.sensor_feat[self.neg_map[idx, 2]][self.neg_map[idx, 3] * self.win_st:self.neg_map[idx, 3] * self.win_st + self.win_len].unsqueeze(0), self.win_len).squeeze(1),
                time_idx < self.neg_map[idx, 4],
                torch.ones(self.win_len, dtype=torch.bool),
                torch.zeros(1, dtype=torch.float32)
            )

    def __len__(self) -> int:
        return len(self.pos_map) + len(self.neg_map)

    @property
    def neg_ratio(self) -> float:
        return len(self.neg_map) / len(self.pos_map)

class CorVSFitDataModule(BaseFitDataModule[CorVSFitDataset]):
    def __init__(
            self,
            path: PathLike,
            hparams: dict[str, Any] | Namespace | DictConfig,
            split_track_ids: Optional[tuple[ArrayLike, ArrayLike, ArrayLike]] = None,
            split_ratio: Optional[tuple[float, float, float]] = None,
            start: Optional[float | str | datetime] = None,
            end: Optional[float | str | datetime] = None,
            seed: Optional[int] = None
        ) -> None:

        if split_track_ids is not None and split_ratio is not None:
            raise ValueError("either split track IDs or ratio can be passed")

        super().__init__(hparams)

        self.root_path = Path(path)
        self.start = None if start is None else utils.to_unix(start, utils.JST)
        self.end = None if end is None else utils.to_unix(end, utils.JST)
        self.seed = seed

        self.track_ids: dict[Literal["train", "val", "test"], NDArray[np.uint32]] | None
        if split_track_ids is not None:
            self.track_ids = {
                "train": np.asanyarray(split_track_ids[0], dtype=np.uint32),
                "val": np.asanyarray(split_track_ids[1], dtype=np.uint32),
                "test": np.asanyarray(split_track_ids[2], dtype=np.uint32)
            }
        elif split_ratio is not None:
            traj_data = load_traj_data(self.root_path / "trajectory", start=self.start, end=self.end)
            traj_data = traj_data[traj_data["label"] < 1000]
            label_ids = preproc.rand_split(traj_data["label"].unique(), split_ratio, random.default_rng(seed=self.seed))
            self.track_ids = {
                "train": traj_data[traj_data["label"].isin(label_ids[0])]["track"].unique(),
                "val": traj_data[traj_data["label"].isin(label_ids[1])]["track"].unique(),
                "test": traj_data[traj_data["label"].isin(label_ids[2])]["track"].unique()
            }
        else:
            self.track_ids = None

    def setup(self, stage: Literal["fit", "validate", "test"]) -> None:
        match stage:
            case "fit" | "validate":
                if stage == "fit" and "train" not in self.datasets:
                    self.datasets["train"] = CorVSFitDataset(
                        self.root_path,
                        self.track_ids["train"],
                        self.hparams["freq"],
                        self.hparams["smooth"],
                        self.hparams["win_len"],
                        self.hparams["win_st"],
                        self.hparams["min_in_len"],
                        self.hparams["pos_factor"],
                        self.hparams["pos_mask"],
                        self.hparams["pos_shift"],
                        self.hparams["neg_ratio"],
                        self.start,
                        self.end,
                        Path(self.trainer.log_dir) / "train_data.pt",
                        self.seed
                    )
                if "val" not in self.datasets:
                    self.datasets["val"] = CorVSFitDataset(
                        self.root_path,
                        self.track_ids["val"],
                        self.hparams["freq"],
                        self.hparams["smooth"],
                        self.hparams["win_len"],
                        min_valid_len=self.hparams["min_in_len"],
                        start=self.start,
                        end=self.end,
                        pt_path=Path(self.trainer.log_dir) / "val_data.pt",
                        seed=self.seed
                    )
            case "test":
                if "test" not in self.datasets:
                    self.datasets["test"] = CorVSFitDataset(
                        self.root_path,
                        self.track_ids["test"],
                        self.hparams["freq"],
                        self.hparams["smooth"],
                        self.hparams["win_len"],
                        min_valid_len=self.hparams["min_in_len"],
                        start=self.start,
                        end=self.end,
                        pt_path=Path(self.trainer.log_dir) / "test_data.pt",
                        seed=self.seed
                    )

    @classmethod
    def load_from_pts(cls, path: PathLike, hparams: dict[str, Any] | Namespace | DictConfig) -> Self:
        self = cls(path, hparams)

        for m in ("train", "val", "test"):
            if (pt_path := self.root_path / f"{m}_data.pt").exists():
                dataset = CorVSFitDataset("", (), -1, -1, -1)  # create a dummy dataset
                dataset.load(pt_path)
                self.datasets[m] = dataset

        return self

    def save_split(self) -> None:
        if self.track_ids is not None:
            OmegaConf.save({m: ti.tolist() for m, ti in self.track_ids.items()}, Path(self.trainer.log_dir) / "split.yaml")

class CorVSPredDataset(CorVSDataset):
    modalities = Modality.TIME, Modality.TRAJ_FEAT, Modality.SENSOR_FEAT, Modality.VALID_MASK, Modality.LABEL

    def __init__(
            self,
            path: PathLike,
            traj_track_id: int,
            sensor_worker_id: int,
            freq_in_hz: float,
            smooth_in_sec: float,
            win_len: int,
            win_st: int = 1,
            min_valid_len: int = 0,
            start: Optional[float] = None,
            end: Optional[float] = None
        ) -> None:

        self.freq = freq_in_hz
        self.win_len, self.win_st = win_len, win_st

        root_path = Path(path)
        traj_data = load_traj_data(root_path / "trajectory", (traj_track_id, ), start=start, end=end)
        sensor_data = load_sensor_data(root_path / "sensor", sensor_worker_id, start, end)

        self.time: list[torch.DoubleTensor] = []
        self.traj_feat: list[torch.FloatTensor] = []
        self.sensor_feat: list[torch.FloatTensor] = []
        map = []
        max_win_num = 0
        if min_valid_len / self.freq < len(sensor_data) / SENSOR_FREQ:
            meas = np.column_stack((linalg.norm(sensor_data[["linacc_x", "linacc_y", "linacc_z"]], axis=1), sensor_data[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]]))
            meas = ndimage.gaussian_filter1d(meas, smooth_in_sec * SENSOR_FREQ, axis=0)

            for i, td in preproc.seg_by_timeout(traj_data, TRAJ_TIMEOUT):
                traj_time = np.arange(td["time"].iat[0], td["time"].iat[-1], step=1 / TRAJ_FREQ, dtype=np.float64)

                if min_valid_len / self.freq < (len(traj_time) - 2) / TRAJ_FREQ:
                    loc = interp1d(td["time"], td[["x", "y"]], axis=0, copy=False, fill_value="extrapolate", assume_sorted=True)(traj_time)
                    spd = ndimage.gaussian_filter1d(preproc.loc_to_spd(loc, TRAJ_FREQ, TRAJ_RESOL), smooth_in_sec * TRAJ_FREQ)
                    turn_rate = ndimage.gaussian_filter1d(preproc.loc_to_turn_rate(loc, TRAJ_FREQ), smooth_in_sec * TRAJ_FREQ)

                    synced_time, synced_spd, synced_turn_rate, synced_meas = preproc.sync(traj_time[:-1] + 0.5 / TRAJ_FREQ, spd, traj_time[1:-1], turn_rate, sensor_data["time"], meas, self.freq)
                    self.traj_feat.append(torch.from_numpy(np.column_stack((synced_spd.astype(np.float32), synced_turn_rate.astype(np.float32)))))
                    self.sensor_feat.append(torch.from_numpy(synced_meas.astype(np.float32)))

                    valid_len = min(self.win_len, len(synced_time))
                    win_num = max(1, (len(synced_time) - self.win_len) // self.win_st + 1)
                    max_win_num = max(win_num, max_win_num)
                    time = torch.empty(win_num, dtype=torch.float64)
                    for j in range(win_num):
                        time[j] = synced_time[j * self.win_st]
                        map.append((i, j, valid_len))
                    self.time.append(time)
        self.map: torch.CharTensor | torch.ShortTensor | torch.IntTensor | torch.LongTensor = torch.tensor(map, dtype=utils.get_min_int_dtype(max_win_num))

        if len(traj_data) > 0:
            self.label: torch.FloatTensor = torch.tensor(traj_data["label"].iat[0].item() == sensor_worker_id, dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple[torch.DoubleTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor, torch.FloatTensor]:
        return (
            self.time[self.map[idx][0]][self.map[idx][1]].unsqueeze(0),
            preproc.pad(self.traj_feat[self.map[idx][0]][self.map[idx][1] * self.win_st:self.map[idx][1] * self.win_st + self.win_len].unsqueeze(0), self.win_len).squeeze(1),
            preproc.pad(self.sensor_feat[self.map[idx][0]][self.map[idx][1] * self.win_st:self.map[idx][1] * self.win_st + self.win_len].unsqueeze(0), self.win_len).squeeze(1),
            torch.arange(self.win_len, dtype=torch.int32) < self.map[idx][2],
            self.label.unsqueeze(0)
        )

    def __len__(self) -> int:
        return len(self.map)

    @property
    def tot_time_in_sec(self) -> float:
        tot_time = 0
        for tf in self.traj_feat:
            tot_time += len(tf) / self.freq
        return tot_time

class CorVSPredDataModule(BasePredDataModule[CorVSPredDataset]):
    def __init__(
            self,
            path: PathLike,
            traj_track_id: int,
            sensor_worker_id: int,
            hparams: dict[str, Any] | Namespace | DictConfig,
            start: Optional[float | str | datetime] = None,
            end: Optional[float | str | datetime] = None
        ) -> None:

        super().__init__(hparams)

        self.root_path = Path(path)
        self.track_id, self.worker_id = traj_track_id, sensor_worker_id
        self.start = None if start is None else utils.to_unix(start, utils.JST)
        self.end = None if end is None else utils.to_unix(end, utils.JST)

    def setup(self, stage: Literal["predict"]) -> None:
        match stage:
            case "predict":
                if "pred" not in self.datasets:
                    self.datasets["pred"] = CorVSPredDataset(
                        self.root_path,
                        self.track_id,
                        self.worker_id,
                        self.hparams["freq"],
                        self.hparams["smooth"],
                        self.hparams["win_len"],
                        min_valid_len=self.hparams["min_in_len"],
                        start=self.start,
                        end=self.end
                    )
