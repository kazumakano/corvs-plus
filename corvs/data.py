from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Collection, Iterable, Literal, Optional
import numpy as np
import pandas as pd
import torch
import tqdm
from lightning import pytorch as L
from numpy import linalg, random
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from scipy import ndimage
from scipy.interpolate import interp1d
from torch.types import FileLike
from torch.utils import data
from corvs import preprocess, utils
from corvs.base import BaseDataset, BaseFitDataset, DataItem, SensorFeat, TrajFeat

TRAJ_FREQ = 2.5
TRAJ_RESOL = 0.01
SENSOR_FREQ = 100

def load_traj_data(path: PathLike, track_ids: Optional[Iterable[int]] = None, label_ids: Optional[Iterable[int]] = None, start: Optional[float] = None, stop: Optional[float] = None) -> pd.DataFrame:
    all_data = []
    for p in sorted(Path(path).glob("trajectory_????????_??_??.csv")):
        data = pd.read_csv(
            p,
            usecols=("time", "track", "x", "y", "label"),
            dtype={"track": np.uint32, "label": np.uint32},
            engine="pyarrow"
        )
        if track_ids is not None:
            data = data[data["track"].isin(track_ids)]
        if label_ids is not None:
            data = data[data["label"].isin(label_ids)]
        if start is not None:
            data = data[data["time"] >= start]
        if stop is not None:
            data = data[data["time"] < stop]
        all_data.append(data)

    if len(all_data) > 0:
        all_data = pd.concat(all_data, ignore_index=True)
    else:
        all_data = pd.DataFrame(columns=("time", "track", "x", "y", "label"))

    return all_data

def load_sensor_data(path: PathLike, worker_id: int, start: Optional[float] = None, stop: Optional[float] = None) -> pd.DataFrame:
    all_data = []
    for p in sorted(Path(path).glob(f"sensor_????????_??_??_{worker_id:02d}_??.csv")):
        data = pd.read_csv(p, usecols=("time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "linacc_x", "linacc_y", "linacc_z"), engine="pyarrow")
        if start is not None:
            data = data[data["time"] >= start]
        if stop is not None:
            data = data[data["time"] < stop]
        all_data.append(data)

    if len(all_data) > 0:
        all_data = pd.concat(all_data, ignore_index=True)
    else:
        all_data = pd.DataFrame(columns=("time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "linacc_x", "linacc_y", "linacc_z"))

    return all_data

class CorVSDataset(BaseDataset):
    traj_feats   = TrajFeat.SPD, TrajFeat.TURN_RATE
    sensor_feats = SensorFeat.LINACC_NORM, SensorFeat.ACC_X, SensorFeat.ACC_Y, SensorFeat.ACC_Z, SensorFeat.GYRO_X, SensorFeat.GYRO_Y, SensorFeat.GYRO_Z

class CorVSFitDataset(CorVSDataset, BaseFitDataset):
    items = DataItem.TRAJ_FEAT, DataItem.SENSOR_FEAT, DataItem.VALID_MASK, DataItem.VISIBLE_MASK, DataItem.LABEL

    def __init__(
            self,
            root_path: PathLike,
            cache_path: FileLike,
            traj_track_ids: Collection[int],
            freq_in_hz: float,
            smooth_in_sec: float,
            min_input_len: int,
            win_len: int,
            win_stride: int = 1,
            pos_factor: int = 1,
            pos_mask: Optional[float] = None,
            pos_shift_in_sec: Optional[float] = None,
            neg_ratio: int = 1,
            start: Optional[float] = None,
            stop: Optional[float] = None,
            seed: Optional[int] = None
        ) -> None:
        self.freq = freq_in_hz
        self.win_len, self.win_stride = win_len, win_stride

        root_path = Path(root_path)
        all_traj_data = load_traj_data(root_path / "trajectory", traj_track_ids, start=start, stop=stop)

        self.traj_feat: list[torch.FloatTensor] = []
        self.sensor_feat: list[torch.FloatTensor] = []
        for ti in tqdm.tqdm(traj_track_ids, desc="loading and preprocessing data"):
            traj_data = all_traj_data[all_traj_data["track"] == ti]
            sensor_data = load_sensor_data(root_path / "sensor", traj_data["label"].iat[0], traj_data["time"].iat[0] - 1 / SENSOR_FREQ, traj_data["time"].iat[-1] + 1 / SENSOR_FREQ)

            if len(sensor_data) / SENSOR_FREQ > min_input_len / self.freq:
                meas = ndimage.gaussian_filter1d(np.column_stack((linalg.norm(sensor_data[["linacc_x", "linacc_y", "linacc_z"]], axis=1), sensor_data[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]])), smooth_in_sec * SENSOR_FREQ, axis=0)

                for _, td in preprocess.seg_by_timeout(traj_data, 5):
                    traj_time = np.arange(td["time"].iat[0], td["time"].iat[-1], step=1 / TRAJ_FREQ, dtype=np.float64)

                    if (len(traj_time) - 2) / TRAJ_FREQ > min_input_len / self.freq:
                        loc = interp1d(td["time"], td[["x", "y"]], axis=0, copy=False, fill_value="extrapolate", assume_sorted=True)(traj_time)
                        spd = ndimage.gaussian_filter1d(preprocess.loc_to_spd(loc, TRAJ_FREQ, TRAJ_RESOL), smooth_in_sec * TRAJ_FREQ)
                        turn_rate = ndimage.gaussian_filter1d(preprocess.loc_to_turn_rate(loc, TRAJ_FREQ), smooth_in_sec * TRAJ_FREQ)

                        synced_spd, synced_turn_rate, synced_meas = preprocess.sync(traj_time[:-1] + 0.5 / TRAJ_FREQ, spd, traj_time[1:-1], turn_rate, sensor_data["time"], meas, self.freq)[1:]
                        self.traj_feat.append(torch.from_numpy(np.column_stack((synced_spd.astype(np.float32), synced_turn_rate.astype(np.float32)))))
                        self.sensor_feat.append(torch.from_numpy(synced_meas.astype(np.float32)))

        rng = random.default_rng(seed=seed)

        pos_map = []
        max_win_num = 0
        for i, tf in enumerate(tqdm.tqdm(self.traj_feat, desc="building positive pairs")):
            valid_len = min(self.win_len, len(tf))
            mask_len = 0 if pos_mask is None else max(0, round(pos_mask * self.win_len) - self.win_len + valid_len)
            win_num = max(1, (len(tf) - self.win_len) // self.win_stride + 1)
            max_win_num = max(win_num, max_win_num)
            for j in range(win_num):
                pos_map.append((i, j, valid_len, 0, 0, 0))
                for _ in range(pos_factor - 1):
                    mask_pos = rng.integers(valid_len - mask_len, dtype=np.int32, endpoint=True)
                    if pos_shift_in_sec is None or valid_len < self.win_len:
                        shift_len = 0
                    else:
                        shift_len = round(rng.normal(scale=pos_shift_in_sec * self.freq))
                        shift_len = max(-j * self.win_stride, shift_len)
                        shift_len = min(shift_len, len(tf) - j * self.win_stride - self.win_len)
                    pos_map.append((i, j, valid_len, mask_pos, mask_len, shift_len))
        self.pos_map: torch.CharTensor | torch.ShortTensor | torch.IntTensor | torch.LongTensor = torch.tensor(pos_map, dtype=utils.get_min_int_dtype(max_win_num))

        neg_map = []
        for i_1, j_1, vl_1 in tqdm.tqdm(self.pos_map[::pos_factor, :3], desc="building negative pairs"):
            cnt = 0
            for i_2, j_2, vl_2 in rng.permutation(self.pos_map[::pos_factor, :3]):
                if i_1 != i_2 or abs(j_1 - j_2) > vl_1 / self.win_stride:
                    neg_map.append((i_1, j_1, i_2, j_2, min(vl_1, vl_2)))
                    cnt += 1
                    if cnt >= neg_ratio:
                        break
        self.neg_map: torch.CharTensor | torch.ShortTensor | torch.IntTensor | torch.LongTensor = torch.tensor(neg_map, dtype=self.pos_map.dtype)

        self.cache(cache_path)

    def cache(self, path: FileLike) -> None:
        torch.save((self.traj_feat, self.sensor_feat, self.pos_map, self.neg_map), path)
        self.traj_feat, self.sensor_feat, self.pos_map, self.neg_map = torch.load(path, mmap=True)

    def __getitem__(self, idx: int) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.BoolTensor, torch.BoolTensor, torch.FloatTensor]:
        time_idx = torch.arange(self.win_len, dtype=torch.int32)
        if idx < len(self.pos_map):
            return (
                preprocess.pad(self.traj_feat[self.pos_map[idx, 0]][self.pos_map[idx, 1] * self.win_stride:self.pos_map[idx, 1] * self.win_stride + self.win_len].unsqueeze(0), self.win_len).squeeze(dim=1),
                preprocess.pad(self.sensor_feat[self.pos_map[idx, 0]][self.pos_map[idx, 1] * self.win_stride + self.pos_map[idx, 5]:self.pos_map[idx, 1] * self.win_stride + self.win_len + self.pos_map[idx, 5]].unsqueeze(0), self.win_len).squeeze(dim=1),
                time_idx < self.pos_map[idx, 2],
                (time_idx < self.pos_map[idx, 3]) | (self.pos_map[idx, 3] + self.pos_map[idx, 4] <= time_idx),
                torch.ones(1, dtype=torch.float32)
            )
        else:
            idx -= len(self.pos_map)
            return (
                preprocess.pad(self.traj_feat[self.neg_map[idx, 0]][self.neg_map[idx, 1] * self.win_stride:self.neg_map[idx, 1] * self.win_stride + self.win_len].unsqueeze(0), self.win_len).squeeze(dim=1),
                preprocess.pad(self.sensor_feat[self.neg_map[idx, 2]][self.neg_map[idx, 3] * self.win_stride:self.neg_map[idx, 3] * self.win_stride + self.win_len].unsqueeze(0), self.win_len).squeeze(dim=1),
                time_idx < self.neg_map[idx, 4],
                torch.ones(self.win_len, dtype=torch.bool),
                torch.zeros(1, dtype=torch.float32)
            )

    def __len__(self) -> int:
        return len(self.pos_map) + len(self.neg_map)

    @property
    def neg_ratio(self) -> float:
        return len(self.neg_map) / len(self.pos_map)

class CorVSFitDataModule(L.LightningDataModule):
    def __init__(
            self,
            path: PathLike,
            hparams: dict[str, Any] | DictConfig,
            split_ratio: tuple[float, float, float] = (0.8, 0.2, 0),
            start: Optional[float | str | datetime] = None,
            stop: Optional[float | str | datetime] = None,
            seed: Optional[int] = None
        ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.datasets: dict[Literal["train", "val", "test"], CorVSFitDataset] = {}
        self.root_path = Path(path)
        self.seed = seed

        self.start = utils.to_unix(start, utils.jst)
        self.stop = utils.to_unix(stop, utils.jst)

        traj_data = load_traj_data(self.root_path / "trajectory", start=self.start, stop=self.stop)
        label = preprocess.rand_split(traj_data["label"].unique(), split_ratio, random.default_rng(seed=self.seed))
        self.track_ids: dict[Literal["train", "val", "test"], NDArray[np.uint32]] = {
            "train": traj_data[traj_data["label"].isin(label[0])]["track"].unique(),
            "val": traj_data[traj_data["label"].isin(label[1])]["track"].unique(),
            "test": traj_data[traj_data["label"].isin(label[2])]["track"].unique()
        }

    def setup(self, stage: Literal["fit", "validate", "test"]) -> None:
        match stage:
            case "fit" | "validate":
                if stage == "fit" and "train" not in self.datasets.keys():
                    self.datasets["train"] = CorVSFitDataset(
                        self.root_path,
                        Path(self.trainer.log_dir) / "train_data.pt",
                        self.track_ids["train"],
                        self.hparams["freq"],
                        self.hparams["smooth"],
                        self.hparams["min_input_len"],
                        self.hparams["win_len"],
                        self.hparams["win_stride"],
                        self.hparams["pos_factor"],
                        self.hparams["pos_mask"],
                        self.hparams["pos_shift"],
                        self.hparams["neg_ratio"],
                        self.start,
                        self.stop,
                        self.seed
                    )
                if "val" not in self.datasets.keys():
                    self.datasets["val"] = CorVSFitDataset(
                        self.root_path,
                        Path(self.trainer.log_dir) / "val_data.pt",
                        self.track_ids["val"],
                        self.hparams["freq"],
                        self.hparams["smooth"],
                        self.hparams["min_input_len"],
                        self.hparams["win_len"],
                        start=self.start,
                        stop=self.stop,
                        seed=self.seed
                    )
            case "test":
                if "test" not in self.datasets.keys():
                    self.datasets["test"] = CorVSFitDataset(
                        self.root_path,
                        Path(self.trainer.log_dir) / "test_data.pt",
                        self.track_ids["test"],
                        self.hparams["freq"],
                        self.hparams["smooth"],
                        self.hparams["min_input_len"],
                        self.hparams["win_len"],
                        start=self.start,
                        stop=self.stop,
                        seed=self.seed
                    )

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.datasets["train"], batch_size=self.hparams["batch_size"], shuffle=True, num_workers=self.hparams["num_workers"], pin_memory=True, drop_last=True, persistent_workers=True)

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.datasets["val"], batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"], pin_memory=True, persistent_workers=True)

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.datasets["test"], batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"], pin_memory=True)

    def save_split(self) -> None:
        OmegaConf.save({k: v.tolist() for k, v in self.track_ids.items()}, Path(self.trainer.log_dir) / "split.yaml")

class CorVSPredictDataset(CorVSDataset):
    items = DataItem.TIME, DataItem.TRAJ_FEAT, DataItem.SENSOR_FEAT, DataItem.VALID_MASK, DataItem.LABEL

    def __init__(
            self,
            path: PathLike,
            traj_track_id: int,
            sensor_worker_id: int,
            freq_in_hz: float,
            smooth_in_sec: float,
            min_input_len: int,
            win_len: int,
            win_stride: int = 1,
            start: Optional[float] = None,
            stop: Optional[float] = None
        ) -> None:
        self.freq = freq_in_hz
        self.win_len, self.win_stride = win_len, win_stride

        path = Path(path)
        traj_data = load_traj_data(path / "trajectory", (traj_track_id, ), start=start, stop=stop)
        sensor_data = load_sensor_data(path / "sensor", sensor_worker_id, start, stop)

        self.time: list[torch.DoubleTensor] = []
        self.traj_feat: list[torch.FloatTensor] = []
        self.sensor_feat: list[torch.FloatTensor] = []
        map = []
        max_win_num = 0
        if len(sensor_data) / SENSOR_FREQ > min_input_len / self.freq:
            meas = ndimage.gaussian_filter1d(np.column_stack((linalg.norm(sensor_data[["linacc_x", "linacc_y", "linacc_z"]], axis=1), sensor_data[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]])), smooth_in_sec * SENSOR_FREQ, axis=0)

            for i, td in preprocess.seg_by_timeout(traj_data, 5):
                traj_time = np.arange(td["time"].iat[0], td["time"].iat[-1], step=1 / TRAJ_FREQ, dtype=np.float64)

                if (len(traj_time) - 2) / TRAJ_FREQ > min_input_len / self.freq:
                    loc = interp1d(td["time"], td[["x", "y"]], axis=0, copy=False, fill_value="extrapolate", assume_sorted=True)(traj_time)
                    spd = ndimage.gaussian_filter1d(preprocess.loc_to_spd(loc, TRAJ_FREQ, TRAJ_RESOL), smooth_in_sec * TRAJ_FREQ)
                    turn_rate = ndimage.gaussian_filter1d(preprocess.loc_to_turn_rate(loc, TRAJ_FREQ), smooth_in_sec * TRAJ_FREQ)

                    synced_time, synced_spd, synced_turn_rate, synced_meas = preprocess.sync(traj_time[:-1] + 0.5 / TRAJ_FREQ, spd, traj_time[1:-1], turn_rate, sensor_data["time"], meas, self.freq)
                    self.traj_feat.append(torch.from_numpy(np.column_stack((synced_spd.astype(np.float32), synced_turn_rate.astype(np.float32)))))
                    self.sensor_feat.append(torch.from_numpy(synced_meas.astype(np.float32)))

                    valid_len = min(self.win_len, len(synced_time))
                    win_num = max(1, (len(synced_time) - self.win_len) // self.win_stride + 1)
                    max_win_num = max(win_num, max_win_num)
                    self.time.append(torch.empty(win_num, dtype=torch.float64))
                    for j in range(win_num):
                        self.time[-1][j] = synced_time[j * self.win_stride]
                        map.append((i, j, valid_len))
        self.map: torch.CharTensor | torch.ShortTensor | torch.IntTensor | torch.LongTensor = torch.tensor(map, dtype=utils.get_min_int_dtype(max_win_num))

        if len(traj_data) > 0:
            self.label: torch.FloatTensor = torch.tensor(traj_data["label"].iat[0].item() == sensor_worker_id, dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple[torch.DoubleTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor, torch.FloatTensor]:
        return (
            self.time[self.map[idx][0]][self.map[idx][1]].unsqueeze(0),
            preprocess.pad(self.traj_feat[self.map[idx][0]][self.map[idx][1] * self.win_stride:self.map[idx][1] * self.win_stride + self.win_len].unsqueeze(0), self.win_len).squeeze(dim=1),
            preprocess.pad(self.sensor_feat[self.map[idx][0]][self.map[idx][1] * self.win_stride:self.map[idx][1] * self.win_stride + self.win_len].unsqueeze(0), self.win_len).squeeze(dim=1),
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

class CorVSPredictDataModule(L.LightningDataModule):
    def __init__(self, path: PathLike, traj_track_id: int, sensor_worker_id: int, hparams: dict[str, Any] | DictConfig, start: Optional[float | str | datetime] = None, stop: Optional[float | str | datetime] = None) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.datasets: dict[Literal["pred"], CorVSPredictDataset] = {}
        self.root_path = path
        self.track_id, self.worker_id = traj_track_id, sensor_worker_id

        self.start = utils.to_unix(start, utils.jst)
        self.stop = utils.to_unix(stop, utils.jst)

    def setup(self, stage: Literal["predict"]) -> None:
        match stage:
            case "predict":
                if "pred" not in self.datasets.keys():
                    self.datasets["pred"] = CorVSPredictDataset(
                        self.root_path,
                        self.track_id,
                        self.worker_id,
                        self.hparams["freq"],
                        self.hparams["smooth"],
                        self.hparams["min_input_len"],
                        self.hparams["win_len"],
                        start=self.start,
                        stop=self.stop
                    )

    def predict_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.datasets["pred"], batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"], pin_memory=True)
