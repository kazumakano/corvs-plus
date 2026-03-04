from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Optional
import pandas as pd
from lightning import pytorch as L
from torch.utils import data


class CorVSDataset(data.Dataset):
    ...

class CorVSDataModule(L.LightningDataModule):
    ...

class CorVSPredictDataset(data.Dataset):
    def __init__(self, path: PathLike, traj_track_id: int, sensor_worker_id: int, start: Optional[float | datetime] = None, stop: Optional[float | datetime] = None) -> None:
        if isinstance(start, datetime):
            start = start.timestamp()
        if isinstance(stop, datetime):
            stop = stop.timestamp()

        # load trajectory data
        traj_data_list = []
        for f in sorted((Path(path) / "trajectory").glob("trajectory_????????_??_??.csv")):
            traj_data_list.append(pd.read_csv(f, usecols=("time", "track", "x", "y")))
        traj_data = pd.concat(traj_data_list, ignore_index=True)
        traj_data = traj_data[traj_data["track"] == traj_track_id]
        if start is not None:
            traj_data = traj_data[traj_data["time"] >= start]
        if stop is not None:
            traj_data = traj_data[traj_data["time"] < stop]

        # load sensor data
        sensor_data_list = []
        for f in sorted((Path(path) / "sensor").glob(f"sensor_????????_??_??_{sensor_worker_id:02d}_??.csv")):
            sensor_data_list.append(pd.read_csv(f, usecols=("time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "linacc_x", "linacc_y", "linacc_z")))
        sensor_data = pd.concat(sensor_data_list, ignore_index=True)
        if start is not None:
            sensor_data = sensor_data[sensor_data["time"] >= start]
        if stop is not None:
            sensor_data = sensor_data[sensor_data["time"] < stop]
