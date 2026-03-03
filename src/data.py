from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Optional
import pandas as pd
from torch.utils import data


class PredictDataset(data.Dataset):
    def __init__(self, path: PathLike, track_id: int, sensor_id: int, start: Optional[datetime] = None, end: Optional[datetime] = None) -> None:
        # load trajectory data
        traj_data_list = []
        for f in sorted((Path(path) / "trajectory").glob("trajectory_????????_??_??.csv")):
            traj_data_list.append(pd.read_csv(f, usecols=("time", "track", "x", "y")))
        traj_data = pd.concat(traj_data_list, ignore_index=True)
        traj_data = traj_data[traj_data["track"] == track_id]
        if start is not None:
            traj_data = traj_data[traj_data["time"] >= start.timestamp()]
        if end is not None:
            traj_data = traj_data[traj_data["time"] < end.timestamp()]

        # load sensor data
