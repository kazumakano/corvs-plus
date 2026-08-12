import pathlib
from os import PathLike
from typing import Generator, Iterable, Optional
import numpy as np
import pandas as pd


def load_traj_data(
        path: str | PathLike[str],
        track_ids: Optional[Iterable[int]] = None,
        label_ids: Optional[Iterable[int]] = None,
        start: Optional[float] = None,
        end: Optional[float] = None
    ) -> pd.DataFrame:

    all_data = []
    for p in sorted(pathlib.Path(path).glob("trajectory_????????_??_??.csv")):
        data = pd.read_csv(p, usecols=lambda cn: cn in ("time", "track", "x", "y", "label"), dtype={"track": np.uint32, "label": np.uint32})  # label column is optional
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

def load_sensor_data(
        path: str | PathLike[str],
        worker_id: int,
        cols: Iterable[str],
        start: Optional[float] = None,
        end: Optional[float] = None
    ) -> pd.DataFrame:

    all_data = []
    for p in sorted(pathlib.Path(path).glob(f"sensor_????????_??_??_{worker_id:02d}_??.csv")):
        data = pd.read_csv(p, usecols=cols, engine="pyarrow")
        if start is not None:
            data = data[data["time"] >= start]
        if end is not None:
            data = data[data["time"] < end]
        all_data.append(data)

    if len(all_data) == 0:
        all_data = pd.DataFrame(columns=cols)
    else:
        all_data = pd.concat(all_data, ignore_index=True)

    return all_data

def iter_all_sensor_data(
        path: str | PathLike[str],
        cols: Iterable[str],
        start: Optional[float] = None,
        end: Optional[float] = None
    ) -> Generator[tuple[int, pd.DataFrame], None, None]:

    worker_ids = set()
    for p in pathlib.Path(path).glob(f"sensor_????????_??_??_??_??.csv"):
        worker_ids.add(int(p.name.split("_")[4]))

    for wi in sorted(worker_ids):
        data = load_sensor_data(path, wi, cols, start, end)
        if len(data) > 0:
            yield wi, data
