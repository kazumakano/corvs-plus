import math
import numpy as np
import pandas as pd
from numpy import linalg
from numpy.typing import ArrayLike, NDArray


def seg_by_timeout(data: pd.DataFrame, timeout_in_sec: float) -> list[pd.DataFrame]:
    return [d for _, d in data.groupby(by=(data["time"].diff() > timeout_in_sec).cumsum(), sort=False)]

def pos_to_spd(pos: ArrayLike, freq_in_hz: float, resol_per_px: float) -> NDArray:
    mv_vec = np.diff(pos, axis=0)
    spd = freq_in_hz * resol_per_px * linalg.norm(mv_vec, axis=1)
    return spd

def to_convex(ang: NDArray) -> NDArray:
    return (ang + math.pi) % (2 * math.pi) - math.pi

def pos_to_ang_vel(pos: ArrayLike, freq_in_hz: float) -> NDArray:
    mv_vec = np.diff(pos, axis=0)
    ang = to_convex(np.diff(np.arctan2(mv_vec[:, 1], mv_vec[:, 0])))
    ang_vel = freq_in_hz * ang
    return ang_vel
