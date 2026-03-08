import math
import numpy as np
import pandas as pd
from numpy import linalg
from numpy.typing import ArrayLike, NDArray


def seg_by_timeout(data: pd.DataFrame, timeout_in_sec: float) -> list[pd.DataFrame]:
    return [d for _, d in data.groupby(by=(data["time"].diff() > timeout_in_sec).cumsum(), sort=False)]

def loc_to_spd(loc: ArrayLike, freq_in_hz: float, resol_per_px: float) -> NDArray:
    vec = np.diff(loc, axis=0)
    dist = resol_per_px * linalg.norm(vec, axis=1)
    spd = freq_in_hz * dist
    return spd

def to_convex(ang: NDArray) -> NDArray:
    return (ang + math.pi) % (2 * math.pi) - math.pi

def loc_to_ang_vel(loc: ArrayLike, freq_in_hz: float) -> NDArray:
    vec = np.diff(loc, axis=0)
    dir = np.arctan2(vec[:, 1], vec[:, 0])
    ang = to_convex(np.diff(dir))
    ang_vel = freq_in_hz * ang
    return ang_vel
