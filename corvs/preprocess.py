import math
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def seg_by_timeout(data: pd.DataFrame, timeout_in_sec: float) -> list[pd.DataFrame]:
    return [d for _, d in data.groupby(by=(data["time"].diff() > timeout_in_sec).cumsum(), sort=False)]

def conv_to_convex(ang: NDArray) -> NDArray:
    return (ang + math.pi) % (2 * math.pi) - math.pi

def pos_to_ang_vel(pos: ArrayLike, freq_in_hz: float) -> NDArray:
    mv_vec = np.diff(pos, axis=0)
    ang_vel = freq_in_hz * conv_to_convex(np.arctan2(mv_vec[:, 1], mv_vec[:, 0]))
    return ang_vel
