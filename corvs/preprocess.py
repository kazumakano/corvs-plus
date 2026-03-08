import math
from typing import TypeVar, overload
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

FloatingT = TypeVar("FloatingT", np.floating)

def to_convex(ang: NDArray[FloatingT]) -> NDArray[FloatingT]:
    return (ang + math.pi) % (2 * math.pi) - math.pi

def loc_to_ang_vel(loc: ArrayLike, freq_in_hz: float) -> NDArray:
    vec = np.diff(loc, axis=0)
    dir = np.arctan2(vec[:, 1], vec[:, 0])
    ang = to_convex(np.diff(dir))
    ang_vel = freq_in_hz * ang
    return ang_vel

@overload
def sync(time_1: ArrayLike, val_1: ArrayLike, time_2: ArrayLike, val_2: ArrayLike, freq_in_hz: float) -> tuple[NDArray[np.float64], ...]:
    ...

@overload
def sync(time_1: ArrayLike, val_1: ArrayLike, time_2: ArrayLike, val_2: ArrayLike, time_3: ArrayLike, val_3: ArrayLike, freq_in_hz: float) -> tuple[NDArray[np.float64], ...]:
    ...

def sync(*args: float | ArrayLike) -> tuple[NDArray[np.float64], ...]:
    time_list = [np.asanyarray(a) for a in args[:-1:2]]
    val_list = [a for a in args[1::2]]
    freq = args[-1]

    time = np.arange(max(t[0] for t in time_list), min(t[-1] for t in time_list), step=1 / freq, dtype=np.float64)
    val_list = [np.interp(time, t, v) for t, v in zip(time_list, val_list)]

    return time, *val_list
