import math
from typing import Literal, Optional, TypeVar, overload
import numba
import numpy as np
import pandas as pd
import torch
from numpy import linalg, random
from numpy.typing import ArrayLike, NDArray
from pandas.core.groupby import DataFrameGroupBy
from scipy.interpolate import interp1d
from torch.nn import functional as F
from torch.nn.utils import rnn


def rand_split(data: ArrayLike, ratio: tuple[float, float, float], rng: Optional[random.Generator] = None) -> list[NDArray]:
    if sum(ratio) != 1:
        raise ValueError("summation of split proportions must be 1")

    if rng is None:
        rng = random.default_rng()
    data = rng.permutation(data)
    data = np.split(data, (round(ratio[0] * len(data)), round(sum(ratio[:2]) * len(data)), len(data)))

    return data

def seg_by_timeout(data: pd.DataFrame, timeout_in_sec: float) -> DataFrameGroupBy:
    return data.groupby(by=(data["time"].diff() > timeout_in_sec).cumsum(), sort=False)

def loc_to_spd(loc: ArrayLike, freq_in_hz: float, resol_per_px: float) -> NDArray:
    vec = np.diff(loc, axis=0)
    dist = resol_per_px * linalg.norm(vec, axis=1)
    spd = freq_in_hz * dist
    return spd

FloatingT = TypeVar("FloatingT", bound=np.floating)

@numba.njit(cache=True)
def to_convex(ang: NDArray[FloatingT]) -> NDArray[FloatingT]:
    return (ang + math.pi) % (2 * math.pi) - math.pi

def loc_to_turn_rate(loc: ArrayLike, freq_in_hz: float) -> NDArray:
    vec = np.diff(loc, axis=0)
    dir = np.arctan2(vec[:, 1], vec[:, 0])
    ang = to_convex(np.diff(dir))
    ang_vel = freq_in_hz * ang
    return ang_vel

@overload
def sync(time_1: ArrayLike, val_1: ArrayLike, time_2: ArrayLike, val_2: ArrayLike, /, freq_in_hz: float, *, kind: str = "linear") -> tuple[NDArray[np.float64], ...]:
    ...

@overload
def sync(time_1: ArrayLike, val_1: ArrayLike, time_2: ArrayLike, val_2: ArrayLike, time_3: ArrayLike, val_3: ArrayLike, /, freq_in_hz: float, *, kind: str = "linear") -> tuple[NDArray[np.float64], ...]:
    ...

def sync(*args: ArrayLike | float, kind: str = "linear", **kwargs: float) -> tuple[NDArray[np.float64], ...]:
    time_list = [np.asanyarray(a, dtype=np.float64) for a in args[:-1:2]]
    val_list = args[1::2]
    freq = kwargs.get("freq_in_hz", args[-1])

    time = np.arange(max(t[0] for t in time_list), min(t[-1] for t in time_list), step=1 / freq, dtype=np.float64)
    val_list = [interp1d(t, v, kind=kind, axis=0, copy=False, fill_value="extrapolate", assume_sorted=True)(time) for t, v in zip(time_list, val_list)]

    return time, *val_list

def pad(seqs: torch.Tensor | list[torch.Tensor], tgt_len: int, batch_first: bool = False, pad_val: float = 0, pad_side: Literal["left", "right"] = "right") -> torch.Tensor:
    """
    Pad or truncate variable length sequences to a uniform length.

    Parameters
    ----------
    seqs : Tensor | list[Tensor]
        List of sequences.
        Shape is (batch, seq, ...).
    tgt_len : int
        Target sequence length.
    batch_first : bool
        Put batch dimension first or not.
    pad_val : float
        Padding value.
    pad_side : 'left' | 'right'
        Padding side.

    Returns
    -------
    seqs : Tensor
        Padded sequences.
        Shape is (batch, tgt_len, ...) if batch first, (tgt_len, batch, ...) otherwise.
    """

    seqs = rnn.pad_sequence(seqs, batch_first=batch_first, padding_value=pad_val, padding_side=pad_side)
    pad = 2 * seqs.ndim * [0]
    match pad_side:
        case "left":
            if batch_first:
                pad[-4] = tgt_len - seqs.shape[1]
            else:
                pad[-2] = tgt_len - len(seqs)
        case "right":
            if batch_first:
                pad[-3] = tgt_len - seqs.shape[1]
            else:
                pad[-1] = tgt_len - len(seqs)
    seqs = F.pad(seqs, pad, value=pad_val)
    return seqs
