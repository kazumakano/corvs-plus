import zoneinfo
from datetime import datetime, tzinfo
from os import PathLike
from typing import Callable, Literal, Optional
import torch
from dateutil import parser
from torch import nn
from torch.nn import functional as F

JST = zoneinfo.ZoneInfo("Asia/Tokyo")

def str_to_mod(act: Literal["relu", "gelu", "silu"], func: bool = False) -> nn.ReLU | nn.GELU | nn.SiLU | Callable[[torch.Tensor], torch.Tensor]:
    match act:
        case "relu":
            return F.relu if func else nn.ReLU()
        case "gelu":
            return F.gelu if func else nn.GELU()
        case "silu":
            return F.silu if func else nn.SiLU()
        case _:
            raise ValueError("only ReLU, GELU, and SiLU are supported")

def to_unix(dt: float | str | datetime, tzinfo: Optional[tzinfo] = None) -> float:
    if isinstance(dt, str):
        dt = parser.parse(dt).replace(tzinfo=tzinfo)
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tzinfo)
        dt = dt.timestamp()
    return dt

def get_min_int_dtype(val: int) -> torch.dtype:
    if val < 2 ** 7:
        return torch.int8
    elif val < 2 ** 15:
        return torch.int16
    elif val < 2 ** 31:
        return torch.int32
    else:
        return torch.int64

def save_txt(data: str, path: PathLike) -> None:
    with open(path, mode="w") as f:
        f.write(data)
