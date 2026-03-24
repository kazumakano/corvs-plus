import zoneinfo
from datetime import datetime, tzinfo
from os import PathLike
from typing import Optional
import torch
from dateutil import parser

jst = zoneinfo.ZoneInfo("Asia/Tokyo")

def to_unix(dt: float | str | datetime, tzinfo: Optional[tzinfo] = None) -> float:
    if isinstance(dt, str):
        dt = parser.parse(dt).replace(tzinfo=tzinfo)
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tzinfo)
        dt = dt.timestamp()
    return dt

def get_min_int_dtype(max_val: int) -> torch.dtype:
    if max_val < 2 ** 7:
        return torch.int8
    elif max_val < 2 ** 15:
        return torch.int16
    elif max_val < 2 ** 31:
        return torch.int32
    else:
        return torch.int64

def save_txt(data: str, path: PathLike) -> None:
    with open(path, mode="w") as f:
        f.write(data)
