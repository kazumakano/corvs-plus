import importlib
import zoneinfo
from datetime import datetime, tzinfo
from os import PathLike
from typing import Any, Literal, Optional, AnyStr
import torch
from dateutil import parser

JST = zoneinfo.ZoneInfo("Asia/Tokyo")

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

def import_by_str(mod: str, qual_name: str) -> Any:
    """
    Import an object by its module and qualified names.

    Parameters
    ----------
    mod : str
        Module name.
    qual_name : str
        Qualified name.

    Returns
    -------
    obj : Any
        Imported object.

    Examples
    -------
    >>> from corvs import utils
    >>> utils.import_by_str("torch.nn", "Module")
    <class 'torch.nn.modules.module.Module'>
    """

    obj = importlib.import_module(mod)
    for qn in qual_name.split("."):
        obj = getattr(obj, qn)
    return obj

def save_txt(data: str, path: str | bytes | PathLike[str] | PathLike[bytes], mode: Literal["w", "x", "a"] = "w") -> None:
    with open(path, mode=mode) as f:
        f.write(data)
