import zoneinfo
from datetime import datetime, tzinfo
from os import PathLike
from typing import Optional
from dateutil import parser

jst = zoneinfo.ZoneInfo("Asia/Tokyo")

def any_to_unix(dt: float | str | datetime, tzinfo: Optional[tzinfo] = None) -> float:
    if isinstance(dt, str):
        dt = parser.parse(dt).replace(tzinfo=tzinfo)
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tzinfo)
        dt = dt.timestamp()
    return dt

def save_txt(data: str, path: PathLike) -> None:
    with open(path, mode="w") as f:
        f.write(data)
