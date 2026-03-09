import zoneinfo
from datetime import datetime, tzinfo
from typing import Optional
from dateutil import parser

jst = zoneinfo.ZoneInfo("Asia/Tokyo")

def str_to_datetime(dt: str, tzinfo: Optional[tzinfo] = None) -> datetime:
    return parser.parse(dt).replace(tzinfo=tzinfo)
