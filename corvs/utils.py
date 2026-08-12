import functools
import importlib
import typing
import warnings
import zoneinfo
from datetime import datetime
from os import PathLike
from typing import Any, Callable, Iterable, Literal, Optional, ParamSpec, TypeVar
import torch
from dateutil import parser


def resol_type_var(child_cls: type, type_var: TypeVar) -> set[type]:
    """
    Resolve a given type variable to concrete types.

    Parameters
    ----------
    child_cls : type
        Child class.
    type_var : TypeVar
        Type variable of parent class.

    Returns
    -------
    types : set[type]
        Concrete types of child class.

    Examples
    --------
    >>> from typing import AnyStr, TextIO
    >>> from corvs import utils
    >>> utils.resol_type_var(TextIO, AnyStr)
    {<class 'str'>}
    """

    type_vars = {type_var}
    types = set()
    for cc in reversed(child_cls.mro()):
        for ob in getattr(cc, "__orig_bases__", ()):
            orig = typing.get_origin(ob)
            params = getattr(orig, "__parameters__", ())
            args = typing.get_args(ob)

            for tv in type_vars.copy():
                if tv in params:
                    arg = args[params.index(tv)]
                    if isinstance(arg, TypeVar):
                        type_vars.add(arg)
                    else:
                        types.add(typing.get_origin(arg) or arg)

    return types

def to_unix(dt: float | str | datetime, tz: Optional[str] = None) -> float:
    if isinstance(dt, str):
        dt = parser.parse(dt)
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=None if tz is None else zoneinfo.ZoneInfo(tz))
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

def save_txt(txt: str, path: str | bytes | PathLike[str] | PathLike[bytes], mode: Literal["w", "x", "a"] = "w") -> None:
    with open(path, mode=mode) as f:
        f.write(txt)

ArgsP = ParamSpec("ArgsP")
RetT = TypeVar("RetT")

def ignore_warn(cat: type[Warning] | Iterable[type[Warning]] = Warning) -> Callable[[Callable[ArgsP, RetT]], Callable[ArgsP, RetT]]:
    def decorator(func: Callable[ArgsP, RetT]) -> Callable[ArgsP, RetT]:
        @functools.wraps(func)
        def wrapper(*args: ArgsP.args, **kwargs: ArgsP.kwargs) -> RetT:
            with warnings.catch_warnings():
                if isinstance(cat, type):
                    warnings.simplefilter("ignore", category=cat)
                else:
                    for c in cat:
                        warnings.simplefilter("ignore", category=c)
                return func(*args, **kwargs)
        return wrapper
    return decorator
