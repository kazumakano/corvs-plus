import typing
import warnings
from typing import Any, Callable, Literal
import torch
from torch import nn
from torch.nn import functional as F


@typing.overload
def get_act(name: Literal["relu", "leaky_relu", "gelu", "silu"], func: Literal[False] = False, **kwargs: Any) -> nn.ReLU | nn.LeakyReLU | nn.GELU | nn.SiLU:
    ...

@typing.overload
def get_act(name: Literal["relu", "leaky_relu", "gelu", "silu"], func: Literal[True]) -> Callable[[torch.FloatTensor], torch.FloatTensor]:
    ...

def get_act(name: Literal["relu", "leaky_relu", "gelu", "silu"], func: bool = False, **kwargs: Any) -> nn.ReLU | nn.LeakyReLU | nn.GELU | nn.SiLU | Callable[[torch.FloatTensor], torch.FloatTensor]:
    if func and len(kwargs) > 0:
        warnings.warn(UserWarning("Keyword arguments are ignored when functional."))

    match name:
        case "relu":
            return F.relu if func else nn.ReLU(**kwargs)
        case "leaky_relu":
            return F.leaky_relu if func else nn.LeakyReLU(**kwargs)
        case "gelu":
            return F.gelu if func else nn.GELU(**kwargs)
        case "silu":
            return F.silu if func else nn.SiLU(**kwargs)
        case _:
            raise ValueError(f"Unknown activation function {name} was specified.")
