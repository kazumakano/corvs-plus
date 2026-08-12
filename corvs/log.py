import logging
from typing import Optional
from lightning import pytorch as L


def init_pl_logger(level: Optional[int] = None) -> None:
    if level is not None:
        L._logger.setLevel(level)  # original level is info
    L._logger.handlers[0].setFormatter(logging.Formatter(fmt="[Lightning %(levelname)s] %(message)s"))  # original format is '%(message)s'

def init_corvs_logger(level: Optional[int] = None) -> None:
    logger = logging.getLogger(name=__package__)
    if level is not None:
        logger.setLevel(level)
    logger.propagate = False
    hdlr = logging.StreamHandler()
    hdlr.setFormatter(logging.Formatter(fmt="[CorVS %(levelname)s] %(message)s"))
    logger.addHandler(hdlr)

def init_all_loggers(level: Optional[int] = None) -> None:
    init_pl_logger(level)
    init_corvs_logger(level)
