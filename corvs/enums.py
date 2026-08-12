import enum


class Mode(enum.Enum):
    TRAIN = enum.auto()
    VAL   = enum.auto()
    TEST  = enum.auto()
    PRED  = enum.auto()

class Modality(enum.Enum):
    TIME         = enum.auto()
    TRACK_ID     = enum.auto()
    WORKER_ID    = enum.auto()
    TRAJ_FEAT    = enum.auto()
    SENSOR_FEAT  = enum.auto()
    VALID_MASK   = enum.auto()
    VISIBLE_MASK = enum.auto()
    LABEL        = enum.auto()

class TrajMet(enum.Enum):
    SPD       = enum.auto()
    TURN_RATE = enum.auto()

class SensorMet(enum.Enum):
    ACC_X       = enum.auto()
    ACC_Y       = enum.auto()
    ACC_Z       = enum.auto()
    GRAV_X      = enum.auto()
    GRAV_Y      = enum.auto()
    GRAV_Z      = enum.auto()
    GYRO_X      = enum.auto()
    GYRO_Y      = enum.auto()
    GYRO_Z      = enum.auto()
    LINACC_NORM = enum.auto()
