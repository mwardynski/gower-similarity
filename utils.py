from dataclasses import dataclass
from enum import Enum


class DataType(Enum):
    BINARY_SYMMETRIC = 0
    BINARY_ASYMMETRIC = 1
    CATEGORICAL_NOMINAL = 2
    CATEGORICAL_ORDINAL = 3
    RATIO_SCALE = 4
    NUMERIC_INTERVAL = 5


@dataclass
class Dataset:
    name: str
    task: str
    metric: str
    labeled: bool


@dataclass
class Data:
    cols_type_mapping = {
        "bin_sym": DataType.BINARY_SYMMETRIC,
        "cat_nom": DataType.CATEGORICAL_NOMINAL,
        "cat_ord": DataType.CATEGORICAL_ORDINAL,
        "ratio": DataType.RATIO_SCALE,
        "num_interval": DataType.NUMERIC_INTERVAL,
        "NULL": None,
    }
    data: dict
    cols_type: dict
    labels: dict
