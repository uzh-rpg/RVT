from enum import auto, Enum

try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum
from typing import Dict, List, Optional, Tuple, Union

import torch as th

from data.genx_utils.labels import ObjectLabels, SparselyBatchedObjectLabels


class DataType(Enum):
    EV_REPR = auto()
    FLOW = auto()
    IMAGE = auto()
    OBJLABELS = auto()
    OBJLABELS_SEQ = auto()
    IS_PADDED_MASK = auto()
    IS_FIRST_SAMPLE = auto()
    TOKEN_MASK = auto()


class DatasetType(Enum):
    GEN1 = auto()
    GEN4 = auto()


class DatasetMode(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    TESTING = auto()


class DatasetSamplingMode(StrEnum):
    RANDOM = 'random'
    STREAM = 'stream'
    MIXED = 'mixed'


class ObjDetOutput(Enum):
    LABELS_PROPH = auto()
    PRED_PROPH = auto()
    EV_REPR = auto()
    SKIP_VIZ = auto()


LoaderDataDictGenX = Dict[DataType, Union[List[th.Tensor], ObjectLabels, SparselyBatchedObjectLabels, List[bool]]]

LstmState = Optional[Tuple[th.Tensor, th.Tensor]]
LstmStates = List[LstmState]

FeatureMap = th.Tensor
BackboneFeatures = Dict[int, th.Tensor]
