from typing import Tuple

import torch.nn as nn


class BaseDetector(nn.Module):
    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        raise NotImplementedError

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        raise NotImplementedError
