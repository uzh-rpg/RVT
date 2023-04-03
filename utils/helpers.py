from typing import Union

import torch as th


def torch_uniform_sample_scalar(min_value: float, max_value: float):
    assert max_value >= min_value, f'{max_value=} is smaller than {min_value=}'
    if max_value == min_value:
        return min_value
    return min_value + (max_value - min_value) * th.rand(1).item()


def clamp(value: Union[int, float], smallest: Union[int, float], largest: Union[int, float]):
    return max(smallest, min(value, largest))
