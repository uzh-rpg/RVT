from abc import ABC, abstractmethod
from typing import Optional, Tuple

import math
import numpy as np
import torch as th


class RepresentationBase(ABC):
    @abstractmethod
    def construct(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        ...

    @abstractmethod
    def get_shape(self) -> Tuple[int, int, int]:
        ...

    @staticmethod
    @abstractmethod
    def get_numpy_dtype() -> np.dtype:
        ...

    @staticmethod
    @abstractmethod
    def get_torch_dtype() -> th.dtype:
        ...

    @property
    def dtype(self) -> th.dtype:
        return self.get_torch_dtype()

    @staticmethod
    def _is_int_tensor(tensor: th.Tensor) -> bool:
        return not th.is_floating_point(tensor) and not th.is_complex(tensor)


class StackedHistogram(RepresentationBase):
    def __init__(self, bins: int, height: int, width: int, count_cutoff: Optional[int] = None, fastmode: bool = True):
        """
        In case of fastmode == True: use uint8 to construct the representation, but could lead to overflow.
        In case of fastmode == False: use int16 to construct the representation, and convert to uint8 after clipping.

        Note: Overflow should not be a big problem because it happens only for hot pixels. In case of overflow,
        the value will just start accumulating from 0 again.
        """
        assert bins >= 1
        self.bins = bins
        assert height >= 1
        self.height = height
        assert width >= 1
        self.width = width
        self.count_cutoff = count_cutoff
        if self.count_cutoff is None:
            self.count_cutoff = 255
        else:
            assert count_cutoff >= 1
            self.count_cutoff = min(count_cutoff, 255)
        self.fastmode = fastmode
        self.channels = 2

    @staticmethod
    def get_numpy_dtype() -> np.dtype:
        return np.dtype('uint8')

    @staticmethod
    def get_torch_dtype() -> th.dtype:
        return th.uint8

    def merge_channel_and_bins(self, representation: th.Tensor):
        assert representation.dim() == 4
        return th.reshape(representation, (-1, self.height, self.width))

    def get_shape(self) -> Tuple[int, int, int]:
        return 2 * self.bins, self.height, self.width

    def construct(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        device = x.device
        assert y.device == pol.device == time.device == device
        assert self._is_int_tensor(x)
        assert self._is_int_tensor(y)
        assert self._is_int_tensor(pol)
        assert self._is_int_tensor(time)

        dtype = th.uint8 if self.fastmode else th.int16

        representation = th.zeros((self.channels, self.bins, self.height, self.width),
                                  dtype=dtype, device=device, requires_grad=False)

        if x.numel() == 0:
            assert y.numel() == 0
            assert pol.numel() == 0
            assert time.numel() == 0
            return self.merge_channel_and_bins(representation.to(th.uint8))
        assert x.numel() == y.numel() == pol.numel() == time.numel()

        assert pol.min() >= 0
        assert pol.max() <= 1

        bn, ch, ht, wd = self.bins, self.channels, self.height, self.width

        # NOTE: assume sorted time
        t0_int = time[0]
        t1_int = time[-1]
        assert t1_int >= t0_int
        t_norm = time - t0_int
        t_norm = t_norm / max((t1_int - t0_int), 1)
        t_norm = t_norm * bn
        t_idx = t_norm.floor()
        t_idx = th.clamp(t_idx, max=bn - 1)

        indices = x.long() + \
                  wd * y.long() + \
                  ht * wd * t_idx.long() + \
                  bn * ht * wd * pol.long()
        values = th.ones_like(indices, dtype=dtype, device=device)
        representation.put_(indices, values, accumulate=True)
        representation = th.clamp(representation, min=0, max=self.count_cutoff)
        if not self.fastmode:
            representation = representation.to(th.uint8)

        return self.merge_channel_and_bins(representation)


def cumsum_channel(x: th.Tensor, num_channels: int):
    for i in reversed(range(num_channels)):
        x[i] = th.sum(input=x[:i + 1], dim=0)
    return x


class MixedDensityEventStack(RepresentationBase):
    def __init__(self, bins: int, height: int, width: int, count_cutoff: Optional[int] = None,
                 allow_compilation: bool = False):
        assert bins >= 1
        self.bins = bins
        assert height >= 1
        self.height = height
        assert width >= 1
        self.width = width
        self.count_cutoff = count_cutoff
        if self.count_cutoff is not None:
            assert isinstance(count_cutoff, int)
            assert 0 <= self.count_cutoff <= 2 ** 7 - 1

        self.cumsum_ch_opt = cumsum_channel

        if allow_compilation:
            # Will most likely not work with multiprocessing.
            try:
                self.cumsum_ch_opt = th.compile(cumsum_channel)
            except AttributeError:
                ...

    @staticmethod
    def get_numpy_dtype() -> np.dtype:
        return np.dtype('int8')

    @staticmethod
    def get_torch_dtype() -> th.dtype:
        return th.int8

    def get_shape(self) -> Tuple[int, int, int]:
        return self.bins, self.height, self.width

    def construct(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        device = x.device
        assert y.device == pol.device == time.device == device
        assert self._is_int_tensor(x)
        assert self._is_int_tensor(y)
        assert self._is_int_tensor(pol)
        assert self._is_int_tensor(time)

        dtype = th.int8

        representation = th.zeros((self.bins, self.height, self.width), dtype=dtype, device=device, requires_grad=False)

        if x.numel() == 0:
            assert y.numel() == 0
            assert pol.numel() == 0
            assert time.numel() == 0
            return representation
        assert x.numel() == y.numel() == pol.numel() == time.numel()

        assert pol.min() >= 0  # maybe remove because too costly
        assert pol.max() <= 1  # maybe remove because too costly
        pol = pol * 2 - 1

        bn, ht, wd = self.bins, self.height, self.width

        # NOTE: assume sorted time
        t0_int = time[0]
        t1_int = time[-1]
        assert t1_int >= t0_int
        t_norm = (time - t0_int) / max((t1_int - t0_int), 1)
        t_norm = th.clamp(t_norm, min=1e-6, max=1 - 1e-6)
        # Let N be the number of bins. I.e. bin \in [0, N):
        # Let f(bin) = t_norm, model the relationship between bin and normalized time \in [0, 1]
        # f(bin=N) = 1
        # f(bin=N-1) = 1/2
        # f(bin=N-2) = 1/2*1/2
        # -> f(bin=N-i) = (1/2)^i
        # Also: f(bin) = t_norm
        #
        # Hence, (1/2)^(N-bin) = t_norm
        # And, bin = N - log(t_norm, base=1/2) = N - log(t_norm)/log(1/2)
        bin_float = self.bins - th.log(t_norm) / math.log(1 / 2)
        # Can go below 0 for t_norm close to 0 -> clamp to 0
        bin_float = th.clamp(bin_float, min=0)
        t_idx = bin_float.floor()

        indices = x.long() + \
                  wd * y.long() + \
                  ht * wd * t_idx.long()
        values = th.asarray(pol, dtype=dtype, device=device)
        representation.put_(indices, values, accumulate=True)
        representation = self.cumsum_ch_opt(representation, num_channels=self.bins)
        if self.count_cutoff is not None:
            representation = th.clamp(representation, min=-self.count_cutoff, max=self.count_cutoff)
        return representation
