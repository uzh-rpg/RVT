from typing import Any, List, Tuple

import torch as th
import torch.nn.functional as F


class InputPadderFromShape:
    def __init__(self, desired_hw: Tuple[int, int], mode: str = 'constant', value: int = 0, type: str = 'corner'):
        """
        :param desired_hw: Desired height and width
        :param mode: See torch.nn.functional.pad
        :param value:  See torch.nn.functional.pad
        :param type: "corner": add zero to bottom and right
        """
        assert isinstance(desired_hw, tuple)
        assert len(desired_hw) == 2
        assert desired_hw[0] % 4 == 0, 'Required for token mask padding'
        assert desired_hw[1] % 4 == 0, 'Required for token mask padding'
        assert type in {'corner'}

        self.desired_hw = desired_hw
        self.mode = mode
        self.value = value
        self.type = type
        self._pad_ev_repr = None
        self._pad_token_mask = None

    @staticmethod
    def _pad_tensor_impl(input_tensor: th.Tensor, desired_hw: Tuple[int, int], mode: str, value: Any) \
            -> Tuple[th.Tensor, List[int]]:
        assert isinstance(input_tensor, th.Tensor)

        ht, wd = input_tensor.shape[-2:]
        ht_des, wd_des = desired_hw
        assert ht <= ht_des
        assert wd <= wd_des

        pad_left = 0
        pad_right = wd_des - wd
        pad_top = 0
        pad_bottom = ht_des - ht

        pad = [pad_left, pad_right, pad_top, pad_bottom]
        return F.pad(input_tensor, pad=pad, mode=mode, value=value if mode == 'constant' else None), pad

    def pad_tensor_ev_repr(self, ev_repr: th.Tensor) -> th.Tensor:
        padded_ev_repr, pad = self._pad_tensor_impl(input_tensor=ev_repr, desired_hw=self.desired_hw,
                                                    mode=self.mode, value=self.value)
        if self._pad_ev_repr is None:
            self._pad_ev_repr = pad
        else:
            assert self._pad_ev_repr == pad
        return padded_ev_repr

    def pad_token_mask(self, token_mask: th.Tensor):
        assert isinstance(token_mask, th.Tensor)

        desired_hw = tuple(x // 4 for x in self.desired_hw)
        padded_token_mask, pad = self._pad_tensor_impl(input_tensor=token_mask, desired_hw=desired_hw,
                                                       mode='constant', value=0)
        if self._pad_token_mask is None:
            self._pad_token_mask = pad
        else:
            assert self._pad_token_mask == pad
        return padded_token_mask
