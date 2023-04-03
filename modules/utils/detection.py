from enum import Enum, auto
from typing import List, Optional, Union, Tuple, Dict, Any

import torch
import torch as th

from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.utils.types import BackboneFeatures, LstmStates, DatasetSamplingMode


class Mode(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


mode_2_string = {
    Mode.TRAIN: 'train',
    Mode.VAL: 'val',
    Mode.TEST: 'test',
}


class BackboneFeatureSelector:
    def __init__(self):
        self.features = None
        self.reset()

    def reset(self):
        self.features = dict()

    def add_backbone_features(self,
                              backbone_features: BackboneFeatures,
                              selected_indices: Optional[List[int]] = None) -> None:
        if selected_indices is not None:
            assert len(selected_indices) > 0
        for k, v in backbone_features.items():
            if k not in self.features:
                self.features[k] = [v[selected_indices]] if selected_indices is not None else [v]
            else:
                self.features[k].append(v[selected_indices] if selected_indices is not None else v)

    def get_batched_backbone_features(self) -> Optional[BackboneFeatures]:
        if len(self.features) == 0:
            return None
        return {k: th.cat(v, dim=0) for k, v in self.features.items()}


class EventReprSelector:
    def __init__(self):
        self.repr_list = None
        self.reset()

    def reset(self):
        self.repr_list = list()

    def __len__(self):
        return len(self.repr_list)

    def add_event_representations(
            self, event_representations: th.Tensor, selected_indices: Optional[List[int]] = None) -> None:
        if selected_indices is not None:
            assert len(selected_indices) > 0
        self.repr_list.extend(x[0] for x in event_representations[selected_indices].split(1))

    def get_event_representations_as_list(
            self, start_idx: int = 0, end_idx: Optional[int] = None) -> Optional[List[th.Tensor]]:
        if len(self) == 0:
            return None
        if end_idx is None:
            end_idx = len(self)
        assert start_idx < end_idx, f'{start_idx=}, {end_idx=}'
        return self.repr_list[start_idx:end_idx]


class RNNStates:
    def __init__(self):
        self.states = {}

    def _has_states(self):
        return len(self.states) > 0

    @classmethod
    def recursive_detach(cls, inp: Union[th.Tensor, List, Tuple, Dict]):
        if isinstance(inp, th.Tensor):
            return inp.detach()
        if isinstance(inp, list):
            return [cls.recursive_detach(x) for x in inp]
        if isinstance(inp, tuple):
            return tuple(cls.recursive_detach(x) for x in inp)
        if isinstance(inp, dict):
            return {k: cls.recursive_detach(v) for k, v in inp.items()}
        raise NotImplementedError

    @classmethod
    def recursive_reset(cls,
                        inp: Union[th.Tensor, List, Tuple, Dict],
                        indices_or_bool_tensor: Optional[Union[List[int], torch.Tensor]] = None):
        if isinstance(inp, th.Tensor):
            assert inp.requires_grad is False, 'Not assumed here but should be the case.'
            if indices_or_bool_tensor is None:
                inp[:] = 0
            else:
                assert len(indices_or_bool_tensor) > 0
                inp[indices_or_bool_tensor] = 0
            return inp
        if isinstance(inp, list):
            return [cls.recursive_reset(x, indices_or_bool_tensor=indices_or_bool_tensor) for x in inp]
        if isinstance(inp, tuple):
            return tuple(cls.recursive_reset(x, indices_or_bool_tensor=indices_or_bool_tensor) for x in inp)
        if isinstance(inp, dict):
            return {k: cls.recursive_reset(v, indices_or_bool_tensor=indices_or_bool_tensor) for k, v in inp.items()}
        raise NotImplementedError

    def save_states_and_detach(self, worker_id: int, states: LstmStates) -> None:
        self.states[worker_id] = self.recursive_detach(states)

    def get_states(self, worker_id: int) -> Optional[LstmStates]:
        if not self._has_states():
            return None
        if worker_id not in self.states:
            return None
        return self.states[worker_id]

    def reset(self, worker_id: int, indices_or_bool_tensor: Optional[Union[List[int], torch.Tensor]] = None):
        if not self._has_states():
            return
        if worker_id in self.states:
            self.states[worker_id] = self.recursive_reset(
                self.states[worker_id], indices_or_bool_tensor=indices_or_bool_tensor)


def mixed_collate_fn(x1: Union[th.Tensor, List[th.Tensor]], x2: Union[th.Tensor, List[th.Tensor]]):
    if isinstance(x1, th.Tensor):
        assert isinstance(x2, th.Tensor)
        return th.cat((x1, x2))
    if isinstance(x1, SparselyBatchedObjectLabels):
        assert isinstance(x2, SparselyBatchedObjectLabels)
        return x1 + x2
    if isinstance(x1, list):
        assert isinstance(x2, list)
        assert len(x1) == len(x2)
        return [mixed_collate_fn(x1=el_1, x2=el_2) for el_1, el_2 in zip(x1, x2)]
    raise NotImplementedError


def merge_mixed_batches(batch: Dict[str, Any]):
    if 'data' in batch:
        return batch
    rnd_data = batch[DatasetSamplingMode.RANDOM]['data']
    stream_batch = batch[DatasetSamplingMode.STREAM]
    # We only care about the worker id of the streaming dataloader because the states will be anyway reset for the
    # random dataloader batch.
    out = {'worker_id': stream_batch['worker_id']}
    stream_data = stream_batch['data']
    assert rnd_data.keys() == stream_data.keys(), f'{rnd_data.keys()=}, {stream_data.keys()=}'
    data_out = dict()
    for key in rnd_data.keys():
        data_out[key] = mixed_collate_fn(stream_data[key], rnd_data[key])
    out.update({'data': data_out})
    return out
