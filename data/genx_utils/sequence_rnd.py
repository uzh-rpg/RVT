from pathlib import Path

from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.genx_utils.sequence_base import SequenceBase
from data.utils.types import DataType, DatasetType, LoaderDataDictGenX
from utils.timers import TimerDummy as Timer


class SequenceForRandomAccess(SequenceBase):
    def __init__(self,
                 path: Path,
                 ev_representation_name: str,
                 sequence_length: int,
                 dataset_type: DatasetType,
                 downsample_by_factor_2: bool,
                 only_load_end_labels: bool):
        super().__init__(path=path,
                         ev_representation_name=ev_representation_name,
                         sequence_length=sequence_length,
                         dataset_type=dataset_type,
                         downsample_by_factor_2=downsample_by_factor_2,
                         only_load_end_labels=only_load_end_labels)

        self.start_idx_offset = None
        for objframe_idx, repr_idx in enumerate(self.objframe_idx_2_repr_idx):
            if repr_idx - self.seq_len + 1 >= 0:
                # We can fit the sequence length to the label
                self.start_idx_offset = objframe_idx
                break
        if self.start_idx_offset is None:
            # This leads to actual length of 0:
            self.start_idx_offset = len(self.label_factory)

        self.length = len(self.label_factory) - self.start_idx_offset
        assert len(self.label_factory) == len(self.objframe_idx_2_repr_idx)

        # Useful for weighted sampler that is based on label statistics:
        self._only_load_labels = False

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> LoaderDataDictGenX:
        corrected_idx = index + self.start_idx_offset
        labels_repr_idx = self.objframe_idx_2_repr_idx[corrected_idx]

        end_idx = labels_repr_idx + 1
        start_idx = end_idx - self.seq_len
        assert_msg = f'{self.ev_repr_file=}, {self.start_idx_offset=}, {start_idx=}, {end_idx=}'
        assert start_idx >= 0, assert_msg

        labels = list()
        for repr_idx in range(start_idx, end_idx):
            if self.only_load_end_labels and repr_idx < end_idx - 1:
                labels.append(None)
            else:
                labels.append(self._get_labels_from_repr_idx(repr_idx))
        sparse_labels = SparselyBatchedObjectLabels(sparse_object_labels_batch=labels)
        if self._only_load_labels:
            return {DataType.OBJLABELS_SEQ: sparse_labels}

        with Timer(timer_name='read ev reprs'):
            ev_repr = self._get_event_repr_torch(start_idx=start_idx, end_idx=end_idx)
        assert len(sparse_labels) == len(ev_repr)

        is_first_sample = True  # Due to random loading
        is_padded_mask = [False] * len(ev_repr)

        out = {
            DataType.EV_REPR: ev_repr,
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.IS_FIRST_SAMPLE: is_first_sample,
            DataType.IS_PADDED_MASK: is_padded_mask,
        }
        return out

    def is_only_loading_labels(self) -> bool:
        return self._only_load_labels

    def only_load_labels(self):
        self._only_load_labels = True

    def load_everything(self):
        self._only_load_labels = False
