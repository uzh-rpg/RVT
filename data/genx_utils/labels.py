from __future__ import annotations

from typing import List, Tuple, Union, Optional

import math
import numpy as np
import torch as th
from einops import rearrange
from torch.nn.functional import pad


class ObjectLabelBase:
    _str2idx = {
        't': 0,
        'x': 1,
        'y': 2,
        'w': 3,
        'h': 4,
        'class_id': 5,
        'class_confidence': 6,
    }

    def __init__(self,
                 object_labels: th.Tensor,
                 input_size_hw: Tuple[int, int]):
        assert isinstance(object_labels, th.Tensor)
        assert object_labels.dtype in {th.float32, th.float64}
        assert object_labels.ndim == 2
        assert object_labels.shape[-1] == len(self._str2idx)
        assert isinstance(input_size_hw, tuple)
        assert len(input_size_hw) == 2

        self.object_labels = object_labels
        self._input_size_hw = input_size_hw
        self._is_numpy = False

    def clamp_to_frame_(self):
        ht, wd = self.input_size_hw
        x0 = th.clamp(self.x, min=0, max=wd - 1)
        y0 = th.clamp(self.y, min=0, max=ht - 1)
        x1 = th.clamp(self.x + self.w, min=0, max=wd - 1)
        y1 = th.clamp(self.y + self.h, min=0, max=ht - 1)
        w = x1 - x0
        h = y1 - y0
        assert th.all(w > 0)
        assert th.all(h > 0)
        self.x = x0
        self.y = y0
        self.w = w
        self.h = h

    def remove_flat_labels_(self):
        keep = (self.w > 0) & (self.h > 0)
        self.object_labels = self.object_labels[keep]

    @classmethod
    def create_empty(cls):
        # This is useful to represent cases where no labels are available.
        return ObjectLabelBase(object_labels=th.empty((0, len(cls._str2idx))), input_size_hw=(0, 0))

    def _assert_not_numpy(self):
        assert not self._is_numpy, "Labels have been converted numpy. \
        Numpy is not supported for the intended operations."

    def to(self, *args, **kwargs):
        # This function executes torch.to on self tensors and returns self.
        self._assert_not_numpy()
        # This will be used by Pytorch Lightning to transfer to the relevant device
        self.object_labels = self.object_labels.to(*args, **kwargs)
        return self

    def numpy_(self) -> None:
        """
        In place conversion to numpy (detach + to cpu + to numpy).
        Cannot be undone.
        """
        self._is_numpy = True
        self.object_labels = self.object_labels.detach().cpu().numpy()

    @property
    def input_size_hw(self) -> Tuple[int, int]:
        return self._input_size_hw

    @input_size_hw.setter
    def input_size_hw(self, height_width: Tuple[int, int]):
        assert isinstance(height_width, tuple)
        assert len(height_width) == 2
        assert height_width[0] > 0
        assert height_width[1] > 0
        self._input_size_hw = height_width

    def get(self, request: str):
        assert request in self._str2idx
        return self.object_labels[:, self._str2idx[request]]

    @property
    def t(self):
        return self.object_labels[:, self._str2idx['t']]

    @property
    def x(self):
        return self.object_labels[:, self._str2idx['x']]

    @x.setter
    def x(self, value: Union[th.Tensor, np.ndarray]):
        self.object_labels[:, self._str2idx['x']] = value

    @property
    def y(self):
        return self.object_labels[:, self._str2idx['y']]

    @y.setter
    def y(self, value: Union[th.Tensor, np.ndarray]):
        self.object_labels[:, self._str2idx['y']] = value

    @property
    def w(self):
        return self.object_labels[:, self._str2idx['w']]

    @w.setter
    def w(self, value: Union[th.Tensor, np.ndarray]):
        self.object_labels[:, self._str2idx['w']] = value

    @property
    def h(self):
        return self.object_labels[:, self._str2idx['h']]

    @h.setter
    def h(self, value: Union[th.Tensor, np.ndarray]):
        self.object_labels[:, self._str2idx['h']] = value

    @property
    def class_id(self):
        return self.object_labels[:, self._str2idx['class_id']]

    @property
    def class_confidence(self):
        return self.object_labels[:, self._str2idx['class_confidence']]

    @property
    def dtype(self):
        return self.object_labels.dtype

    @property
    def device(self):
        return self.object_labels.device


class ObjectLabelFactory(ObjectLabelBase):
    def __init__(self,
                 object_labels: th.Tensor,
                 objframe_idx_2_label_idx: th.Tensor,
                 input_size_hw: Tuple[int, int],
                 downsample_factor: Optional[float] = None):
        super().__init__(object_labels=object_labels, input_size_hw=input_size_hw)
        assert objframe_idx_2_label_idx.dtype == th.int64
        assert objframe_idx_2_label_idx.dim() == 1

        self.objframe_idx_2_label_idx = objframe_idx_2_label_idx
        self.downsample_factor = downsample_factor
        if self.downsample_factor is not None:
            assert self.downsample_factor > 1
        self.clamp_to_frame_()

    @staticmethod
    def from_structured_array(object_labels: np.ndarray,
                              objframe_idx_2_label_idx: np.ndarray,
                              input_size_hw: Tuple[int, int],
                              downsample_factor: Optional[float] = None) -> ObjectLabelFactory:
        np_labels = [object_labels[key].astype('float32') for key in ObjectLabels._str2idx.keys()]
        np_labels = rearrange(np_labels, 'fields L -> L fields')
        torch_labels = th.from_numpy(np_labels)
        objframe_idx_2_label_idx = th.from_numpy(objframe_idx_2_label_idx.astype('int64'))
        assert objframe_idx_2_label_idx.numel() == np.unique(object_labels['t']).size
        return ObjectLabelFactory(object_labels=torch_labels,
                                  objframe_idx_2_label_idx=objframe_idx_2_label_idx,
                                  input_size_hw=input_size_hw,
                                  downsample_factor=downsample_factor)

    def __len__(self):
        return len(self.objframe_idx_2_label_idx)

    def __getitem__(self, item: int) -> ObjectLabels:
        assert item >= 0
        length = len(self)
        assert length > 0
        assert item < length
        is_last_item = (item == length - 1)

        from_idx = self.objframe_idx_2_label_idx[item]
        to_idx = self.object_labels.shape[0] if is_last_item else self.objframe_idx_2_label_idx[item + 1]
        assert to_idx > from_idx
        object_labels = ObjectLabels(
            object_labels=self.object_labels[from_idx:to_idx].clone(),
            input_size_hw=self.input_size_hw)
        if self.downsample_factor is not None:
            object_labels.scale_(scaling_multiplier=1 / self.downsample_factor)
        return object_labels


class ObjectLabels(ObjectLabelBase):
    def __init__(self,
                 object_labels: th.Tensor,
                 input_size_hw: Tuple[int, int]):
        super().__init__(object_labels=object_labels, input_size_hw=input_size_hw)

    def __len__(self) -> int:
        return self.object_labels.shape[0]

    def rotate_(self, angle_deg: float):
        if len(self) == 0:
            return
        # (x0,y0)---(x1,y0)   p00---p10
        #  |             |    |       |
        #  |             |    |       |
        # (x0,y1)---(x1,y1)   p01---p11
        p00 = th.stack((self.x, self.y), dim=1)
        p10 = th.stack((self.x + self.w, self.y), dim=1)
        p01 = th.stack((self.x, self.y + self.h), dim=1)
        p11 = th.stack((self.x + self.w, self.y + self.h), dim=1)
        # points: 4 x N x 2
        points = th.stack((p00, p10, p01, p11), dim=0)

        cx = self._input_size_hw[1] // 2
        cy = self._input_size_hw[0] // 2
        center = th.tensor([cx, cy], device=self.device)

        angle_rad = angle_deg / 180 * math.pi
        # counter-clockwise rotation
        rot_matrix = th.tensor([[math.cos(angle_rad), math.sin(angle_rad)],
                                [-math.sin(angle_rad), math.cos(angle_rad)]], device=self.device)

        points = points - center
        points = th.einsum('ij,pnj->pni', rot_matrix, points)
        points = points + center

        height, width = self.input_size_hw
        x0 = th.clamp(th.min(points[..., 0], dim=0)[0], min=0, max=width - 1)
        y0 = th.clamp(th.min(points[..., 1], dim=0)[0], min=0, max=height - 1)
        x1 = th.clamp(th.max(points[..., 0], dim=0)[0], min=0, max=width - 1)
        y1 = th.clamp(th.max(points[..., 1], dim=0)[0], min=0, max=height - 1)

        self.x = x0
        self.y = y0
        self.w = x1 - x0
        self.h = y1 - y0

        self.remove_flat_labels_()

        assert th.all(self.x >= 0)
        assert th.all(self.y >= 0)
        assert th.all(self.x + self.w <= self.input_size_hw[1] - 1)
        assert th.all(self.y + self.h <= self.input_size_hw[0] - 1)

    def zoom_in_and_rescale_(self, zoom_coordinates_x0y0: Tuple[int, int], zoom_in_factor: float):
        """
        1) Computes a new smaller canvas size: original canvas scaled by a factor of 1/zoom_in_factor (downscaling)
        2) Places the smaller canvas inside the original canvas at the top-left coordinates zoom_coordinates_x0y0
        3) Extract the smaller canvas and rescale it back to the original resolution
        """
        if len(self) == 0:
            return
        assert len(zoom_coordinates_x0y0) == 2
        assert zoom_in_factor >= 1
        if zoom_in_factor == 1:
            return
        z_x0, z_y0 = zoom_coordinates_x0y0
        h_orig, w_orig = self.input_size_hw
        assert 0 <= z_x0 <= w_orig - 1
        assert 0 <= z_y0 <= h_orig - 1
        zoom_window_h, zoom_window_w = tuple(x / zoom_in_factor for x in self.input_size_hw)
        z_x1 = min(z_x0 + zoom_window_w, w_orig - 1)
        assert z_x1 <= w_orig - 1, f'{z_x1=} is larger than {w_orig-1=}'
        z_y1 = min(z_y0 + zoom_window_h, h_orig - 1)
        assert z_y1 <= h_orig - 1, f'{z_y1=} is larger than {h_orig-1=}'

        x0 = th.clamp(self.x, min=z_x0, max=z_x1 - 1)
        y0 = th.clamp(self.y, min=z_y0, max=z_y1 - 1)

        x1 = th.clamp(self.x + self.w, min=z_x0, max=z_x1 - 1)
        y1 = th.clamp(self.y + self.h, min=z_y0, max=z_y1 - 1)

        self.x = x0 - z_x0
        self.y = y0 - z_y0
        self.w = x1 - x0
        self.h = y1 - y0
        self.input_size_hw = (zoom_window_h, zoom_window_w)

        self.remove_flat_labels_()

        self.scale_(scaling_multiplier=zoom_in_factor)

    def zoom_out_and_rescale_(self, zoom_coordinates_x0y0: Tuple[int, int], zoom_out_factor: float):
        """
        1) Scales the input by a factor of 1/zoom_out_factor (i.e. reduces the canvas size)
        2) Places the downscaled canvas into the original canvas at the top-left coordinates zoom_coordinates_x0y0
        """
        if len(self) == 0:
            return
        assert len(zoom_coordinates_x0y0) == 2
        assert zoom_out_factor >= 1
        if zoom_out_factor == 1:
            return

        h_orig, w_orig = self.input_size_hw
        self.scale_(scaling_multiplier=1 / zoom_out_factor)

        self.input_size_hw = (h_orig, w_orig)
        z_x0, z_y0 = zoom_coordinates_x0y0
        assert 0 <= z_x0 <= w_orig - 1
        assert 0 <= z_y0 <= h_orig - 1

        self.x = self.x + z_x0
        self.y = self.y + z_y0

    def scale_(self, scaling_multiplier: float):
        if len(self) == 0:
            return
        assert scaling_multiplier > 0
        if scaling_multiplier == 1:
            return
        img_ht, img_wd = self.input_size_hw
        new_img_ht = scaling_multiplier * img_ht
        new_img_wd = scaling_multiplier * img_wd
        self.input_size_hw = (new_img_ht, new_img_wd)
        x1 = th.clamp((self.x + self.w) * scaling_multiplier, max=new_img_wd - 1)
        y1 = th.clamp((self.y + self.h) * scaling_multiplier, max=new_img_ht - 1)
        self.x = self.x * scaling_multiplier
        self.y = self.y * scaling_multiplier

        self.w = x1 - self.x
        self.h = y1 - self.y

        self.remove_flat_labels_()

    def flip_lr_(self) -> None:
        if len(self) == 0:
            return
        self.x = self.input_size_hw[1] - 1 - self.x - self.w

    def get_labels_as_tensors(self, format_: str = 'yolox') -> th.Tensor:
        self._assert_not_numpy()

        if format_ == 'yolox':
            out = th.zeros((len(self), 5), dtype=th.float32, device=self.device)
            if len(self) == 0:
                return out
            out[:, 0] = self.class_id
            out[:, 1] = self.x + 0.5 * self.w
            out[:, 2] = self.y + 0.5 * self.h
            out[:, 3] = self.w
            out[:, 4] = self.h
            return out
        else:
            raise NotImplementedError

    @staticmethod
    def get_labels_as_batched_tensor(obj_label_list: List[ObjectLabels], format_: str = 'yolox') -> th.Tensor:
        num_object_frames = len(obj_label_list)
        assert num_object_frames > 0
        max_num_labels_per_object_frame = max([len(x) for x in obj_label_list])
        assert max_num_labels_per_object_frame > 0

        if format_ == 'yolox':
            tensor_labels = []
            for labels in obj_label_list:
                obj_labels_tensor = labels.get_labels_as_tensors(format_=format_)
                num_to_pad = max_num_labels_per_object_frame - len(labels)
                padded_labels = pad(obj_labels_tensor, (0, 0, 0, num_to_pad), mode='constant', value=0)
                tensor_labels.append(padded_labels)
            tensor_labels = th.stack(tensors=tensor_labels, dim=0)
            return tensor_labels
        else:
            raise NotImplementedError


class SparselyBatchedObjectLabels:
    def __init__(self, sparse_object_labels_batch: List[Optional[ObjectLabels]]):
        # Can contain None elements that indicate missing labels.
        for entry in sparse_object_labels_batch:
            assert isinstance(entry, ObjectLabels) or entry is None
        self.sparse_object_labels_batch = sparse_object_labels_batch
        self.set_empty_labels_to_none_()

    def __len__(self) -> int:
        return len(self.sparse_object_labels_batch)

    def __iter__(self):
        return iter(self.sparse_object_labels_batch)

    def __getitem__(self, item: int) -> Optional[ObjectLabels]:
        if item < 0 or item >= len(self):
            raise IndexError(f'Index ({item}) out of range (0, {len(self) - 1})')
        return self.sparse_object_labels_batch[item]

    def __add__(self, other: SparselyBatchedObjectLabels):
        sparse_object_labels_batch = self.sparse_object_labels_batch + other.sparse_object_labels_batch
        return SparselyBatchedObjectLabels(sparse_object_labels_batch=sparse_object_labels_batch)

    def set_empty_labels_to_none_(self):
        for idx, obj_label in enumerate(self.sparse_object_labels_batch):
            if obj_label is not None and len(obj_label) == 0:
                self.sparse_object_labels_batch[idx] = None

    @property
    def input_size_hw(self) -> Optional[Union[Tuple[int, int], Tuple[float, float]]]:
        for obj_labels in self.sparse_object_labels_batch:
            if obj_labels is not None:
                return obj_labels.input_size_hw
        return None

    def zoom_in_and_rescale_(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].zoom_in_and_rescale_(*args, **kwargs)
        # We may have deleted labels. If no labels are left, set the object to None
        self.set_empty_labels_to_none_()

    def zoom_out_and_rescale_(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].zoom_out_and_rescale_(*args, **kwargs)

    def rotate_(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].rotate_(*args, **kwargs)

    def scale_(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].scale_(*args, **kwargs)
        # We may have deleted labels. If no labels are left, set the object to None
        self.set_empty_labels_to_none_()

    def flip_lr_(self):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].flip_lr_()

    def to(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].to(*args, **kwargs)
        return self

    def get_valid_labels_and_batch_indices(self) -> Tuple[List[ObjectLabels], List[int]]:
        out = list()
        valid_indices = list()
        for idx, label in enumerate(self.sparse_object_labels_batch):
            if label is not None:
                out.append(label)
                valid_indices.append(idx)
        return out, valid_indices

    @staticmethod
    def transpose_list(list_of_sparsely_batched_object_labels: List[SparselyBatchedObjectLabels]) \
            -> List[SparselyBatchedObjectLabels]:
        return [SparselyBatchedObjectLabels(list(labels_as_tuple)) for labels_as_tuple \
                in zip(*list_of_sparsely_batched_object_labels)]
