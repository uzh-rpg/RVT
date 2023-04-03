from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path
from typing import List

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.genx_utils.sequence_rnd import SequenceForRandomAccess
from data.utils.augmentor import RandomSpatialAugmentorGenX
from data.utils.types import DatasetMode, LoaderDataDictGenX, DatasetType, DataType


class SequenceDataset(Dataset):
    def __init__(self,
                 path: Path,
                 dataset_mode: DatasetMode,
                 dataset_config: DictConfig):
        assert path.is_dir()

        ### extract settings from config ###
        sequence_length = dataset_config.sequence_length
        assert isinstance(sequence_length, int)
        assert sequence_length > 0
        self.output_seq_len = sequence_length

        ev_representation_name = dataset_config.ev_repr_name
        downsample_by_factor_2 = dataset_config.downsample_by_factor_2
        only_load_end_labels = dataset_config.only_load_end_labels

        augm_config = dataset_config.data_augmentation

        ####################################
        if dataset_config.name == 'gen1':
            dataset_type = DatasetType.GEN1
        elif dataset_config.name == 'gen4':
            dataset_type = DatasetType.GEN4
        else:
            raise NotImplementedError
        self.sequence = SequenceForRandomAccess(path=path,
                                                ev_representation_name=ev_representation_name,
                                                sequence_length=sequence_length,
                                                dataset_type=dataset_type,
                                                downsample_by_factor_2=downsample_by_factor_2,
                                                only_load_end_labels=only_load_end_labels)

        self.spatial_augmentor = None
        if dataset_mode == DatasetMode.TRAIN:
            resolution_hw = tuple(dataset_config.resolution_hw)
            assert len(resolution_hw) == 2
            ds_by_factor_2 = dataset_config.downsample_by_factor_2
            if ds_by_factor_2:
                resolution_hw = tuple(x // 2 for x in resolution_hw)
            self.spatial_augmentor = RandomSpatialAugmentorGenX(
                dataset_hw=resolution_hw,
                automatic_randomization=True,
                augm_config=augm_config.random)

    def only_load_labels(self):
        self.sequence.only_load_labels()

    def load_everything(self):
        self.sequence.load_everything()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index: int) -> LoaderDataDictGenX:

        item = self.sequence[index]

        if self.spatial_augmentor is not None and not self.sequence.is_only_loading_labels():
            item = self.spatial_augmentor(item)

        return item


class CustomConcatDataset(ConcatDataset):
    datasets: List[SequenceDataset]

    def __init__(self, datasets: Iterable[SequenceDataset]):
        super().__init__(datasets=datasets)

    def only_load_labels(self):
        for idx, dataset in enumerate(self.datasets):
            self.datasets[idx].only_load_labels()

    def load_everything(self):
        for idx, dataset in enumerate(self.datasets):
            self.datasets[idx].load_everything()


def build_random_access_dataset(dataset_mode: DatasetMode, dataset_config: DictConfig) -> CustomConcatDataset:
    dataset_path = Path(dataset_config.path)
    assert dataset_path.is_dir(), f'{str(dataset_path)}'

    mode2str = {DatasetMode.TRAIN: 'train',
                DatasetMode.VALIDATION: 'val',
                DatasetMode.TESTING: 'test'}

    split_path = dataset_path / mode2str[dataset_mode]
    assert split_path.is_dir()

    seq_datasets = list()
    for entry in tqdm(split_path.iterdir(), desc=f'creating rnd access {mode2str[dataset_mode]} datasets'):
        seq_datasets.append(SequenceDataset(path=entry, dataset_mode=dataset_mode, dataset_config=dataset_config))

    return CustomConcatDataset(seq_datasets)


def get_weighted_random_sampler(dataset: CustomConcatDataset) -> WeightedRandomSampler:
    class2count = dict()
    ClassAndCount = namedtuple('ClassAndCount', ['class_ids', 'counts'])
    classandcount_list = list()
    print('--- START generating weighted random sampler ---')
    dataset.only_load_labels()
    for idx, data in enumerate(tqdm(dataset, desc='iterate through dataset')):
        labels: SparselyBatchedObjectLabels = data[DataType.OBJLABELS_SEQ]
        label_list, valid_batch_indices = labels.get_valid_labels_and_batch_indices()
        class_ids_seq = list()
        for label in label_list:
            class_ids_numpy = np.asarray(label.class_id.numpy(), dtype='int32')
            class_ids_seq.append(class_ids_numpy)
        class_ids_seq, counts_seq = np.unique(np.concatenate(class_ids_seq), return_counts=True)
        for class_id, count in zip(class_ids_seq, counts_seq):
            class2count[class_id] = class2count.get(class_id, 0) + count
        classandcount_list.append(ClassAndCount(class_ids=class_ids_seq, counts=counts_seq))
    dataset.load_everything()

    class2weight = {}
    for class_id, count in class2count.items():
        count = max(count, 1)
        class2weight[class_id] = 1 / count

    weights = []
    for classandcount in classandcount_list:
        weight = 0
        for class_id, count in zip(classandcount.class_ids, classandcount.counts):
            # Not only weight depending on class but also depending on number of occurrences.
            # This will bias towards sampling "frames" with more bounding boxes.
            weight += class2weight[class_id] * count
        weights.append(weight)

    print('--- DONE generating weighted random sampler ---')
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
