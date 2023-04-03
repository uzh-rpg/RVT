from functools import partial
from typing import Any, Dict, Optional, Union

import math
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from data.genx_utils.collate import custom_collate_rnd, custom_collate_streaming
from data.genx_utils.dataset_rnd import build_random_access_dataset, get_weighted_random_sampler, CustomConcatDataset
from data.genx_utils.dataset_streaming import build_streaming_dataset
from data.utils.spatial import get_dataloading_hw
from data.utils.types import DatasetMode, DatasetSamplingMode


def get_dataloader_kwargs(dataset: Union[Dataset, CustomConcatDataset],
                          sampling_mode: DatasetSamplingMode,
                          dataset_mode: DatasetMode,
                          dataset_config: DictConfig,
                          batch_size: int,
                          num_workers: int) -> Dict[str, Any]:
    if dataset_mode == DatasetMode.TRAIN:
        if sampling_mode == DatasetSamplingMode.STREAM:
            return dict(
                dataset=dataset,
                batch_size=None,
                shuffle=False,  # Done already in the streaming datapipe
                num_workers=num_workers,
                pin_memory=False,
                drop_last=False,  # Cannot be done with streaming datapipes
                collate_fn=custom_collate_streaming,
            )
        if sampling_mode == DatasetSamplingMode.RANDOM:
            use_weighted_rnd_sampling = dataset_config.train.random.weighted_sampling
            sampler = get_weighted_random_sampler(dataset) if use_weighted_rnd_sampling else None
            return dict(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=sampler is None,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=True,  # Maintain the same batch size for logging
                collate_fn=custom_collate_rnd,
            )
        raise NotImplementedError
    elif dataset_mode in (DatasetMode.VALIDATION, DatasetMode.TESTING):
        if sampling_mode == DatasetSamplingMode.STREAM:
            return dict(
                dataset=dataset,
                batch_size=None,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=False,  # Cannot be done with streaming datapipes
                collate_fn=custom_collate_streaming,
            )
        if sampling_mode == DatasetSamplingMode.RANDOM:
            return dict(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=True,  # Maintain the same batch size for logging
                collate_fn=custom_collate_rnd,
            )
        raise NotImplementedError
    raise NotImplementedError


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_config: DictConfig,
                 num_workers_train: int,
                 num_workers_eval: int,
                 batch_size_train: int,
                 batch_size_eval: int):
        super().__init__()
        assert num_workers_train >= 0
        assert num_workers_eval >= 0
        assert batch_size_train >= 1
        assert batch_size_eval >= 1

        self.dataset_config = dataset_config
        self.train_sampling_mode = dataset_config.train.sampling
        self.eval_sampling_mode = dataset_config.eval.sampling

        assert self.train_sampling_mode in iter(DatasetSamplingMode)
        assert self.eval_sampling_mode in (DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)

        # In DDP all configs are per process/GPU (num_workers, batch_size, ...).
        self.overall_batch_size_train = batch_size_train
        self.overall_batch_size_eval = batch_size_eval
        self.overall_num_workers_train = num_workers_train
        self.overall_num_workers_eval = num_workers_eval

        if self.eval_sampling_mode == DatasetSamplingMode.STREAM:
            self.build_eval_dataset = partial(build_streaming_dataset,
                                              batch_size=self.overall_batch_size_eval,
                                              num_workers=self.overall_num_workers_eval)
        elif self.eval_sampling_mode == DatasetSamplingMode.RANDOM:
            self.build_eval_dataset = build_random_access_dataset
        else:
            raise NotImplementedError

        self.sampling_mode_2_dataset = dict()
        self.sampling_mode_2_train_workers = dict()
        self.sampling_mode_2_train_batch_size = dict()
        self.validation_dataset = None
        self.test_dataset = None

    def get_dataloading_hw(self):
        return get_dataloading_hw(dataset_config=self.dataset_config)

    def set_mixed_sampling_mode_variables_for_train(self):
        assert self.overall_batch_size_train >= 2, 'Cannot use mixed mode with batch size smaller than 2'
        assert self.overall_num_workers_train >= 2, 'Cannot use mixed mode with num workers smaller than 2'
        weight_random = self.dataset_config.train.mixed.w_random
        weight_stream = self.dataset_config.train.mixed.w_stream
        assert weight_random > 0
        assert weight_stream > 0

        # Set batch size according to weights.
        bs_rnd = min(round(self.overall_batch_size_train * weight_random / (weight_stream + weight_random)),
                     self.overall_batch_size_train - 1)
        bs_str = self.overall_batch_size_train - bs_rnd
        self.sampling_mode_2_train_batch_size[DatasetSamplingMode.RANDOM] = bs_rnd
        self.sampling_mode_2_train_batch_size[DatasetSamplingMode.STREAM] = bs_str

        # Set num workers according to batch size. Random sampling typically takes longer than stream sampling!
        workers_rnd = min(math.ceil(self.overall_num_workers_train * bs_rnd / self.overall_batch_size_train),
                          self.overall_num_workers_train - 1)
        workers_str = self.overall_num_workers_train - workers_rnd
        self.sampling_mode_2_train_workers[DatasetSamplingMode.RANDOM] = workers_rnd
        self.sampling_mode_2_train_workers[DatasetSamplingMode.STREAM] = workers_str

        print(f'[Train] Local batch size for:\nstream sampling:\t{bs_str}\nrandom sampling:\t{bs_rnd}\n'
              f'[Train] Local num workers for:\nstream sampling:\t{workers_str}\nrandom sampling:\t{workers_rnd}')

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            if self.train_sampling_mode == DatasetSamplingMode.MIXED:
                self.set_mixed_sampling_mode_variables_for_train()
            else:
                self.sampling_mode_2_train_workers[self.train_sampling_mode] = self.overall_num_workers_train
                self.sampling_mode_2_train_batch_size[self.train_sampling_mode] = self.overall_batch_size_train
            # This code is a bit hacky because at this point we not use DatasetSamplingMode.MIXED anymore
            # because we split it up into random and streaming. DatasetSamplingMode.MIXED was just used to determine
            # whether we use both or not.
            if self.train_sampling_mode in (DatasetSamplingMode.RANDOM, DatasetSamplingMode.MIXED):
                self.sampling_mode_2_dataset[DatasetSamplingMode.RANDOM] = \
                    build_random_access_dataset(dataset_mode=DatasetMode.TRAIN, dataset_config=self.dataset_config)
            if self.train_sampling_mode in (DatasetSamplingMode.STREAM, DatasetSamplingMode.MIXED):
                self.sampling_mode_2_dataset[DatasetSamplingMode.STREAM] = \
                    build_streaming_dataset(
                        dataset_mode=DatasetMode.TRAIN, dataset_config=self.dataset_config,
                        batch_size=self.sampling_mode_2_train_batch_size[DatasetSamplingMode.STREAM],
                        num_workers=self.sampling_mode_2_train_workers[DatasetSamplingMode.STREAM])

            self.validation_dataset = self.build_eval_dataset(dataset_mode=DatasetMode.VALIDATION,
                                                              dataset_config=self.dataset_config)
        elif stage == 'validate':
            self.validation_dataset = self.build_eval_dataset(dataset_mode=DatasetMode.VALIDATION,
                                                              dataset_config=self.dataset_config)
        elif stage == 'test':
            self.test_dataset = self.build_eval_dataset(dataset_mode=DatasetMode.TESTING,
                                                        dataset_config=self.dataset_config)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        train_loaders = dict()
        for sampling_mode, dataset in self.sampling_mode_2_dataset.items():
            train_loaders[sampling_mode] = DataLoader(
                **get_dataloader_kwargs(dataset=dataset, sampling_mode=sampling_mode, dataset_mode=DatasetMode.TRAIN,
                                        dataset_config=self.dataset_config,
                                        batch_size=self.sampling_mode_2_train_batch_size[sampling_mode],
                                        num_workers=self.sampling_mode_2_train_workers[sampling_mode]))
        if len(train_loaders) == 1:
            train_loaders = next(iter(train_loaders.values()))
            # Returns a single dataloader.
            return train_loaders
        assert len(train_loaders) == 2
        # Returns a mapping from dataset sampling modes to dataloader.
        return train_loaders

    def val_dataloader(self):
        return DataLoader(**get_dataloader_kwargs(
            dataset=self.validation_dataset, sampling_mode=self.eval_sampling_mode, dataset_mode=DatasetMode.VALIDATION,
            dataset_config=self.dataset_config, batch_size=self.overall_batch_size_eval,
            num_workers=self.overall_num_workers_eval))

    def test_dataloader(self):
        return DataLoader(**get_dataloader_kwargs(
            dataset=self.test_dataset, sampling_mode=self.eval_sampling_mode, dataset_mode=DatasetMode.TESTING,
            dataset_config=self.dataset_config, batch_size=self.overall_batch_size_eval,
            num_workers=self.overall_num_workers_eval))
