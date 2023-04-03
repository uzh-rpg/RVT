from typing import Any, Iterator, List, Optional, Type

import torch as th
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import (
    Concater,
    IterableWrapper,
    IterDataPipe,
    Zipper,
)
from torchdata.datapipes.map import MapDataPipe


class DummyIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        super().__init__()
        assert isinstance(source_dp, IterDataPipe)
        self.source_dp = source_dp

    def __iter__(self):
        yield from self.source_dp


class ConcatStreamingDataPipe(IterDataPipe):
    """This Dataset avoids the sharding problem by instantiating randomized stream concatenation at the batch and
    worker level.
    Pros:
    - Every single batch has valid samples. Consequently, the batch size is always constant.
    Cons:
    - There might be repeated samples in a batch. Although they should be different because of data augmentation.
    - Cannot be used for validation or testing because we repeat the dataset multiple times in an epoch.

    TLDR: preferred approach for training but not useful for validation or testing.
    """

    def __init__(self,
                 datapipe_list: List[MapDataPipe],
                 batch_size: int,
                 num_workers: int,
                 augmentation_pipeline: Optional[Type[IterDataPipe]] = None,
                 print_seed_debug: bool = False):
        super().__init__()
        assert batch_size > 0

        if augmentation_pipeline is not None:
            self.augmentation_dp = augmentation_pipeline
        else:
            self.augmentation_dp = DummyIterDataPipe

        # We require MapDataPipes instead of IterDataPipes because IterDataPipes must be deepcopied in each worker.
        # Instead, MapDataPipes can be converted to IterDataPipes in each worker without requiring a deepcopy.
        self.datapipe_list = datapipe_list
        self.batch_size = batch_size

        self.print_seed_debug = print_seed_debug

    @staticmethod
    def random_torch_shuffle_list(data: List[Any]) -> Iterator[Any]:
        assert isinstance(data, List)
        return (data[idx] for idx in th.randperm(len(data)).tolist())

    def _get_zipped_streams(self, datapipe_list: List[MapDataPipe], batch_size: int):
        """Use it only in the iter function of this class!!!
        Reason: randomized shuffling must happen within each worker. Otherwise, the same random order will be used
        for all workers.
        """
        assert isinstance(datapipe_list, List)
        assert batch_size > 0
        streams = Zipper(*(Concater(*(self.augmentation_dp(x.to_iter_datapipe())
                                      for x in self.random_torch_shuffle_list(datapipe_list)))
                           for _ in range(batch_size)))
        return streams

    def _print_seed_debug_info(self):
        worker_info = th.utils.data.get_worker_info()
        local_worker_id = 0 if worker_info is None else worker_info.id

        worker_torch_seed = worker_info.seed
        local_num_workers = 1 if worker_info is None else worker_info.num_workers
        if dist.is_available() and dist.is_initialized():
            global_rank = dist.get_rank()
        else:
            global_rank = 0
        global_worker_id = global_rank * local_num_workers + local_worker_id

        rnd_number = th.randn(1)
        print(f'{worker_torch_seed=},\t{global_worker_id=},\t{global_rank=},\t{local_worker_id=},\t{rnd_number=}',
              flush=True)

    def _get_zipped_streams_with_worker_id(self):
        """Use it only in the iter function of this class!!!
        """
        worker_info = th.utils.data.get_worker_info()
        local_worker_id = 0 if worker_info is None else worker_info.id
        worker_id_stream = IterableWrapper([local_worker_id]).cycle(count=None)
        zipped_stream = self._get_zipped_streams(datapipe_list=self.datapipe_list, batch_size=self.batch_size)
        return zipped_stream.zip(worker_id_stream)

    def __iter__(self):
        if self.print_seed_debug:
            self._print_seed_debug_info()
        return iter(self._get_zipped_streams_with_worker_id())
