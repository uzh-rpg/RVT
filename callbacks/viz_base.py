import random
from enum import Enum
from typing import Any, List, Optional, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch as th
from einops import rearrange, reduce
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from loggers.wandb_logger import WandbLogger


class VizCallbackBase(Callback):
    def __init__(self, config: DictConfig, buffer_entries: Type[Enum]):
        super().__init__()

        self.log_config = config.logging

        self._training_has_started = False
        self._selected_val_batches = False

        self.buffer_entries = buffer_entries
        self._val_batch_indices = list()
        self._buffer = None
        self._reset_buffer()

    def _reset_buffer(self):
        self._buffer = {entry: [] for entry in self.buffer_entries}

    # Functions to be USED in the base class ---------------------------------------------------------------------------

    def add_to_buffer(self, key: Enum, value: Union[np.ndarray, th.Tensor]):
        if isinstance(value, th.Tensor):
            assert not value.requires_grad
            value = value.cpu()
        else:
            assert isinstance(value, np.ndarray)
        assert type(key) == self.buffer_entries
        assert key in self._buffer
        self._buffer[key].append(value)

    def get_from_buffer(self, key: Enum) -> List[th.Tensor]:
        assert type(key) == self.buffer_entries
        return self._buffer[key]

    # Functions to be IMPLEMENTED in the base class --------------------------------------------------------------------

    def on_train_batch_end_custom(self,
                                  logger: WandbLogger,
                                  outputs: Any,
                                  batch: Any,
                                  log_n_samples: int,
                                  global_step: int) -> None:
        raise NotImplementedError

    def on_validation_batch_end_custom(self,
                                       batch: Any,
                                       outputs: Any) -> None:
        raise NotImplementedError

    def on_validation_epoch_end_custom(self,
                                       logger: WandbLogger) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        log_train_hd = self.log_config.train.high_dim
        if not log_train_hd.enable:
            return

        step = trainer.global_step
        assert log_train_hd.every_n_steps > 0
        if step % log_train_hd.every_n_steps != 0:
            return

        n_samples = log_train_hd.n_samples

        logger: Optional[WandbLogger] = trainer.logger
        assert isinstance(logger, WandbLogger)

        global_step = trainer.global_step

        self.on_train_batch_end_custom(
            logger=logger,
            outputs=outputs,
            batch=batch,
            log_n_samples=n_samples,
            global_step=global_step)

    @rank_zero_only
    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Optional[Any],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        log_val_hd = self.log_config.validation.high_dim
        log_freq_val_epochs = log_val_hd.every_n_epochs
        if not log_val_hd.enable:
            return
        if dataloader_idx > 0:
            raise NotImplementedError
        if not self._training_has_started:
            # PL has a short sanity check for validation. Hence, we have to make sure that one training run is done.
            return
        if not self._selected_val_batches:
            # We only want to add validation batch indices during the first true validation run.
            self._val_batch_indices.append(batch_idx)
            return
        assert len(self._val_batch_indices) > 0
        if batch_idx not in self._val_batch_indices:
            return
        if trainer.current_epoch % log_freq_val_epochs != 0:
            return

        self.on_validation_batch_end_custom(batch, outputs)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._reset_buffer()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        log_val_hd = self.log_config.validation.high_dim
        log_n_samples = log_val_hd.n_samples
        log_freq_val_epochs = log_val_hd.every_n_epochs
        if len(self._val_batch_indices) == 0:
            return
        if not self._selected_val_batches:
            random.seed(0)
            num_samples = min(len(self._val_batch_indices), log_n_samples)
            # draw without replacement
            sampled_indices = random.sample(self._val_batch_indices, num_samples)
            self._val_batch_indices = sampled_indices
            self._selected_val_batches = True
            return
        if trainer.current_epoch % log_freq_val_epochs != 0:
            return

        logger: Optional[WandbLogger] = trainer.logger
        assert isinstance(logger, WandbLogger)
        self.on_validation_epoch_end_custom(logger)

    def on_train_batch_start(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        self._training_has_started = True

    @staticmethod
    def ev_repr_to_img(x: np.ndarray):
        ch, ht, wd = x.shape[-3:]
        assert ch > 1 and ch % 2 == 0
        ev_repr_reshaped = rearrange(x, '(posneg C) H W -> posneg C H W', posneg=2)
        img_neg = np.asarray(reduce(ev_repr_reshaped[0], 'C H W -> H W', 'sum'), dtype='int32')
        img_pos = np.asarray(reduce(ev_repr_reshaped[1], 'C H W -> H W', 'sum'), dtype='int32')
        img_diff = img_pos - img_neg
        img = 127 * np.ones((ht, wd, 3), dtype=np.uint8)
        img[img_diff > 0] = 255
        img[img_diff < 0] = 0
        return img
