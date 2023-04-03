from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from callbacks.utils.visualization import get_grad_flow_figure


class GradFlowLogCallback(Callback):
    def __init__(self, log_every_n_train_steps: int):
        super().__init__()
        assert log_every_n_train_steps > 0
        self.log_every_n_train_steps = log_every_n_train_steps

    @rank_zero_only
    def on_before_zero_grad(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: Any) -> None:
        # NOTE: before we had this in the on_after_backward callback.
        # This was fine for fp32 but showed unscaled gradients for fp16.
        # That is why we move it to on_before_zero_grad where gradients are scaled.
        global_step = trainer.global_step
        if global_step % self.log_every_n_train_steps != 0:
            return
        named_parameters = pl_module.named_parameters()
        figure = get_grad_flow_figure(named_parameters)
        trainer.logger.log_metrics({'train/gradients': figure}, step=global_step)
