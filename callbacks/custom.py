from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint

from callbacks.detection import DetectionVizCallback


def get_ckpt_callback(config: DictConfig) -> ModelCheckpoint:
    model_name = config.model.name

    prefix = 'val'
    if model_name == 'rnndet':
        metric = 'AP'
        mode = 'max'
    else:
        raise NotImplementedError
    ckpt_callback_monitor = prefix + '/' + metric
    filename_monitor_str = prefix + '_' + metric

    ckpt_filename = 'epoch={epoch:03d}-step={step}-' + filename_monitor_str + '={' + ckpt_callback_monitor + ':.2f}'
    cktp_callback = ModelCheckpoint(
        monitor=ckpt_callback_monitor,
        filename=ckpt_filename,
        auto_insert_metric_name=False,  # because backslash would create a directory
        save_top_k=1,
        mode=mode,
        every_n_epochs=config.logging.ckpt_every_n_epochs,
        save_last=True,
        verbose=True)
    cktp_callback.CHECKPOINT_NAME_LAST = 'last_epoch={epoch:03d}-step={step}'
    return cktp_callback


def get_viz_callback(config: DictConfig) -> Callback:
    model_name = config.model.name

    if model_name == 'rnndet':
        return DetectionVizCallback(config=config)
    raise NotImplementedError
