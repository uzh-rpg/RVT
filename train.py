import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from pytorch_lightning.strategies import DDPStrategy

from callbacks.custom import get_ckpt_callback, get_viz_callback
from callbacks.gradflow import GradFlowLogCallback
from config.modifier import dynamically_modify_train_config
from data.utils.types import DatasetSamplingMode
from loggers.utils import get_wandb_logger, get_ckpt_path
from modules.utils.fetch import fetch_data_module, fetch_model_module


@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # ---------------------
    # Reproducibility
    # ---------------------
    dataset_train_sampling = config.dataset.train.sampling
    assert dataset_train_sampling in iter(DatasetSamplingMode)
    disable_seed_everything = dataset_train_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.MIXED)
    if disable_seed_everything:
        print('Disabling PL seed everything because of unresolved issues with shuffling during training on streaming '
              'datasets')
    seed = config.reproduce.seed_everything
    if seed is not None and not disable_seed_everything:
        assert isinstance(seed, int)
        print(f'USING pl.seed_everything WITH {seed=}')
        pl.seed_everything(seed=seed, workers=True)

    # ---------------------
    # DDP
    # ---------------------
    gpu_config = config.hardware.gpus
    gpus = OmegaConf.to_container(gpu_config) if OmegaConf.is_config(gpu_config) else gpu_config
    gpus = gpus if isinstance(gpus, list) else [gpus]
    distributed_backend = config.hardware.dist_backend
    assert distributed_backend in ('nccl', 'gloo'), f'{distributed_backend=}'
    strategy = DDPStrategy(process_group_backend=distributed_backend,
                           find_unused_parameters=False,
                           gradient_as_bucket_view=True) if len(gpus) > 1 else None

    # ---------------------
    # Data
    # ---------------------
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    logger = get_wandb_logger(config)
    ckpt_path = None
    if config.wandb.artifact_name is not None:
        ckpt_path = get_ckpt_path(logger, wandb_config=config.wandb)

    # ---------------------
    # Model
    # ---------------------
    module = fetch_model_module(config=config)
    if ckpt_path is not None and config.wandb.resume_only_weights:
        print('Resuming only the weights instead of the full training state')
        module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config})
        ckpt_path = None

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = list()
    callbacks.append(get_ckpt_callback(config))
    callbacks.append(GradFlowLogCallback(config.logging.train.log_model_every_n_steps))
    if config.training.lr_scheduler.use:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    if config.logging.train.high_dim.enable or config.logging.validation.high_dim.enable:
        viz_callback = get_viz_callback(config=config)
        callbacks.append(viz_callback)
    callbacks.append(ModelSummary(max_depth=2))

    logger.watch(model=module, log='all', log_freq=config.logging.train.log_model_every_n_steps, log_graph=True)

    # ---------------------
    # Training
    # ---------------------

    val_check_interval = config.validation.val_check_interval
    check_val_every_n_epoch = config.validation.check_val_every_n_epoch
    assert val_check_interval is None or check_val_every_n_epoch is None

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        enable_checkpointing=True,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        default_root_dir=None,
        devices=gpus,
        gradient_clip_val=config.training.gradient_clip_val,
        gradient_clip_algorithm='value',
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.validation.limit_val_batches,
        logger=logger,
        log_every_n_steps=config.logging.train.log_every_n_steps,
        plugins=None,
        precision=config.training.precision,
        max_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps,
        strategy=strategy,
        sync_batchnorm=False if strategy is None else True,
        move_metrics_to_cpu=False,
        benchmark=config.reproduce.benchmark,
        deterministic=config.reproduce.deterministic_flag,
    )
    trainer.fit(model=module, ckpt_path=ckpt_path, datamodule=data_module)


if __name__ == '__main__':
    main()
