"""
This is a modified version of the Pytorch Lightning logger
"""

import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from weakref import ReferenceType

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

pl_is_ge_1_6 = float(pl.__version__[:3]) >= 1.6
assert pl_is_ge_1_6

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.logger import rank_zero_experiment, Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn
from pytorch_lightning.utilities.logger import _add_prefix, _convert_params, _flatten_dict, _sanitize_callable_params

import wandb
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run


class WandbLogger(Logger):
    LOGGER_JOIN_CHAR = "-"
    STEP_METRIC = "trainer/global_step"

    def __init__(
            self,
            name: Optional[str] = None,
            project: Optional[str] = None,
            group: Optional[str] = None,
            wandb_id: Optional[str] = None,
            prefix: Optional[str] = "",
            log_model: Optional[bool] = True,
            save_last_only_final: Optional[bool] = False,
            config_args: Optional[Dict[str, Any]] = None,
            **kwargs,
    ):
        super().__init__()
        self._experiment = None
        self._log_model = log_model
        self._prefix = prefix
        self._logged_model_time = {}
        self._checkpoint_callback = None
        # Save last is determined by the checkpoint callback argument
        self._save_last = None
        # Whether to save the last checkpoint continuously (more storage) or only when the run is aborted
        self._save_last_only_final = save_last_only_final
        # Save the configuration args (e.g. parsed arguments) and log it in wandb
        self._config_args = config_args
        # set wandb init arguments
        self._wandb_init = dict(
            name=name,
            project=project,
            group=group,
            id=wandb_id,
            resume="allow",
            save_code=True,
        )
        self._wandb_init.update(**kwargs)
        # extract parameters
        self._name = self._wandb_init.get("name")
        self._id = self._wandb_init.get("id")
        # for save_top_k
        self._public_run = None

        # start wandb run (to create an attach_id for distributed modes)
        wandb.require("service")
        _ = self.experiment

    def get_checkpoint(self, artifact_name: str, artifact_filepath: Optional[Path] = None) -> Path:
        artifact = self.experiment.use_artifact(artifact_name)
        if artifact_filepath is None:
            assert artifact is not None, 'You are probably using DDP, ' \
                                         'in which case you should provide an artifact filepath.'
            # TODO: specify download directory
            artifact_dir = artifact.download()
            artifact_filepath = next(Path(artifact_dir).iterdir())
        assert artifact_filepath.exists()
        assert artifact_filepath.suffix == '.ckpt'
        return artifact_filepath

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # args needed to reload correct experiment
        if self._experiment is not None:
            state["_id"] = getattr(self._experiment, "id", None)
            state["_attach_id"] = getattr(self._experiment, "_attach_id", None)
            state["_name"] = self._experiment.name

        # cannot be pickled
        state["_experiment"] = None
        return state

    @property
    @rank_zero_experiment
    def experiment(self) -> Run:
        if self._experiment is None:
            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                self._experiment = wandb.init(**self._wandb_init)
                if self._config_args is not None:
                    self._experiment.config.update(self._config_args, allow_val_change=True)

                # define default x-axis
                if isinstance(self._experiment, (Run, RunDisabled)) and getattr(
                        self._experiment, "define_metric", None
                ):
                    self._experiment.define_metric(self.STEP_METRIC)
                    self._experiment.define_metric("*", step_metric=self.STEP_METRIC, step_sync=True)

        assert isinstance(self._experiment, (Run, RunDisabled))
        return self._experiment

    def watch(self, model: nn.Module, log: str = 'all', log_freq: int = 100, log_graph: bool = True):
        self.experiment.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)

    def add_step_metric(self, input_dict: dict, step: int) -> None:
        input_dict.update({self.STEP_METRIC: step})

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)
        params = _sanitize_callable_params(params)
        self.experiment.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        if step is not None:
            self.add_step_metric(metrics, step)
            self.experiment.log({**metrics}, step=step)
        else:
            self.experiment.log(metrics)

    @rank_zero_only
    def log_images(self, key: str, images: List[Any], step: Optional[int] = None, **kwargs: str) -> None:
        """Log images (tensors, numpy arrays, PIL Images or file paths).
        Optional kwargs are lists passed to each image (ex: caption, masks, boxes).
        
        How to use: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.loggers.wandb.html#weights-and-biases-logger
        Taken from: https://github.com/PyTorchLightning/pytorch-lightning/blob/11e289ad9f95f5fe23af147fa4edcc9794f9b9a7/pytorch_lightning/loggers/wandb.py#L420
        """
        if not isinstance(images, list):
            raise TypeError(f'Expected a list as "images", found {type(images)}')
        n = len(images)
        for k, v in kwargs.items():
            if len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        kwarg_list = [{k: kwargs[k][i] for k in kwargs.keys()} for i in range(n)]
        metrics = {key: [wandb.Image(img, **kwarg) for img, kwarg in zip(images, kwarg_list)]}
        self.log_metrics(metrics, step)

    @rank_zero_only
    def log_videos(self,
                   key: str,
                   videos: List[Union[np.ndarray, str]],
                   step: Optional[int] = None,
                   captions: Optional[List[str]] = None,
                   fps: int = 4,
                   format_: Optional[str] = None):
        """
        :param video: List[(T,C,H,W)] or List[(N,T,C,H,W)]
        :param captions: List[str] or None

        More info: https://docs.wandb.ai/ref/python/data-types/video and
        https://docs.wandb.ai/guides/track/log/media#other-media
        """
        assert isinstance(videos, list)
        if captions is not None:
            assert isinstance(captions, list)
            assert len(captions) == len(videos)
        wandb_videos = list()
        for idx, video in enumerate(videos):
            caption = captions[idx] if captions is not None else None
            wandb_videos.append(wandb.Video(data_or_path=video, caption=caption, fps=fps, format=format_))
        self.log_metrics(metrics={key: wandb_videos}, step=step)

    @property
    def name(self) -> Optional[str]:
        # This function seems to be only relevant if LoggerCollection is used.
        # don't create an experiment if we don't have one
        return self._experiment.project_name() if self._experiment else self._name

    @property
    def version(self) -> Optional[str]:
        # This function seems to be only relevant if LoggerCollection is used.
        # don't create an experiment if we don't have one
        return self._experiment.id if self._experiment else self._id

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: "ReferenceType[ModelCheckpoint]") -> None:
        # log checkpoints as artifacts
        if self._checkpoint_callback is None:
            self._checkpoint_callback = checkpoint_callback
            self._save_last = checkpoint_callback.save_last
        if self._log_model:
            self._scan_and_log_checkpoints(checkpoint_callback, self._save_last and not self._save_last_only_final)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        # log checkpoints as artifacts
        if self._checkpoint_callback and self._log_model:
            self._scan_and_log_checkpoints(self._checkpoint_callback, self._save_last)

    def _get_public_run(self):
        if self._public_run is None:
            experiment = self.experiment
            runpath = experiment._entity + '/' + experiment._project + '/' + experiment._run_id
            api = wandb.Api()
            self._public_run = api.run(path=runpath)
        return self._public_run

    def _num_logged_artifact(self):
        public_run = self._get_public_run()
        return len(public_run.logged_artifacts())

    def _scan_and_log_checkpoints(self, checkpoint_callback: "ReferenceType[ModelCheckpoint]", save_last: bool) -> None:
        assert self._log_model
        if self._checkpoint_callback is None:
            self._checkpoint_callback = checkpoint_callback
            self._save_last = checkpoint_callback.save_last

        checkpoints = {
            checkpoint_callback.best_model_path: checkpoint_callback.best_model_score,
            **checkpoint_callback.best_k_models,
        }
        assert len(checkpoints) <= max(checkpoint_callback.save_top_k, 0)

        if save_last:
            last_model_path = Path(checkpoint_callback.last_model_path)
            if last_model_path.exists():
                checkpoints.update({checkpoint_callback.last_model_path: checkpoint_callback.current_score})
            else:
                print(f'last model checkpoint not found at {checkpoint_callback.last_model_path}')

        checkpoints = sorted(
            ((Path(path).stat().st_mtime, path, score) for path, score in checkpoints.items() if Path(path).is_file()),
            key=lambda x: x[0])
        # Retain only checkpoints that we have not logged before with one exception:
        # If the name is the same (e.g. last checkpoint which should be overwritten),
        # make sure that they are newer than the previously saved checkpoint by checking their modification time
        checkpoints = [ckpt for ckpt in checkpoints if
                       ckpt[1] not in self._logged_model_time.keys() or self._logged_model_time[ckpt[1]] < ckpt[0]]
        # remove checkpoints with undefined (None) score
        checkpoints = [x for x in checkpoints if x[2] is not None]

        num_ckpt_logged_before = self._num_logged_artifact()
        num_new_cktps = len(checkpoints)

        if num_new_cktps == 0:
            return

        # log iteratively all new checkpoints
        for time_, path, score in checkpoints:
            score = score.item() if isinstance(score, torch.Tensor) else score
            is_best = path == checkpoint_callback.best_model_path
            is_last = path == checkpoint_callback.last_model_path
            metadata = ({
                "score": score,
                "original_filename": Path(path).name,
                "ModelCheckpoint": {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }
            )
            aliases = []
            if is_best:
                aliases.append('best')
            if is_last:
                aliases.append('last')
            artifact_name = f'checkpoint-{self.experiment.id}-' + ('last' if is_last else 'topK')
            artifact = wandb.Artifact(name=artifact_name, type='model', metadata=metadata)
            assert Path(path).exists()
            artifact.add_file(path, name=f'{self.experiment.id}.ckpt')
            self.experiment.log_artifact(artifact, aliases=aliases)
            # remember logged models - timestamp needed in case filename didn't change (last.ckpt or custom name)
            self._logged_model_time[path] = time_

        timeout = 20
        time_spent = 0
        while self._num_logged_artifact() < num_ckpt_logged_before + num_new_cktps:
            time.sleep(1)
            time_spent += 1
            if time_spent >= timeout:
                rank_zero_warn("Timeout: Num logged artifacts never reached expected value.")
                print(f'self._num_logged_artifact() = {self._num_logged_artifact()}')
                print(f'num_ckpt_logged_before = {num_ckpt_logged_before}')
                print(f'num_new_cktps = {num_new_cktps}')
                break

        try:
            self._rm_but_top_k(checkpoint_callback.save_top_k)
        except KeyError:
            pass

    def _rm_but_top_k(self, top_k: int):
        # top_k == -1: save all models
        # top_k == 0: no models saved at all. The checkpoint callback does not return checkpoints.
        # top_k > 0: keep only top k models (last and best will not be deleted)
        def is_last(artifact):
            return 'last' in artifact.aliases

        def is_best(artifact):
            return 'best' in artifact.aliases

        def try_delete(artifact):
            try:
                artifact.delete(delete_aliases=True)
            except wandb.errors.CommError:
                print(f'Failed to delete artifact {artifact.name} due to wandb.errors.CommError')

        public_run = self._get_public_run()

        score2art = list()
        for artifact in public_run.logged_artifacts():
            score = artifact.metadata['score']
            original_filename = artifact.metadata['original_filename']
            if score == 'Infinity':
                print(
                    f'removing INF artifact (name, score, original_filename): ({artifact.name}, {score}, {original_filename})')
                try_delete(artifact)
                continue
            if score is None:
                print(
                    f'removing None artifact (name, score, original_filename): ({artifact.name}, {score}, {original_filename})')
                try_delete(artifact)
                continue
            score2art.append((score, artifact))

        # From high score to low score
        score2art.sort(key=lambda x: x[0], reverse=True)

        count = 0
        for score, artifact in score2art:
            original_filename = artifact.metadata['original_filename']
            if 'last' in original_filename and not is_last(artifact):
                try_delete(artifact)
                continue
            if is_last(artifact):
                continue
            count += 1
            if is_best(artifact):
                continue
            # if top_k == -1, we do not delete anything
            if 0 <= top_k < count:
                try_delete(artifact)
