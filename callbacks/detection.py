from enum import Enum, auto
from typing import Any

import torch
from einops import rearrange
from omegaconf import DictConfig

from data.utils.types import ObjDetOutput
from loggers.wandb_logger import WandbLogger
from utils.evaluation.prophesee.visualize.vis_utils import LABELMAP_GEN1, LABELMAP_GEN4_SHORT, draw_bboxes
from .viz_base import VizCallbackBase


class DetectionVizEnum(Enum):
    EV_IMG = auto()
    LABEL_IMG_PROPH = auto()
    PRED_IMG_PROPH = auto()


class DetectionVizCallback(VizCallbackBase):
    def __init__(self, config: DictConfig):
        super().__init__(config=config, buffer_entries=DetectionVizEnum)

        dataset_name = config.dataset.name
        if dataset_name == 'gen1':
            self.label_map = LABELMAP_GEN1
        elif dataset_name == 'gen4':
            self.label_map = LABELMAP_GEN4_SHORT
        else:
            raise NotImplementedError

    def on_train_batch_end_custom(self,
                                  logger: WandbLogger,
                                  outputs: Any,
                                  batch: Any,
                                  log_n_samples: int,
                                  global_step: int) -> None:
        if outputs is None:
            # If we tried to skip the training step (not supported in DDP in PL, atm)
            return
        ev_tensors = outputs[ObjDetOutput.EV_REPR]
        num_samples = len(ev_tensors)
        assert num_samples > 0
        log_n_samples = min(num_samples, log_n_samples)

        merged_img = []
        captions = []
        start_idx = num_samples - 1
        end_idx = start_idx - log_n_samples
        # for sample_idx in range(log_n_samples):
        for sample_idx in range(start_idx, end_idx, -1):
            ev_img = self.ev_repr_to_img(ev_tensors[sample_idx].cpu().numpy())

            predictions_proph = outputs[ObjDetOutput.PRED_PROPH][sample_idx]
            prediction_img = ev_img.copy()
            draw_bboxes(prediction_img, predictions_proph, labelmap=self.label_map)

            labels_proph = outputs[ObjDetOutput.LABELS_PROPH][sample_idx]
            label_img = ev_img.copy()
            draw_bboxes(label_img, labels_proph, labelmap=self.label_map)

            merged_img.append(rearrange([prediction_img, label_img], 'pl H W C -> (pl H) W C', pl=2, C=3))
            captions.append(f'sample_{sample_idx}')

        logger.log_images(key='train/predictions',
                          images=merged_img,
                          caption=captions,
                          step=global_step)

    def on_validation_batch_end_custom(self, batch: Any, outputs: Any):
        if outputs[ObjDetOutput.SKIP_VIZ]:
            return
        ev_tensor = outputs[ObjDetOutput.EV_REPR]
        assert isinstance(ev_tensor, torch.Tensor)

        ev_img = self.ev_repr_to_img(ev_tensor.cpu().numpy())

        predictions_proph = outputs[ObjDetOutput.PRED_PROPH]
        prediction_img = ev_img.copy()
        draw_bboxes(prediction_img, predictions_proph, labelmap=self.label_map)
        self.add_to_buffer(DetectionVizEnum.PRED_IMG_PROPH, prediction_img)

        labels_proph = outputs[ObjDetOutput.LABELS_PROPH]
        label_img = ev_img.copy()
        draw_bboxes(label_img, labels_proph, labelmap=self.label_map)
        self.add_to_buffer(DetectionVizEnum.LABEL_IMG_PROPH, label_img)

    def on_validation_epoch_end_custom(self, logger: WandbLogger):
        pred_imgs = self.get_from_buffer(DetectionVizEnum.PRED_IMG_PROPH)
        label_imgs = self.get_from_buffer(DetectionVizEnum.LABEL_IMG_PROPH)
        assert len(pred_imgs) == len(label_imgs)
        merged_img = []
        captions = []
        for idx, (pred_img, label_img) in enumerate(zip(pred_imgs, label_imgs)):
            merged_img.append(rearrange([pred_img, label_img], 'pl H W C -> (pl H) W C', pl=2, C=3))
            captions.append(f'sample_{idx}')

        logger.log_images(key='val/predictions',
                          images=merged_img,
                          caption=captions)
