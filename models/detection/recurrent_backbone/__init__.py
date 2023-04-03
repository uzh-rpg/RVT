from omegaconf import DictConfig

from .maxvit_rnn import RNNDetector as MaxViTRNNDetector


def build_recurrent_backbone(backbone_cfg: DictConfig):
    name = backbone_cfg.name
    if name == 'MaxViTRNN':
        return MaxViTRNNDetector(backbone_cfg)
    else:
        raise NotImplementedError
