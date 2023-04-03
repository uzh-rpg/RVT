from omegaconf import DictConfig

from data.utils.types import DatasetType

_type_2_hw = {
    DatasetType.GEN1: (240, 304),
    DatasetType.GEN4: (720, 1280),
}

_str_2_type = {
    'gen1': DatasetType.GEN1,
    'gen4': DatasetType.GEN4,
}


def get_original_hw(dataset_type: DatasetType):
    return _type_2_hw[dataset_type]


def get_dataloading_hw(dataset_config: DictConfig):
    dataset_name = dataset_config.name
    hw = get_original_hw(dataset_type=_str_2_type[dataset_name])
    downsample_by_factor_2 = dataset_config.downsample_by_factor_2
    if downsample_by_factor_2:
        hw = tuple(x // 2 for x in hw)
    return hw
