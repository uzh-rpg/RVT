from .io.box_filtering import filter_boxes
from .metrics.coco_eval import evaluate_detection


def evaluate_list(result_boxes_list,
                  gt_boxes_list,
                  height: int,
                  width: int,
                  camera: str = 'gen1',
                  apply_bbox_filters: bool = True,
                  downsampled_by_2: bool = False,
                  return_aps: bool = True):
    assert camera in {'gen1', 'gen4'}

    if camera == 'gen1':
        classes = ("car", "pedestrian")
    elif camera == 'gen4':
        classes = ("pedestrian", "two-wheeler", "car")
    else:
        raise NotImplementedError

    if apply_bbox_filters:
        # Default values taken from: https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/0393adea2bf22d833893c8cb1d986fcbe4e6f82d/src/psee_evaluator.py#L23-L24
        min_box_diag = 60 if camera == 'gen4' else 30
        # In the supplementary mat, they say that min_box_side is 20 for gen4.
        min_box_side = 20 if camera == 'gen4' else 10
        if downsampled_by_2:
            assert min_box_diag % 2 == 0
            min_box_diag //= 2
            assert min_box_side % 2 == 0
            min_box_side //= 2

        half_sec_us = int(5e5)
        filter_boxes_fn = lambda x: filter_boxes(x, half_sec_us, min_box_diag, min_box_side)

        gt_boxes_list = map(filter_boxes_fn, gt_boxes_list)
        # NOTE: We also filter the prediction to follow the prophesee protocol of evaluation.
        result_boxes_list = map(filter_boxes_fn, result_boxes_list)

    return evaluate_detection(gt_boxes_list, result_boxes_list,
                              height=height, width=width,
                              classes=classes, return_aps=return_aps)
