"""
Compute the COCO metric on bounding box files by matching timestamps

Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os

import numpy as np
from pycocotools.coco import COCO

try:
    coco_eval_type = 'cpp-based'
    from detectron2.evaluation.fast_eval_api import COCOeval_opt as COCOeval
except ImportError:
    coco_eval_type = 'python-based'
    from pycocotools.cocoeval import COCOeval
print(f'Using {coco_eval_type} detection evaluation')


def evaluate_detection(gt_boxes_list, dt_boxes_list, classes=("car", "pedestrian"), height=240, width=304,
                       time_tol=50000, return_aps: bool = True):
    """
    Compute detection KPIs on list of boxes in the numpy format, using the COCO python API
    https://github.com/cocodataset/cocoapi
    KPIs are only computed on timestamps where there is actual at least one box
    (fully empty frames are not considered)

    :param gt_boxes_list: list of numpy array for GT boxes (one per file)
    :param dt_boxes_list: list of numpy array for detected boxes
    :param classes: iterable of classes names
    :param height: int for box size statistics
    :param width: int for box size statistics
    :param time_tol: int size of the temporal window in micro seconds to look for a detection around a gt box
    """
    flattened_gt = []
    flattened_dt = []
    for gt_boxes, dt_boxes in zip(gt_boxes_list, dt_boxes_list):
        assert np.all(gt_boxes['t'][1:] >= gt_boxes['t'][:-1])
        assert np.all(dt_boxes['t'][1:] >= dt_boxes['t'][:-1])

        all_ts = np.unique(gt_boxes['t'])
        n_steps = len(all_ts)

        gt_win, dt_win = _match_times(all_ts, gt_boxes, dt_boxes, time_tol)
        flattened_gt = flattened_gt + gt_win
        flattened_dt = flattened_dt + dt_win
    return _coco_eval(flattened_gt, flattened_dt, height, width, labelmap=classes, return_aps=return_aps)


def _match_times(all_ts, gt_boxes, dt_boxes, time_tol):
    """
    match ground truth boxes and ground truth detections at all timestamps using a specified tolerance
    return a list of boxes vectors
    """
    gt_size = len(gt_boxes)
    dt_size = len(dt_boxes)

    windowed_gt = []
    windowed_dt = []

    low_gt, high_gt = 0, 0
    low_dt, high_dt = 0, 0
    for ts in all_ts:

        while low_gt < gt_size and gt_boxes[low_gt]['t'] < ts:
            low_gt += 1
        # the high index is at least as big as the low one
        high_gt = max(low_gt, high_gt)
        while high_gt < gt_size and gt_boxes[high_gt]['t'] <= ts:
            high_gt += 1

        # detection are allowed to be inside a window around the right detection timestamp
        low = ts - time_tol
        high = ts + time_tol
        while low_dt < dt_size and dt_boxes[low_dt]['t'] < low:
            low_dt += 1
        # the high index is at least as big as the low one
        high_dt = max(low_dt, high_dt)
        while high_dt < dt_size and dt_boxes[high_dt]['t'] <= high:
            high_dt += 1

        windowed_gt.append(gt_boxes[low_gt:high_gt])
        windowed_dt.append(dt_boxes[low_dt:high_dt])

    return windowed_gt, windowed_dt


def _coco_eval(gts, detections, height, width, labelmap=("car", "pedestrian"), return_aps: bool = True):
    """simple helper function wrapping around COCO's Python API
    :params:  gts iterable of numpy boxes for the ground truth
    :params:  detections iterable of numpy boxes for the detections
    :params:  height int
    :params:  width int
    :params:  labelmap iterable of class labels
    """
    categories = [{"id": id + 1, "name": class_name, "supercategory": "none"}
                  for id, class_name in enumerate(labelmap)]

    num_detections = 0
    for detection in detections:
        num_detections += detection.size

    # Meaning: https://cocodataset.org/#detection-eval
    out_keys = ('AP', 'AP_50', 'AP_75', 'AP_S', 'AP_M', 'AP_L')
    out_dict = {k: 0.0 for k in out_keys}

    if num_detections == 0:
        # Corner case at the very beginning of the training.
        print('no detections for evaluation found.')
        return out_dict if return_aps else None

    dataset, results = _to_coco_format(gts, detections, categories, height=height, width=width)

    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    coco_pred = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = np.arange(1, len(gts) + 1, dtype=int)
    coco_eval.evaluate()
    coco_eval.accumulate()
    if return_aps:
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            # info: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
            coco_eval.summarize()
        for idx, key in enumerate(out_keys):
            out_dict[key] = coco_eval.stats[idx]
        return out_dict
    # Print the whole summary instead without return
    coco_eval.summarize()


def coco_eval_return_metrics(coco_eval: COCOeval):
    pass


def _to_coco_format(gts, detections, categories, height=240, width=304):
    """
    utilitary function producing our data in a COCO usable format
    """
    annotations = []
    results = []
    images = []

    # to dictionary
    for image_id, (gt, pred) in enumerate(zip(gts, detections)):
        im_id = image_id + 1

        images.append(
            {"date_captured": "2019",
             "file_name": "n.a",
             "id": im_id,
             "license": 1,
             "url": "",
             "height": height,
             "width": width})

        for bbox in gt:
            x1, y1 = bbox['x'], bbox['y']
            w, h = bbox['w'], bbox['h']
            area = w * h

            annotation = {
                "area": float(area),
                "iscrowd": False,
                "image_id": im_id,
                "bbox": [x1, y1, w, h],
                "category_id": int(bbox['class_id']) + 1,
                "id": len(annotations) + 1
            }
            annotations.append(annotation)

        for bbox in pred:
            image_result = {
                'image_id': im_id,
                'category_id': int(bbox['class_id']) + 1,
                'score': float(bbox['class_confidence']),
                'bbox': [bbox['x'], bbox['y'], bbox['w'], bbox['h']],
            }
            results.append(image_result)

    dataset = {"info": {},
               "licenses": [],
               "type": 'instances',
               "images": images,
               "annotations": annotations,
               "categories": categories}
    return dataset, results
