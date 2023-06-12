# RVT: Recurrent Vision Transformers for Object Detection with Event Cameras
<p align="center">
  <img src="https://rpg.ifi.uzh.ch/img/papers/arxiv22_detection_mgehrig/combo.png" width="750">
</p>

This is the official Pytorch implementation of the CVPR 2023 paper [Recurrent Vision Transformers for Object Detection with Event Cameras](https://arxiv.org/abs/2212.05598).

Watch the [**video**](https://youtu.be/xZ-pNwHxHgY) for a quick overview.

```bibtex
@InProceedings{Gehrig_2023_CVPR,
  author  = {Mathias Gehrig and Davide Scaramuzza},
  title   = {Recurrent Vision Transformers for Object Detection with Event Cameras},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2023},
}
```

## Conda Installation
We highly recommend to use [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) to reduce the installation time.
```Bash
conda create -y -n rvt python=3.9 pip
conda activate rvt
conda config --set channel_priority flexible

CUDA_VERSION=11.8

conda install -y h5py=3.8.0 blosc-hdf5-plugin=1.0.0 \
hydra-core=1.3.2 einops=0.6.0 torchdata=0.6.0 tqdm numba \
pytorch=2.0.0 torchvision=0.15.0 pytorch-cuda=$CUDA_VERSION \
-c pytorch -c nvidia -c conda-forge

python -m pip install pytorch-lightning==1.8.6 wandb==0.14.0 \
pandas==1.5.3 plotly==5.13.1 opencv-python==4.6.0.66 tabulate==0.9.0 \
pycocotools==2.0.6 bbox-visualizer==0.1.0 StrEnum==0.4.10
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
Detectron2 is not strictly required but speeds up the evaluation.

## Required Data
To evaluate or train RVT you will need to download the required preprocessed datasets:

<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">1 Mpx</th>
<th valign="bottom">Gen1</th>
<tr><td align="left">pre-processed dataset</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen4.tar">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen1.tar">download</a></td>
</tr>
<tr><td align="left">crc32</td>
<td align="center"><tt>c5ec7c38</tt></td>
<td align="center"><tt>5acab6f3</tt></td>
</tr>
</tbody></table>

You may also pre-process the dataset yourself by following the [instructions](scripts/genx/README.md).

## Pre-trained Checkpoints
### 1 Mpx
<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">RVT-Base</th>
<th valign="bottom">RVT-Small</th>
<th valign="bottom">RVT-Tiny</th>
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/checkpoints/1mpx/rvt-b.ckpt">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/checkpoints/1mpx/rvt-s.ckpt">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/checkpoints/1mpx/rvt-t.ckpt">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>72923a</tt></td>
<td align="center"><tt>a94207</tt></td>
<td align="center"><tt>5a3c78</tt></td>
</tr>
</tbody></table>

### Gen1
<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">RVT-Base</th>
<th valign="bottom">RVT-Small</th>
<th valign="bottom">RVT-Tiny</th>
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/checkpoints/gen1/rvt-b.ckpt">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/checkpoints/gen1/rvt-s.ckpt">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/checkpoints/gen1/rvt-t.ckpt">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>839317</tt></td>
<td align="center"><tt>840f2b</tt></td>
<td align="center"><tt>a770b9</tt></td>
</tr>
</tbody></table>

## Evaluation
- Set `DATA_DIR` as the path to either the 1 Mpx or Gen1 dataset directory
- Set `CKPT_PATH` to the path of the *correct* checkpoint matching the choice of the model and dataset.
- Set
  - `MDL_CFG=base`, or
  - `MDL_CFG=small`, or
  - `MDL_CFG=tiny`
  
  to load either the base, small, or tiny model configuration
- Set
  - `USE_TEST=1` to evaluate on the test set, or
  - `USE_TEST=0` to evaluate on the validation set
- Set `GPU_ID` to the PCI BUS ID of the GPU that you want to use. e.g. `GPU_ID=0`.
  Only a single GPU is supported for evaluation
### 1 Mpx
```Bash
python validation.py dataset=gen4 dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
use_test_set=${USE_TEST} hardware.gpus=${GPU_ID} +experiment/gen4="${MDL_CFG}.yaml" \
batch_size.eval=8 model.postprocess.confidence_threshold=0.001
```
### Gen1
```Bash
python validation.py dataset=gen1 dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
use_test_set=${USE_TEST} hardware.gpus=${GPU_ID} +experiment/gen1="${MDL_CFG}.yaml" \
batch_size.eval=8 model.postprocess.confidence_threshold=0.001
```

## Training
- Set `DATA_DIR` as the path to either the 1 Mpx or Gen1 dataset directory
- Set
    - `MDL_CFG=base`, or
    - `MDL_CFG=small`, or
    - `MDL_CFG=tiny`

  to load either the base, small, or tiny model configuration
- Set `GPU_IDS` to the PCI BUS IDs of the GPUs that you want to use. e.g. `GPU_IDS=[0,1]` for using GPU 0 and 1.
  **Using a list of IDS will enable single-node multi-GPU training.**
  Pay attention to the batch size which is defined per GPU:
- Set `BATCH_SIZE_PER_GPU` such that the effective batch size is matching the parameters below.
  The **effective batch size** is (batch size per gpu)*(number of GPUs).
- If you would like to change the effective batch size, we found the following learning rate scaling to work well for 
all models on both datasets:
  
  `lr = 2e-4 * sqrt(effective_batch_size/8)`.
- The training code uses [W&B](https://wandb.ai/) for logging during the training.
Hence, we assume that you have a W&B account. 
  - The training script below will create a new project called `RVT`. Adapt the project name and group name if necessary.
 
### 1 Mpx
- The effective batch size for the 1 Mpx training is 24.
- To train on 2 GPUs using 6 workers per GPU for training and 2 workers per GPU for evaluation:
```Bash
GPU_IDS=[0,1]
BATCH_SIZE_PER_GPU=12
TRAIN_WORKERS_PER_GPU=6
EVAL_WORKERS_PER_GPU=2
python train.py model=rnndet dataset=gen4 dataset.path=${DATA_DIR} wandb.project_name=RVT \
wandb.group_name=1mpx +experiment/gen4="${MDL_CFG}.yaml" hardware.gpus=${GPU_IDS} \
batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU}
```
If you instead want to execute the training on 4 GPUs simply adapt `GPU_IDS` and `BATCH_SIZE_PER_GPU` accordingly:
```Bash
GPU_IDS=[0,1,2,3]
BATCH_SIZE_PER_GPU=6
```
### Gen1
- The effective batch size for the Gen1 training is 8.
- To train on 1 GPU using 6 workers for training and 2 workers for evaluation:
```Bash
GPU_IDS=0
BATCH_SIZE_PER_GPU=8
TRAIN_WORKERS_PER_GPU=6
EVAL_WORKERS_PER_GPU=2
python train.py model=rnndet dataset=gen1 dataset.path=${DATA_DIR} wandb.project_name=RVT \
wandb.group_name=gen1 +experiment/gen1="${MDL_CFG}.yaml" hardware.gpus=${GPU_IDS} \
batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU}
```

## Code Acknowledgments
This project has used code from the following projects:
- [timm](https://github.com/huggingface/pytorch-image-models) for the MaxViT layer implementation in Pytorch
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) for the detection PAFPN/head
