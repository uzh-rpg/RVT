## 1. Use an NVIDIA CUDA image as the base
## -----------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

## 2. Create the 'dev' user as 'sudo 'member group
## -----------------------------------
# 'dev' user password: dev
RUN useradd -m dev && \
    echo "dev:dev" | chpasswd && \
    adduser dev sudo

## 3. Set the working directory
## -----------------------------------
WORKDIR /home/dev

# Create a directory for the host shared drive
RUN mkdir /home/dev/host-shared-drive

## 4. Install all required packages
## -----------------------------------
# Install wget and other necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Download and install Miniforge
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O Miniforge3.sh \
    && /bin/bash Miniforge3.sh -b -p /miniforge \
    && rm Miniforge3.sh

# Set the path to include the miniforge bin directory
ENV PATH=/miniforge/bin:$PATH
ENV CUDA_VERSION=11.8

RUN conda create -y -n rvt python=3.9 pip \
    && conda init \
    && . /root/.bashrc \
    && conda activate rvt \
    && conda config --set channel_priority flexible \
    && conda install -y h5py=3.8.0 blosc-hdf5-plugin=1.0.0 \
        hydra-core=1.3.2 einops=0.6.0 torchdata=0.6.0 tqdm numba \
        pytorch=2.0.0 torchvision=0.15.0 pytorch-cuda=$CUDA_VERSION \
        -c pytorch -c nvidia -c conda-forge

RUN python -m pip install pytorch-lightning==1.8.6 wandb==0.14.0 \
    pandas==1.5.3 plotly==5.13.1 opencv-python==4.6.0.66 tabulate==0.9.0 \
    pycocotools==2.0.6 bbox-visualizer==0.1.0 StrEnum==0.4.10

# Detectron2 installation
# Branch origin/pep517 fixes the installation errors on the main branch
# Details: https://github.com/python-poetry/poetry/issues/8330
RUN pip install --no-deps --use-pep517 git+https://github.com/abn/detectron2.git@pep517

# Just to document which ports the container will listen on at runtime
# Make port 80 available outside the container
EXPOSE 80

# Update ownership to ensure the 'dev' user has full access to miniforge 'rvt' environment directory.
RUN chown -R dev /miniforge/envs/rvt

USER dev

# Initialize Conda, reload the shell configuration to apply changes and activate the 'RVT' environment.
# To enable the 'rvt' environment interactively, execute the following command:
# conda init && . /home/dev/.bashrc && conda activate rvt
