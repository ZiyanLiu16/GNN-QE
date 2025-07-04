# # Use an official CUDA base image with Python 3.8
# FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
# FROM nvcr.io/nvidia/pytorch:21.12-py3
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8, pip, and system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-distutils \
    python3-pip \
    wget git curl ca-certificates \
    libxrender1 libsm6 libxext6 && \
    [ ! -e /usr/bin/python ] && ln -s /usr/bin/python3.8 /usr/bin/python || true && \
    [ ! -e /usr/bin/pip ] && ln -s /usr/bin/pip3 /usr/bin/pip || true && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torch==1.10.2 -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
#     pip install --no-cache-dir torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html && \
#     pip install --no-cache-dir torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html && \
    pip install --no-cache-dir torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html && \
    pip install --no-cache-dir torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html && \
    pip install torchdrug==0.1.3 && \
#     pip install torchdrug==0.2.1 && \
    pip install easydict pyyaml ogb

RUN apt update && apt install vim

RUN mkdir -p /data

# Clone repositories
WORKDIR /workspace
RUN git clone https://github.com/ZiyanLiu16/GNN-QE.git
RUN cd GNN-QE && git pull && git checkout zl/adversarial_training
# RUN cd GNN-QE
# RUN git pull
# RUN git checkout zl/mask_edge_attack

# # Set working directory
# WORKDIR /workspace/GNN-QE
#
# CMD ["/bin/bash"]