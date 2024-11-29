# hash:sha256:6a2cbb68a2c7a11a104b6d002fe9190d677ccb34973b5bff02613ec21308f674
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python-opencv=3.2.0+dfsg-4ubuntu0.1 \
        python-tk=2.7.17-1~18.04 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    imageio==2.8.0 \
    imgaug==0.4.0 \
    matplotlib==3.2.2 \
    opencv-python==4.2.0.34 \
    sklearn==0.0 \
    torch==1.5.1
