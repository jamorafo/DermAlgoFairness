# GPU-enabled environment for the DermAlgoFairness revised public results workflow.
#
# The host server reports NVIDIA driver 560.35.05 and CUDA 12.6 through nvidia-smi.
# The container uses CUDA 11.8 with cuDNN 8 because this stack is better aligned
# with TensorFlow 2.13 GPU compatibility.

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TZ=America/Toronto

WORKDIR /workspace/DermAlgoFairness

RUN ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        python3 \
        python3-dev \
        python3-pip \
        build-essential \
        git \
        curl \
        wget \
        ca-certificates \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /tmp/requirements.txt

# The repository is mounted at runtime. This image intentionally does not copy
# the full repository so that scripts run against the checked-out Git branch.
CMD ["/bin/bash"]
