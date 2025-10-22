# Use the full CUDA 12.6.1 development image with cuDNN
FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

# Set CUDA environment variables in Dockerfile
ENV CUDA_HOME=/usr/local/cuda-12.6
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME/nvvm/libdevice

# Set environment variables to avoid timezone/region prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=2/138

# Preconfigure tzdata to avoid manual prompt
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y --no-install-recommends tzdata && \
    dpkg-reconfigure -f noninteractive tzdata

# Install required system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    build-essential \
    curl \
    software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Manually add CUDA repository and install CUDA toolkit
RUN add-apt-repository ppa:graphics-drivers/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-toolkit-12-6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    tensorflow==2.13.0 \
    tensorflow-addons \
    keras \
    opencv-python-headless \
    jupyter \
    transformers

# Set the working directory in the container
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Expose port for Jupyter Notebook
EXPOSE 1111

# Run Jupyter Notebook on port 1111
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=1111", "--no-browser", "--allow-root"]
