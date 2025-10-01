# Start from an official NVIDIA CUDA 11.8 base image for Ubuntu 22.04.
# This image includes CUDA Toolkit 11.8 and cuDNN 8 (for deep learning).
# 'runtime' images are smaller and suitable for running applications.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variable for non-interactive apt commands to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory for your application inside the container
WORKDIR /app

# --- Core System Setup: Install essential tools including SSH, curl, build-essentials, and git ---
# build-essential is crucial for compiling Python packages with native extensions.
# git is often needed for cloning repositories if your environment.yml pulls from git.
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    sudo \
    curl \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH for root login (as was in your original)
RUN mkdir /var/run/sshd && \
    mkdir -p /root/.ssh && chmod 700 /root/.ssh && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# --- Install Miniconda and Mamba ---
# The nvidia/cuda image does NOT come with Conda/Mamba pre-installed, so we add it.
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p "$CONDA_DIR" && \
    rm ~/miniconda.sh

# Add Conda's bin directory to the PATH for the current shell and future RUN commands
ENV PATH="$CONDA_DIR/bin:$PATH"

# Configure Conda channels for better package resolution
RUN conda config --add channels conda-forge \
    && conda config --set channel_priority strict

RUN set -eux; \
    conda update -n base -c defaults conda && \
    conda install -y -c conda-forge mamba && \
    conda clean --all -f -y
# --- Conda Environment Creation and Pip Installation ---
# Copy your environment.yml file into the container
COPY environment.yml .

# Create the Conda environment 'nougat-gpu' from your environment.yml
# This step also handles pip dependencies declared in environment.yml's pip section.
RUN set -eux; \
    mamba env remove --name nougat-gpu --yes || true; \
    mamba env create -f environment.yml --name nougat-gpu && \
    # Install yq for parsing environment.yml to extract pip dependencies
    YQ_VERSION=v4.45.4; \
    curl -L https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/yq_linux_amd64 -o /usr/local/bin/yq && \
    chmod +x /usr/local/bin/yq && \
    # Extract pip dependencies from environment.yml and install them into the newly created conda env
    /bin/bash -c "source \"$CONDA_DIR/bin/activate\" nougat-gpu && \
                  /usr/local/bin/yq '.dependencies[] | select(has(\"pip\")) | .pip[]' environment.yml | xargs -n 1 pip install --no-cache-dir" && \
    conda clean --all -f -y

# --- Set default shell to enter the 'nougat-gpu' Conda environment ---
# This ensures all subsequent commands (like CMD) and interactive shells are within this environment.
SHELL ["conda", "run", "--no-capture-output", "-n", "nougat-gpu", "/bin/bash", "-c"]

# --- Copy app files ---
# These files will be copied into the WORKDIR (/app)
COPY wav2seg_local_windows.py .
COPY data/ ./data/

# --- Expose SSH port ---
EXPOSE 22

# --- Start SSH service and then launch your Python application in the foreground ---
# 'tail -f /dev/null' keeps the container running even if the Python script finishes,
# allowing SSH access to persist.
CMD ["service ssh start && python wav2seg_local_windows.py && tail -f /dev/null"]