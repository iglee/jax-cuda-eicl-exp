ARG CUDA="11.7.1"
ARG CUDNN="8"
ARG TAG="devel"
ARG OS="ubuntu20.04"
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-${TAG}-${OS}

RUN apt-get update && \
    apt-get install -y --fix-missing \
        git \
        vim \
        htop \
        python3 \
        wget \
        tmux \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

ARG USER_ID
ARG GROUP_ID
ARG NAME
RUN groupadd --gid ${GROUP_ID} ${NAME}
RUN useradd \
    --no-log-init \
    --create-home \
    --uid ${USER_ID} \
    --gid ${GROUP_ID} \
    -s /bin/sh ${NAME}

ARG WORKDIR_PATH
WORKDIR ${WORKDIR_PATH}

# Copy the environment YAML file into the container
COPY environment.yaml .

# Create the conda environment
RUN conda env create -f environment.yaml

# Activate the environment (you can use this command later in the container)
SHELL ["conda", "run", "-n", "eicl_venv", "/bin/bash", "-c"]




ARG JAX_CUDA_CUDNN="cuda"
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "jax[$JAX_CUDA_CUDNN]" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

CMD ["conda", "run", "-n", "eicl_venv", "/bin/bash"]