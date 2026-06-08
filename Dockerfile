# ======================================================================
# Base image & args
# ======================================================================
ARG ISAACSIM_BASE_IMAGE_ARG=nvcr.io/nvidia/isaac-sim
ARG ISAACSIM_VERSION_ARG=4.5.0
FROM ${ISAACSIM_BASE_IMAGE_ARG}:${ISAACSIM_VERSION_ARG} AS base
ARG ISAACSIM_VERSION_ARG
ENV ISAACSIM_VERSION=${ISAACSIM_VERSION_ARG}

SHELL ["/bin/bash", "-c"]

LABEL version="sslab-isaaclab-1.0"
LABEL description="Isaac Sim container with IsaacLab, RSL-RL, and sslab_isaaclab extension installed."

# Paths (with sensible defaults)
ARG ISAACSIM_ROOT_PATH_ARG=/isaac-sim
ENV ISAACSIM_ROOT_PATH=${ISAACSIM_ROOT_PATH_ARG}

# IsaacLab lives at /workspace/IsaacLab (your repo must include this folder)
ARG ISAACLAB_PATH_ARG=/workspace/IsaacLab
ENV ISAACLAB_PATH=${ISAACLAB_PATH_ARG}

# Docker user home
ARG DOCKER_USER_HOME_ARG=/root
ENV DOCKER_USER_HOME=${DOCKER_USER_HOME_ARG}

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

USER root

# ======================================================================
# System deps + yq (used by your job to read YAML inside the container)
# ======================================================================
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      curl \
      git \
      git-lfs \
      libglib2.0-0 \
      ncurses-term \
      wget \
      ffmpeg \
      libgl1 libsm6 libxext6 libxrender1 \
    && git lfs install \
    && curl -L -o /usr/local/bin/yq \
         https://github.com/mikefarah/yq/releases/download/v4.44.3/yq_linux_amd64 \
    && chmod +x /usr/local/bin/yq \
    && apt -y autoremove && apt clean autoclean \
    && rm -rf /var/lib/apt/lists/*

# ======================================================================
# Make "python"/"pip" wrappers that call Isaac Sim's Python (no symlinks)
# ======================================================================
RUN printf '#!/bin/bash\nexec %s "$@"\n' "/isaac-sim/python.sh" > /usr/local/bin/python \
 && chmod +x /usr/local/bin/python \
 && ln -sf /usr/local/bin/python /usr/local/bin/python3 \
 && printf '#!/bin/bash\nexec %s -m pip "$@"\n' "/isaac-sim/python.sh" > /usr/local/bin/pip \
 && chmod +x /usr/local/bin/pip \
 && ln -sf /usr/local/bin/pip /usr/local/bin/pip3 \
 && /isaac-sim/python.sh -m pip install --upgrade pip setuptools wheel --no-cache-dir

# ======================================================================
# Copy your entire repo (incl. IsaacLab/, external/rsl_rl/, and your code)
# Make sure submodules are populated BEFORE build.
# ======================================================================
WORKDIR /workspace
COPY . /workspace
# make scripts executable (optional, convenient)
RUN find /workspace -maxdepth 3 -type f -name "*.sh" -exec chmod +x {} +

# ======================================================================
# IsaacLab: link to Isaac Sim root; install deps; install IsaacLab
# ======================================================================
# Ensure IsaacLab exists (it should be in your repo as a submodule)
RUN if [ ! -d "${ISAACLAB_PATH}" ]; then \
      echo "[Docker] ERROR: IsaacLab not found at ${ISAACLAB_PATH}. Did you clone submodules?"; \
      exit 2; \
    fi

# Link _isaac_sim into the IsaacLab tree (as in the official Dockerfile)
RUN ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim

# Install toml needed by tools
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install toml

# Install apt deps for IsaacLab extensions that declare them
RUN --mount=type=cache,target=/var/cache/apt \
    ${ISAACLAB_PATH}/tools/install_deps.py apt ${ISAACLAB_PATH}/source || true && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# Create dirs needed for Singularity/Apptainer binding (as in official file)
RUN mkdir -p ${ISAACSIM_ROOT_PATH}/kit/cache && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/ov && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/pip && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/nvidia/GLCache &&  \
    mkdir -p ${DOCKER_USER_HOME}/.nv/ComputeCache && \
    mkdir -p ${DOCKER_USER_HOME}/.nvidia-omniverse/logs && \
    mkdir -p ${DOCKER_USER_HOME}/.local/share/ov/data && \
    mkdir -p ${DOCKER_USER_HOME}/Documents

# NVIDIA binary placeholders (for Singularity)
RUN touch /bin/nvidia-smi && \
    touch /bin/nvidia-debugdump && \
    touch /bin/nvidia-persistenced && \
    touch /bin/nvidia-cuda-mps-control && \
    touch /bin/nvidia-cuda-mps-server && \
    touch /etc/localtime && \
    mkdir -p /var/run/nvidia-persistenced && \
    touch /var/run/nvidia-persistenced/socket

# Install IsaacLab (pip cache mounted to avoid re-downloading large wheels)
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh --install

# Remove quadprog if it slipped in as a dep (kept from official)
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip uninstall -y quadprog || true

# ======================================================================
# RSL-RL: install from PyPI
# ======================================================================
RUN /isaac-sim/python.sh -m pip install --no-cache-dir rsl-rl-lib==2.3.3

# ======================================================================
# Quality-of-life aliases (match official IsaacLab image behavior)
# ======================================================================
RUN echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> ${HOME}/.bashrc && \
    echo "alias isaaclab=${ISAACLAB_PATH}/isaaclab.sh" >> ${HOME}/.bashrc && \
    echo "alias python=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> ${HOME}/.bashrc && \
    echo "alias python3=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> ${HOME}/.bashrc && \
    echo "alias pip='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias pip3='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias tensorboard='${ISAACLAB_PATH}/_isaac_sim/python.sh ${ISAACLAB_PATH}/_isaac_sim/tensorboard'" >> ${HOME}/.bashrc && \
    echo "export TZ=$(date +%Z)" >> ${HOME}/.bashrc && \
    echo "shopt -s histappend" >> /root/.bashrc && \
    echo "PROMPT_COMMAND=\"history -a\"" >> /root/.bashrc

# ======================================================================
# Your project / extension (sslab_isaaclab): install last
# ======================================================================
RUN /isaac-sim/python.sh -m pip install -e source --no-cache-dir

RUN /isaac-sim/python.sh -m pip install --no-cache-dir "numpy<2" "opencv-python<4.9" "qpth"

RUN /isaac-sim/kit/python/bin/python3 -m pip install --no-cache-dir "setuptools<70" "wandb" \
    && /isaac-sim/kit/python/bin/python3 -c "import pkg_resources; print('pkg_resources OK')" \
    && /isaac-sim/kit/python/bin/python3 -c "import wandb; print('wandb OK')"

# Helpful PYTHONPATH (pip -e already handles imports, but this can help dev)
ENV PYTHONPATH="/workspace:/workspace/IsaacLab:${PYTHONPATH}"

# Work from your repo root so scripts like scripts/rsl_rl/train.py resolve
WORKDIR /workspace

# Default entrypoint; your SLURM job runs: python scripts/rsl_rl/train.py ...
ENTRYPOINT ["/bin/bash"]