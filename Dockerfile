# The base image and torch version are pinned together. Without the constraint,
# pip can replace the image's torch while resolving HD-BET's dependencies.
ARG PYTORCH_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
ARG TORCH_VERSION=2.9.1
FROM ${PYTORCH_IMAGE}

# Set environment variables for non-interactive installation.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/opt/huggingface

# Install MIST Glioma from the checked-out source, so the image matches the commit
# being built.
ARG TORCH_VERSION
COPY . /tmp/mist-glioma
RUN echo "torch==${TORCH_VERSION}" > /tmp/constraints.txt \
    && pip install --no-cache-dir --constraint /tmp/constraints.txt /tmp/mist-glioma \
    && rm -rf /tmp/mist-glioma /tmp/constraints.txt

# Include the model weights so the image can run without network access.
RUN python -c "from HD_BET.checkpoint_download import maybe_download_parameters; \
from mist_glioma.model_assets import get_model_directories; \
maybe_download_parameters(); get_model_directories()"

# Create app directory.
RUN mkdir /app
WORKDIR /app
