# MIST Glioma builds on the MIST image, which already has MIST 3, PyTorch 2.9.1,
# cuDNN 9.15 or later (so BF16 inference runs at full speed), and a default
# username, so the image runs under a numeric UID (docker run -u, Kubernetes
# runAsUser). Keep MIST_IMAGE on a MIST release that MIST Glioma supports.
ARG MIST_IMAGE=mistmedical/mist:3.0.0
FROM ${MIST_IMAGE}

ENV HF_HOME=/opt/huggingface

# Install MIST Glioma from the checked-out source, so the image matches the commit
# being built. The image's torch is pinned as a constraint; without it, pip can
# replace torch while resolving HD-BET's dependencies.
#
# Resolving torch's dependencies also puts back the cuDNN that PyTorch 2.9 pins
# (9.10), which runs BF16 3D convolutions about 3x slower. So upgrade cuDNN again
# afterwards, as MIST's own image does. pip warns that torch pins 9.10; PyTorch
# works with the newer cuDNN.
COPY . /tmp/mist-glioma
RUN python -c "import torch; print('torch==' + torch.__version__.split('+')[0])" > /tmp/constraints.txt \
    && pip install --no-cache-dir --constraint /tmp/constraints.txt /tmp/mist-glioma \
    && pip install --no-cache-dir -U "nvidia-cudnn-cu12>=9.15" \
    && rm -rf /tmp/mist-glioma /tmp/constraints.txt

# Fail the build if installing MIST Glioma replaced the image's MIST 3 or its
# newer cuDNN.
RUN python -c "import torch; from importlib.metadata import version; m = version('mist-medical'); c = torch.backends.cudnn.version(); print('mist-medical', m, '| cuDNN', c); assert int(m.split('.')[0]) >= 3, m; assert c and c >= 91500, c"

# Include the model weights so the image can run without network access.
RUN python -c "from HD_BET.checkpoint_download import maybe_download_parameters; \
from mist_glioma.model_assets import get_model_directories; \
maybe_download_parameters(); get_model_directories()"

WORKDIR /app
