# Multi-model LoRA trainer for Runpod Serverless, built on tdrussell/diffusion-pipe.
# No model weights are baked in: checkpoints are downloaded at job start and cached on
# the network volume (/runpod-volume/diffusion-pipe/models) when one is attached.
#
# Upstream has no release tags; pin an explicit SHA (never a floating main). This SHA is
# only the Docker pin — our GitHub release for the Hub must use targetCommitish=main.
ARG BASE_IMAGE=pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG DIFFUSION_PIPE_SHA=c114c67eef87ef3a16eb82e4bb24c8ef700c26e4

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    DIFFUSION_PIPE_DIR=/opt/diffusion-pipe

# git: clone diffusion-pipe + submodules; ffmpeg: video decode; libgl/libglib: opencv;
# build-essential + ninja: Deepspeed / optional flash-attn JIT.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git ffmpeg ca-certificates libgl1 libglib2.0-0 build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Clone the repo, check out the pinned SHA, then init submodules (ComfyUI, HunyuanVideo, …).
# --depth 1 on submodules keeps the image smaller. Do not clone a floating main.
RUN git clone https://github.com/tdrussell/diffusion-pipe.git ${DIFFUSION_PIPE_DIR} \
    && cd ${DIFFUSION_PIPE_DIR} \
    && git checkout ${DIFFUSION_PIPE_SHA} \
    && git rev-parse HEAD > /opt/diffusion-pipe.commit \
    && test "$(cat /opt/diffusion-pipe.commit)" = "${DIFFUSION_PIPE_SHA}" \
    && git submodule update --init --recursive --depth 1 \
    && rm -rf .git submodules/*/.git

# Keep the base image's torch stack frozen so we do not upgrade CUDA/torch at install time.
WORKDIR ${DIFFUSION_PIPE_DIR}
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+)==' > /tmp/constraints.txt \
    && cat /tmp/constraints.txt \
    && pip install -r requirements.txt -c /tmp/constraints.txt

# flash-attn is optional: some models need it. Prefer a wheel; if the build fails, the
# image still ships and those models return a clear runtime error.
RUN pip install flash-attn --no-build-isolation -c /tmp/constraints.txt \
    || echo "WARN: flash-attn could not be installed; models that require it will error at runtime"

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt -c /tmp/constraints.txt \
    && pip freeze > /app/requirements.lock \
    && python -c "import torch, deepspeed, runpod, diffusers, transformers, peft; \
print('torch', torch.__version__, '| cuda', torch.version.cuda, \
'| python', __import__('sys').version.split()[0], \
'| deepspeed', deepspeed.__version__, '| runpod', runpod.__version__, \
'| pipe sha', open('/opt/diffusion-pipe.commit').read().strip())"

COPY handler.py /app/handler.py
COPY src /app/src

# No secrets are baked in: HF_TOKEN / BUCKET_* come from Runpod secrets at runtime.
CMD ["python", "-u", "/app/handler.py"]
