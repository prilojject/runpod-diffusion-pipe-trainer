# AGENTS.md — runpod-diffusion-pipe-trainer

This is a Runpod Serverless worker that trains LoRAs with [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) (pinned SHA `c114c67eef87ef3a16eb82e4bb24c8ef700c26e4`, no upstream tags). License: **GPL-3.0**.

## What this worker does

- Accepts a job `input` (or Hub form env) with `dataset_url`, `trigger_word`, and optional `base_model`.
- Default model is **SDXL** (public, ~6.5 GiB, 24 GB). Also Wan 2.1 / 2.2 and FLUX.1-dev.
- Writes two TOML files (train + dataset) and launches training. One GPU: `python -u train.py --deepspeed --config train.toml` (avoids a Deepspeed-launcher hang on serverless). Two or more GPUs: `deepspeed --num_gpus=N --master_addr=127.0.0.1 --master_port=29500 train.py --deepspeed --config train.toml`. Always sets `NCCL_SHM_DISABLE=1`.
- Delivers the `.safetensors` LoRA to a network volume, Hugging Face, a presigned PUT URL, or `BUCKET_*`.

## Do / don't

- Do **not** put `HF_TOKEN` or bucket keys in the job input. Use Runpod secrets.
- Do **not** bake model weights into the image. Cache them on `/runpod-volume/diffusion-pipe/models/`.
- Training **dtype must match the weight files**. Official default paths are **bfloat16**. Do not copy `fp16` from the musubi video worker unless the files are actually fp16.
- Do **not** download Comfy `fp8_scaled` files for v1 default paths.
- On RTX 4090 / 40-series set `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`.
- `blocks_to_swap` is incompatible with `pipeline_stages > 1`.
- Wan 2.2 experts are trained **separately** (`-low` / `-high`), not both in one job.
- I2V models need a **video-only** dataset.
- Hub / smoke tests: **SDXL, 20 steps, 512 px, no samples, 1× RTX 4090**. Do not run 14B / Flux / multi-GPU on the paid Hub test.
- Our GitHub release for the Hub must use `targetCommitish=main`, **not** a SHA (Hub clones with `git clone -b <target>`).

## Layout

- `handler.py` — Runpod entry point (`refresh_worker: true`).
- `src/config.py` — input schema, presets, validation.
- `src/dataset.py` — zip-slip-safe unpack, captions.
- `src/models.py` — HF downloads (SDXL file, official catalogs).
- `src/train.py` — TOML + Deepspeed command + output collection.
- `src/outputs.py` — HF / S3 / presigned delivery.
- `.runpod/hub.json`, `.runpod/tests.json` — Hub listing and official test.
- `Dockerfile` — pins `DIFFUSION_PIPE_SHA`, inits submodules, does not clone floating `main`.

## Multi-GPU

Default `gpuCount=1`, `pipeline_stages=1`. To use two GPUs, deploy with `gpuCount=2` and `pipeline_stages=2`.
