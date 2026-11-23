# Diffusion-Pipe LoRA Trainer (SDXL, Wan, FLUX) — Runpod Serverless worker

Train a **LoRA** for **SDXL**, **Wan 2.1 / 2.2**, or **FLUX.1-dev** on a **Runpod Serverless** GPU
with [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) (pinned
`c114c67eef87ef3a16eb82e4bb24c8ef700c26e4`) and **Deepspeed pipeline parallelism**.
You send a URL to a zip of your images and/or videos (with optional `.txt` captions) and a trigger word;
you get back a **`.safetensors` LoRA** on your Hugging Face repo, S3 bucket, presigned URL or network volume.
No Pod, no UI, pay per second of training.

This is **not** a duplicate of the other trainers in this series:

- [runpod-image-lora-trainer](https://github.com/prilojject/runpod-image-lora-trainer) — kohya **sd-scripts**, SDXL / FLUX only, single GPU.
- [runpod-video-lora-trainer](https://github.com/prilojject/runpod-video-lora-trainer) — kohya **musubi-tuner**, Wan video LoRAs only.
- **This worker** — **diffusion-pipe**, one API for images *and* video, many architectures, optional **multi-GPU pipeline parallelism**.

Useful search terms: *train SDXL LoRA on Runpod serverless*, *diffusion-pipe multi-GPU LoRA*,
*Wan 2.1 LoRA Deepspeed*, *FLUX.1-dev LoRA trainer Runpod Hub*, *pipeline parallel LoRA training*.

<!-- The badge and Hub URL go live after Runpod reviews the listing. -->
[![Runpod](https://api.runpod.io/badge/prilojject/runpod-diffusion-pipe-trainer)](https://console.runpod.io/hub/prilojject/runpod-diffusion-pipe-trainer)

**Deploy on Runpod Hub:** [console.runpod.io/hub/prilojject/runpod-diffusion-pipe-trainer](https://console.runpod.io/hub/prilojject/runpod-diffusion-pipe-trainer)

> **Listing status:** submitted — the live Hub URL works after Runpod's review.

Source: [github.com/prilojject/runpod-diffusion-pipe-trainer](https://github.com/prilojject/runpod-diffusion-pipe-trainer) (**GPL-3.0**, same as upstream diffusion-pipe).
For agents: [llms.txt](./llms.txt) · [AGENTS.md](./AGENTS.md) · [openapi.yaml](./openapi.yaml) · [CITATION.cff](./CITATION.cff)

---

## What you send / what you get

**You send** (job `input`; defaults can come from the Hub deploy form):

| Field | Example | Notes |
|---|---|---|
| `base_model` | `sdxl` **(default)** | Also `wan-t2v-1.3b`, `wan-t2v-14b`, `wan-i2v-14b`, `wan22-t2v-a14b-low` / `-high`, `flux-dev`. |
| `dataset_url` | `https://.../set.zip` | `.zip` or `.tar(.gz)` of images (`.jpg/.png/.webp`) and/or videos (`.mp4/.webm/.mov/.mkv`), optional same-name `.txt` captions. WebP *videos* are not supported. |
| `trigger_word` | `zxowl` | Rare token that activates the LoRA. Prepended to every caption. Empty captions get `trigger_word` + `class_word` (never left blank). |
| `class_word` | `figurine` | Optional generic class used in trigger-only captions. |
| `steps`, `rank`, `learning_rate` | `2000`, `16`, `2e-5` | diffusion-pipe defaults (not musubi's `2e-4`). |
| `resolution` | `1024` or `832x480` | A single number is square. Multiple of 16. Smoke test uses `512`. |
| `frame_buckets` / `frames` | `[1, 17]` | Video lengths. Image-only jobs use `[1]`. |
| `pipeline_stages` | `1` (default) | Deepspeed pipeline parallelism. GPU count must be divisible by this. Incompatible with `blocks_to_swap`. |
| `blocks_to_swap` | `auto` | `auto` / `0` / an integer. LoRA only, and only when `pipeline_stages=1`. |
| `caption_mode` | `auto` / `trigger` / `existing` | `auto` uses `.txt` when present, otherwise the trigger word. |
| `hf_repo_id` | `your-name/my-lora` | Upload LoRA + model card (needs `HF_TOKEN` with write access). |
| `upload_url` | presigned `PUT` URL | The `.safetensors` is `PUT` there. |
| `dry_run` | `true` | Validate the dataset and check that model catalogs are reachable. No GPU work. |

Tokens (`HF_TOKEN`, bucket keys) belong in **Runpod secrets / env**, never in the job input.

**You get** (job `output`, trimmed):

```json
{
  "status": "completed",
  "gpu": {"gpu": "NVIDIA GeForce RTX 4090", "vram_gb": 23.5},
  "training": {"global_step": 20, "final_loss": 0.12, "train_loop_seconds": 90.0},
  "lora_format": "kohya",
  "artifacts": {
    "lora_file": "zxowl_sdxl_lora.safetensors",
    "lora_size_mb": 25.4,
    "hf_repo_url": "https://huggingface.co/your-name/zxowl-sdxl-lora",
    "volume_path": "/runpod-volume/diffusion-pipe/runs/<run_name>/..."
  },
  "dataset": {"images_used": 3, "videos_used": 0, "captions_from_files": 3}
}
```

LoRA format depends on the model: **kohya** (SDXL, works in ComfyUI / A1111), **ComfyUI** (Wan, LTX, Qwen), **Diffusers** (FLUX, HunyuanVideo).

> **Deliver the LoRA somewhere.** Serverless workers are ephemeral. Set at least one of: a network volume
> (files land in `/runpod-volume/diffusion-pipe/runs/<run_name>/`), `hf_repo_id` + `HF_TOKEN`, `upload_url`, or the
> `BUCKET_*` env vars. Without them the response carries a warning (and the LoRA only if you set `return_lora_base64`).

Always use async `POST /run` and raise `policy.executionTimeout` — the 600 s default is too short (a cold start may download ~6.5 GiB for SDXL, much more for 14B / Flux).

---

## Supported models (v1)

| `base_model` | `[model].type` | Weights | LoRA format | VRAM |
|---|---|---|---|---|
| `sdxl` **(default)** | `sdxl` | `stabilityai/stable-diffusion-xl-base-1.0` (`sd_xl_base_1.0_0.9vae.safetensors`, ≈ 6.46 GiB), no token, `dtype=bfloat16` | kohya | 24 GB |
| `wan-t2v-1.3b` | `wan` | Official `Wan-AI/Wan2.1-T2V-1.3B`, Apache-2.0, no token, `dtype=bfloat16` | ComfyUI | 24 GB |
| `wan-t2v-14b` | `wan` | `Wan-AI/Wan2.1-T2V-14B` with the min-VRAM recipe (`transformer_dtype=float8`, `blocks_to_swap=32`, `AdamW8bitKahan`) | ComfyUI | 24 GB tight / 48 GB better |
| `wan-i2v-14b` | `wan` | `Wan-AI/Wan2.1-I2V-14B-720P`. **Video-only** dataset (first frame = condition). | ComfyUI | 48 GB recommended |
| `wan22-t2v-a14b-low` / `-high` | `wan` | `Wan-AI/Wan2.2-T2V-A14B`. Experts are trained **separately** (not both in one job). Timestep range 0–0.875 (low) or 0.875–1 (high). | ComfyUI | 48 GB recommended |
| `wan22-i2v-a14b-low` / `-high` | `wan` | Same idea; I2V boundary is 0.9. Video-only dataset. | ComfyUI | 48 GB recommended |
| `flux-dev` | `flux` | Diffusers `black-forest-labs/FLUX.1-dev` (**gated**, needs `HF_TOKEN`), `transformer_dtype=float8`, `flux_shift=true` | Diffusers | 24–48 GB |
| `ltx-video` | `ltx-video` | `Lightricks/LTX-Video` | ComfyUI | 24–48 GB |
| `qwen-image` | `qwen_image` | `Qwen/Qwen-Image` with the 24 GB recipe (`blocks_to_swap=8`) | ComfyUI | 24 GB |

Official default paths use **bfloat16** weights. The training dtype is kept in sync with the files (`dtype`, `adapter.dtype`, `save_dtype`). If you point at Comfy **fp16** safetensors, set `dtype=float16` — do not force bf16. v1 default paths never download Comfy `fp8_scaled` files; `transformer_dtype=float8` is a runtime quant for LoRA.

**Not in v1** (upstream diffusion-pipe supports them; this worker returns a clear error): Flux 2, Flux Kontext, Chroma, HiDream, SD3, Cosmos*, Lumina, OmniGen2, AuraFlow, Z-Image, HunyuanVideo-1.5, HunyuanImage-2.1, Anima, Ernie-Image, LTX 2.3, Ideogram4, Krea 2, MiniMax*, MOSS, Qwen-Image-Edit / 2.1, full fine-tune, and HunyuanVideo's ComfyUI file set (stub in v1).

---

## Multi-GPU

Default and the Hub test use **1 GPU**, `pipeline_stages=1`.

To train a model that does not fit on one 24 GB card:

1. Deploy the endpoint with `gpuCount=2` (or more).
2. Set `pipeline_stages` to 2 (GPU count must be divisible by this). With 4 GPUs and `pipeline_stages=2` you get two data-parallel replicas, each split across 2 GPUs.
3. Leave `blocks_to_swap` at 0 — block swap only works with `pipeline_stages=1`.

On **RTX 4090 / 40-series** the worker sets `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` (required by Deepspeed on those cards).

---

## FAQ

**Why is the default SDXL, not Wan?**
SDXL is public, ~6.5 GiB, and a cheap 24 GB smoke test. Use `wan-t2v-1.3b` when you want a video LoRA without a token. For Wan-only jobs with musubi-tuner, see [runpod-video-lora-trainer](https://github.com/prilojject/runpod-video-lora-trainer).

**Do I need a Hugging Face token?**
Not for SDXL or official Wan-AI catalogs. **Yes** for `flux-dev` (accept the license, then set `HF_TOKEN` as a Runpod secret). Also needed to upload with `hf_repo_id`.

**Where are the weights cached?**
On the attached network volume under `/runpod-volume/diffusion-pipe/models/`. They are **not** baked into the image. Without a volume they go to `/tmp` and are re-downloaded on every cold start.

**What about sample images after training?**
v1 does not generate samples (`num_samples` is accepted and ignored). Keep the Hub test cheap.

**Can I change the learning rate / rank / steps?**
Yes. Job input wins over the Hub form, which wins over the preset.

---

## Quick start

```bash
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"dry_run": true, "base_model": "sdxl",
                 "dataset_url": "https://raw.githubusercontent.com/prilojject/runpod-diffusion-pipe-trainer/main/examples/datasets/tiny_sdxl.zip",
                 "trigger_word": "zxowl", "class_word": "figurine"}}'
```

Hub test: SDXL, 20 steps, 512 px, no samples, RTX 4090. 14B / Flux / multi-GPU are not part of the official test.

Image: `ghcr.io/prilojject/runpod-diffusion-pipe-trainer:v1.0.0` (make the package **public** in the GitHub Packages UI after the first push).

License: [GPL-3.0](./LICENSE). See [NOTICE](./NOTICE) for upstream attribution.
