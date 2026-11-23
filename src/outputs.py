"""Deliver the LoRA: HF Hub repo, S3-compatible bucket, presigned PUT URL; optional base64."""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

import requests

from .secret_utils import redact

log = logging.getLogger(__name__)

BUCKET_ENV = ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY")
MAX_BASE64_BYTES = 80 * 2**20  # 80 MiB cap so job output stays reasonable

BASE_HF = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "wan-t2v-1.3b": "Wan-AI/Wan2.1-T2V-1.3B",
    "wan-t2v-14b": "Wan-AI/Wan2.1-T2V-14B",
    "wan-i2v-14b": "Wan-AI/Wan2.1-I2V-14B-720P",
    "wan22-t2v-a14b-low": "Wan-AI/Wan2.2-T2V-A14B",
    "wan22-t2v-a14b-high": "Wan-AI/Wan2.2-T2V-A14B",
    "wan22-i2v-a14b-low": "Wan-AI/Wan2.2-I2V-A14B",
    "wan22-i2v-a14b-high": "Wan-AI/Wan2.2-I2V-A14B",
    "flux-dev": "black-forest-labs/FLUX.1-dev",
    "ltx-video": "Lightricks/LTX-Video",
    "qwen-image": "Qwen/Qwen-Image",
    "hunyuan-video": "tencent/HunyuanVideo",
}


def bucket_configured() -> bool:
    return all(os.environ.get(k) for k in BUCKET_ENV)


def encode_file_base64(path: str, max_bytes: int = MAX_BASE64_BYTES) -> str | None:
    size = os.path.getsize(path)
    if size > max_bytes:
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def model_card(cfg, output_name: str, train: dict[str, Any]) -> str:
    w, h = cfg.resolution_wh
    base = BASE_HF.get(cfg.base_model, cfg.base_model)
    tags = ["lora", "diffusion-pipe", "runpod"]
    if cfg.base_model == "sdxl":
        tags += ["stable-diffusion-xl", "text-to-image"]
    elif cfg.base_model == "flux-dev":
        tags += ["flux", "text-to-image"]
    elif "wan" in cfg.base_model:
        tags += ["wan", "text-to-video" if not cfg.is_i2v else "image-to-video"]
    lines = [
        "---",
        f"base_model: {base}",
        "tags:",
        *[f"- {t}" for t in tags],
    ]
    if cfg.trigger_word:
        lines.append(f"instance_prompt: {cfg.trigger_word}")
    lines += [
        "---", "", f"# {output_name}", "",
        f"LoRA for **{base}** (`{cfg.base_model}`) trained with "
        "[tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) "
        "(`c114c67eef87ef3a16eb82e4bb24c8ef700c26e4`) on Runpod Serverless using "
        "[prilojject/runpod-diffusion-pipe-trainer](https://github.com/prilojject/runpod-diffusion-pipe-trainer).",
        "",
    ]
    if cfg.trigger_word:
        lines += [f"Trigger word: `{cfg.trigger_word}`", ""]
    lines += [
        f"- steps: {cfg.steps}",
        f"- rank: {cfg.rank}",
        f"- learning rate: {cfg.learning_rate:g}",
        f"- resolution: {w}x{h}",
        f"- dtype: {cfg.normalized_dtype}",
        f"- pipeline_stages: {cfg.pipeline_stages}",
        f"- LoRA format: {cfg.lora_format}",
        f"- final loss: {train.get('final_loss')}",
        "",
        "This worker and diffusion-pipe are licensed under GPL-3.0.",
        "",
    ]
    return "\n".join(lines)


def push_to_hf(cfg, token: str, lora_path: str, extras: list[str], card: str) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=cfg.hf_repo_id, repo_type="model", private=cfg.hf_private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=os.path.basename(lora_path), path_or_fileobj=lora_path),
           CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode())]
    for s in extras:
        ops.append(CommitOperationAdd(path_in_repo=os.path.basename(s), path_or_fileobj=s))
    api.create_commit(repo_id=cfg.hf_repo_id, repo_type="model", operations=ops,
                      commit_message=f"LoRA {os.path.basename(lora_path)} ({cfg.base_model}, {cfg.steps} steps)")
    return f"https://huggingface.co/{cfg.hf_repo_id}"


def upload_bucket(path: str, prefix: str) -> str:
    from runpod.serverless.utils.rp_upload import upload_file_to_bucket
    return upload_file_to_bucket(file_name=os.path.basename(path), file_location=path,
                                 bucket_name=os.environ.get("BUCKET_NAME") or None, prefix=prefix)


def put_presigned(url: str, path: str) -> int:
    with open(path, "rb") as f:
        r = requests.put(url, data=f, headers={"Content-Type": "application/octet-stream"}, timeout=(15, 900))
    if r.status_code >= 300:
        raise RuntimeError(f"presigned PUT returned HTTP {r.status_code}: {r.text[:200]}")
    return r.status_code


def deliver(cfg, run_name: str, output_name: str, lora_path: str, extras: list[str],
            train: dict[str, Any], token: str | None, persistent: bool) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {
        "lora_file": os.path.basename(lora_path),
        "lora_path": lora_path,
        "lora_size_mb": round(os.path.getsize(lora_path) / 2**20, 2),
        "lora_format": cfg.lora_format,
        "persistent_storage": persistent,
    }
    if persistent:
        out["volume_path"] = lora_path
    warnings: list[str] = []
    delivered = persistent
    card = model_card(cfg, output_name, train)
    with open(os.path.join(os.path.dirname(lora_path), "README.md"), "w", encoding="utf-8") as f:
        f.write(card)

    if cfg.hf_repo_id and token:
        try:
            out["hf_repo_url"] = push_to_hf(cfg, token, lora_path, extras, card)
            delivered = True
        except Exception as exc:
            out["hf_error"] = redact(str(exc)).splitlines()[0][:400]
            log.error("HF upload failed: %s", out["hf_error"])
    if cfg.upload_url:
        try:
            out["upload_url_status"] = put_presigned(cfg.upload_url, lora_path)
            delivered = True
        except Exception as exc:
            out["upload_url_error"] = redact(str(exc))[:400].replace(cfg.upload_url, "<presigned url>")
    if cfg.upload_to_bucket and bucket_configured():
        try:
            prefix = f"diffusion-pipe/{run_name}"
            out["bucket_lora_url"] = upload_bucket(lora_path, prefix)
            for s in extras:
                out.setdefault("bucket_extra_urls", []).append(upload_bucket(s, prefix))
            delivered = True
        except Exception as exc:
            out["bucket_error"] = redact(str(exc))[:400]
    if not delivered:
        warnings.append(
            "the LoRA was not delivered anywhere persistent: attach a network volume, or set "
            "hf_repo_id (+HF_TOKEN), upload_url, or BUCKET_* env vars. It is lost when the worker stops. "
            "Set return_lora_base64=true only if you need the file in the job payload."
        )
    return out, warnings
