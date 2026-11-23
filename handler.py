"""Runpod Serverless entry point: multi-model LoRA training with tdrussell/diffusion-pipe.

Local dry run (no GPU, no model weights):
    python handler.py --test_input '{"input": {"dry_run": true, "dataset_url": "https://.../data.zip"}}'
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import __version__  # noqa: E402
from src.config import ConfigError, resolve_config  # noqa: E402
from src.dataset import DatasetError, prepare_dataset  # noqa: E402
from src.paths import configure_caches, has_volume, output_root  # noqa: E402
from src.secret_utils import RedactFilter, get_hf_token, redact  # noqa: E402

configure_caches()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for h in logging.getLogger().handlers:
    h.addFilter(RedactFilter())
log = logging.getLogger("diffusion-pipe-worker")


def _progress_fn(job: dict):
    def send(payload: dict) -> None:
        log.info("progress: %s", payload)
        if job.get("id") and not str(job["id"]).startswith("local") and os.environ.get("RUNPOD_POD_ID"):
            try:
                import runpod
                runpod.serverless.progress_update(job, payload)
            except Exception as exc:  # best effort
                log.debug("progress_update failed: %s", exc)
    return send


def handler(job: dict) -> dict[str, Any]:
    result = _handle(job)
    # Always start the next job on a fresh worker (frees GPU memory, drops partial downloads).
    result["refresh_worker"] = True
    return result


def _handle(job: dict) -> dict[str, Any]:
    started = time.time()
    try:
        cfg = resolve_config(job.get("input") or {})
    except ConfigError as exc:
        return {"error": f"invalid input: {exc}"}

    run_name = cfg.run_name or str(job.get("id") or f"run-{int(started)}")
    output_name = cfg.output_name or "_".join(
        x for x in ((cfg.trigger_word or "lora").replace(" ", "_"), cfg.base_model.replace("-", "_"), "lora") if x)
    token = get_hf_token()
    progress = _progress_fn(job)
    persistent = has_volume()
    run_dir = os.path.join(output_root(), run_name)
    work = os.path.join("/tmp/diffusion-pipe/work", run_name)
    media_dir = os.path.join(work, "media")
    videos_dir = os.path.join(media_dir, "videos")
    images_dir = os.path.join(media_dir, "images")
    shutil.rmtree(work, ignore_errors=True)

    result: dict[str, Any] = {
        "worker_version": __version__,
        "run_name": run_name,
        "base_model": cfg.base_model,
        "lora_format": cfg.lora_format,
        "config": cfg.public_dict(),
        "warnings": list(cfg.warnings),
    }
    if not persistent:
        result["warnings"].append(
            "no network volume: base models are re-downloaded on every cold start "
            "(attach one to cache them under /runpod-volume/diffusion-pipe/models)"
        )

    try:
        progress({"stage": "preparing_dataset"})
        report = prepare_dataset(cfg, work, videos_dir, images_dir)
    except DatasetError as exc:
        return {"error": f"dataset: {exc}", **{k: result[k] for k in ("run_name", "config")}}
    except Exception as exc:
        log.error("dataset failed: %s", redact(traceback.format_exc()))
        return {"error": f"dataset failed: {redact(str(exc)).splitlines()[0][:400]}"}
    result["dataset"] = report.as_dict()
    result["warnings"] += report.warnings

    if cfg.dry_run:
        from src.models import check_access
        access = check_access(cfg, token)
        result.update({
            "dry_run": True,
            "status": "ok" if all(v.get("ok") for v in access.values()) else "error",
            "model_access": access,
            "model_download_gb": round(sum(v.get("size_gb") or 0 for v in access.values() if v.get("ok")), 2),
            "elapsed_seconds": round(time.time() - started, 2),
        })
        if result["status"] == "error":
            result["error"] = "base model files not reachable: " + "; ".join(
                f"{k}: {v.get('error')}" for k, v in access.items() if not v.get("ok"))
        shutil.rmtree(work, ignore_errors=True)
        return result

    from src.models import ModelError, ensure_models
    from src.train import (TrainingError, auto_blocks_to_swap, build_train_command,
                           cleanup_deepspeed_checkpoints, copy_final_lora, fill_loss_from_tensorboard,
                           find_outputs, gpu_info, run_command, write_dataset_toml, write_train_toml)

    gpu = gpu_info()
    result["gpu"] = gpu
    if not gpu["gpu"]:
        return {"error": "no CUDA GPU visible to the worker", **{k: result[k] for k in ("run_name", "config")}}
    if gpu.get("supported") is False:
        return {"error": f"GPU {gpu['gpu']} (compute {gpu.get('compute_capability')}) is not supported by this "
                         "image's PyTorch/CUDA build (Blackwell MIG GPUs are excluded in hub.json); "
                         "redeploy on Ampere/Ada (RTX 4090, L40S, A6000 Ada, A6000, A40)",
                **{k: result[k] for k in ("run_name", "config")}, "gpu": gpu}
    gpu_count = int(gpu.get("gpu_count") or 1)
    if gpu_count % cfg.pipeline_stages != 0:
        return {"error": f"visible GPU count {gpu_count} is not divisible by pipeline_stages={cfg.pipeline_stages}",
                "run_name": run_name, "gpu": gpu}
    if cfg.is_14b or cfg.base_model == "flux-dev":
        if gpu["vram_gb"] < 40:
            result["warnings"].append(
                f"{cfg.base_model} on {gpu['vram_gb']} GB VRAM will use block swap and/or float8; "
                "a 48 GB GPU is recommended"
            )

    try:
        models, minfo = ensure_models(cfg, token, progress)
    except ModelError as exc:
        return {"error": str(exc), **{k: result[k] for k in ("run_name", "config")}}
    result["models"] = minfo

    os.makedirs(run_dir, exist_ok=True)
    media_dirs = []
    if report.images_used:
        media_dirs.append(images_dir)
    if report.videos_used:
        media_dirs.append(videos_dir)
    dataset_toml = os.path.join(work, "dataset.toml")
    train_toml = os.path.join(work, "train.toml")
    write_dataset_toml(cfg, media_dirs, dataset_toml, report.images_used + report.videos_used)
    blocks = auto_blocks_to_swap(cfg, gpu["vram_gb"])
    result["blocks_to_swap"] = blocks
    write_train_toml(cfg, models, dataset_toml, run_dir, train_toml, blocks)
    # Keep a copy of the generated configs next to the outputs (no secrets in them).
    shutil.copy2(dataset_toml, os.path.join(run_dir, "dataset.toml"))
    shutil.copy2(train_toml, os.path.join(run_dir, "train.toml"))

    log_path = os.path.join(run_dir, "train.log")
    timeout_s = int(os.environ.get("TRAIN_TIMEOUT_SECONDS", "0")) or None
    try:
        cmd, extra_env = build_train_command(cfg, train_toml, gpu_count, gpu.get("gpu"))
        result["nccl_p2p_disabled"] = extra_env.get("NCCL_P2P_DISABLE") == "1"
        progress({"stage": "training", "step": 0, "total_steps": cfg.steps})
        train = run_command(cmd, log_path, progress, "training", total_steps=cfg.steps,
                            timeout_s=timeout_s, extra_env=extra_env)
    except TrainingError as exc:
        outs = find_outputs(run_dir)
        return {"error": redact(str(exc))[:4000], "run_name": run_name, "gpu": gpu,
                "checkpoints": [os.path.basename(os.path.dirname(c)) for c in outs["checkpoints"]],
                "log_path": log_path}
    except Exception as exc:
        log.error("training crashed: %s", redact(traceback.format_exc()))
        return {"error": f"training crashed: {redact(str(exc)).splitlines()[0][:500]}", "run_name": run_name}

    fill_loss_from_tensorboard(train, run_dir)
    result["training"] = train

    outs = find_outputs(run_dir)
    if not outs["lora_path"]:
        return {"error": "training finished but no LoRA file was written", "run_name": run_name,
                "training": train, "log_path": log_path}

    lora_path = copy_final_lora(outs["lora_path"], run_dir, output_name)
    cleanup_deepspeed_checkpoints(run_dir)

    from src.outputs import deliver, encode_file_base64
    progress({"stage": "uploading"})
    artifacts, warns = deliver(cfg, run_name, output_name, lora_path, [], train, token, persistent)
    result["warnings"] += warns
    artifacts["checkpoints"] = [os.path.basename(os.path.dirname(c)) for c in outs["checkpoints"]]
    if cfg.return_lora_base64:
        b64 = encode_file_base64(lora_path)
        if b64:
            artifacts["lora_base64"] = b64
        else:
            result["warnings"].append("return_lora_base64 skipped: LoRA is larger than 80 MiB")
    result["artifacts"] = artifacts
    result.update({"status": "completed", "elapsed_seconds": round(time.time() - started, 1)})
    shutil.rmtree(work, ignore_errors=True)
    return result


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
