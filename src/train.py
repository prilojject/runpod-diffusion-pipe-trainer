"""Build diffusion-pipe train + dataset TOML, launch Deepspeed, collect LoRA outputs."""

from __future__ import annotations

import glob
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

from .config import ResolvedConfig
from .models import SNAPSHOTS
from .secret_utils import redact

log = logging.getLogger(__name__)

PIPE_DIR = os.environ.get("DIFFUSION_PIPE_DIR", "/opt/diffusion-pipe")
# Deepspeed / tqdm / our own prints
STEP_RE = re.compile(
    r"(?:step[s]?\s*[=:]\s*|global_step\s*[=:]\s*|\|\s*)(\d+)(?:/(\d+))?",
    re.I,
)
LOSS_RE = re.compile(r"(?:train/)?loss\s*[=:]\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", re.I)


class TrainingError(RuntimeError):
    pass


def gpu_info() -> dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            major, minor = torch.cuda.get_device_capability(0)
            archs = torch.cuda.get_arch_list()
            sm = f"sm_{major}{minor}"
            supported = sm in archs or any(
                a.startswith(f"sm_{major}") and a[3:].isdigit() and int(a[3:]) <= major * 10 + minor
                for a in archs
            )
            count = torch.cuda.device_count()
            return {"gpu": p.name, "vram_gb": round(p.total_memory / 2**30, 1),
                    "compute_capability": f"{major}.{minor}", "supported": bool(supported),
                    "gpu_count": count}
    except Exception:  # pragma: no cover
        pass
    return {"gpu": None, "vram_gb": 0.0, "gpu_count": 0}


def is_rtx_40xx(gpu_name: str | None) -> bool:
    if not gpu_name:
        return False
    n = gpu_name.upper()
    return "RTX 40" in n or "GEFORCE RTX 40" in n or "4090" in n or "4080" in n or "4070" in n or "4060" in n


def toml_escape(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def auto_blocks_to_swap(cfg: ResolvedConfig, vram_gb: float) -> int:
    """Pick blocks_to_swap. 0 means disabled. Always 0 when pipeline_stages > 1."""
    if cfg.pipeline_stages > 1:
        return 0
    if cfg.blocks_to_swap != "auto":
        return int(cfg.blocks_to_swap)
    bm = cfg.base_model
    if bm in ("wan-t2v-14b", "wan-i2v-14b") or bm.startswith("wan22-"):
        return 32
    if bm == "qwen-image":
        return 8
    if bm in ("flux-dev", "hunyuan-video", "ltx-video") and vram_gb and vram_gb < 28:
        return 8
    return 0


def activation_checkpointing(cfg: ResolvedConfig) -> str | bool:
    if cfg.is_14b or cfg.base_model == "qwen-image":
        return "unsloth"
    return True


def optimizer_type(cfg: ResolvedConfig) -> str:
    if cfg.is_14b or cfg.base_model == "qwen-image":
        return "AdamW8bitKahan"
    return "adamw_optimi"


def write_dataset_toml(cfg: ResolvedConfig, media_dirs: list[str], path: str,
                       n_media: int) -> None:
    """Write a diffusion-pipe dataset.toml. Cache lives in cache/ inside each directory."""
    w, h = cfg.resolution_wh
    n = max(1, n_media)
    repeats = max(1, min(50, 100 // n))
    frames = cfg.frame_buckets
    # Image-only jobs should not request long video buckets.
    if not cfg.is_video_model and frames == [1, 17]:
        frames = [1]
    lines = [
        f"resolutions = [[{w}, {h}]]" if w != h else f"resolutions = [{w}]",
        "enable_ar_bucket = true",
        "min_ar = 0.5",
        "max_ar = 2.0",
        "num_ar_buckets = 7",
        f"frame_buckets = [{', '.join(str(x) for x in frames)}]",
        "skip_empty_caption = true",
        "",
    ]
    if cfg.is_i2v:
        lines += ["video_clip_mode = 'single_beginning'", ""]
    for d in media_dirs:
        lines += [
            "[[directory]]",
            f"path = {toml_escape(d)}",
            f"num_repeats = {repeats}",
            "",
        ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def write_train_toml(cfg: ResolvedConfig, models: dict[str, str], dataset_toml: str,
                     output_dir: str, path: str, blocks: int) -> None:
    """Write the diffusion-pipe training TOML. Dtype of model/adapter/save always match."""
    dtype = cfg.normalized_dtype
    warmup = 0 if cfg.steps <= 30 else min(100, max(5, cfg.steps // 20))
    save_every = cfg.save_every_n_steps or cfg.steps
    act = activation_checkpointing(cfg)
    act_toml = "'unsloth'" if act == "unsloth" else "true"
    opt = optimizer_type(cfg)
    spec = SNAPSHOTS[cfg.base_model]

    lines = [
        f"output_dir = {toml_escape(output_dir)}",
        f"dataset = {toml_escape(dataset_toml)}",
        "",
        "epochs = 1000",
        f"max_steps = {cfg.steps}",
        f"micro_batch_size_per_gpu = {cfg.batch_size}",
        f"pipeline_stages = {cfg.pipeline_stages}",
        "gradient_accumulation_steps = 1",
        "gradient_clipping = 1.0",
        f"warmup_steps = {warmup}",
        "eval_before_first_step = false",
        f"save_every_n_steps = {save_every}",
        f"activation_checkpointing = {act_toml}",
        "partition_method = 'parameters'",
        f"save_dtype = '{dtype}'",
        "caching_batch_size = 1",
        "steps_per_print = 1",
        "logging_steps = 1",
        "skip_final_checkpoint = true",
        "",
        "[model]",
        f"type = '{cfg.pipe_type}'",
        f"dtype = '{dtype}'",
    ]
    # Path keys from the downloader.
    if "checkpoint_path" in models:
        lines.append(f"checkpoint_path = {toml_escape(models['checkpoint_path'])}")
    if "ckpt_path" in models:
        lines.append(f"ckpt_path = {toml_escape(models['ckpt_path'])}")
    if "diffusers_path" in models:
        lines.append(f"diffusers_path = {toml_escape(models['diffusers_path'])}")
    if "transformer_path" in models:
        lines.append(f"transformer_path = {toml_escape(models['transformer_path'])}")
    if "single_file_path" in models:
        lines.append(f"single_file_path = {toml_escape(models['single_file_path'])}")
    if spec.get("transformer_dtype"):
        lines.append(f"transformer_dtype = '{spec['transformer_dtype']}'")
    if spec.get("flux_shift") or cfg.base_model == "flux-dev":
        lines.append("flux_shift = true")
    if cfg.pipe_type in ("wan", "ltx-video", "hunyuan-video", "qwen_image"):
        lines.append("timestep_sample_method = 'logit_normal'")
    if "min_t" in spec:
        lines.append(f"min_t = {spec['min_t']}")
        lines.append(f"max_t = {spec['max_t']}")
    if cfg.base_model == "sdxl":
        # Matches official diffusion-pipe SDXL example ratios, scaled to the job LR.
        # Default lr is 2e-5 → unet 4e-5, text encoders 2e-5.
        lines.append(f"unet_lr = {cfg.learning_rate * 2:g}")
        lines.append(f"text_encoder_1_lr = {cfg.learning_rate:g}")
        lines.append(f"text_encoder_2_lr = {cfg.learning_rate:g}")
    if blocks:
        lines += ["", f"blocks_to_swap = {blocks}"]
    if cfg.is_i2v:
        lines += ["video_clip_mode = 'single_beginning'"]

    lines += [
        "",
        "[adapter]",
        "type = 'lora'",
        f"rank = {cfg.rank}",
        f"dtype = '{dtype}'",
        "",
        "[optimizer]",
        f"type = '{opt}'",
        f"lr = {cfg.learning_rate:g}",
        "betas = [0.9, 0.99]",
        "weight_decay = 0.01",
    ]
    if opt == "AdamW8bitKahan":
        lines.append("stabilize = false")
    else:
        lines.append("eps = 1e-8")
    lines += [
        "",
        "[monitoring]",
        "enable_wandb = false",
        "",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def build_train_command(cfg: ResolvedConfig, train_toml: str, gpu_count: int,
                        gpu_name: str | None) -> tuple[list[str], dict[str, str]]:
    """Return (argv, extra env). GPU count must be divisible by pipeline_stages."""
    stages = cfg.pipeline_stages
    if gpu_count < 1:
        raise TrainingError("no CUDA GPU visible to the worker")
    if gpu_count % stages != 0:
        raise TrainingError(
            f"GPU count {gpu_count} is not divisible by pipeline_stages={stages}"
        )
    num = gpu_count  # use every visible GPU; data-parallel degree is num / stages
    script = os.path.join(PIPE_DIR, "train.py")
    # 1 GPU: call train.py directly. The Deepspeed launcher can hang after
    # "Unable to find hostfile..." on Runpod serverless (tiny /dev/shm, no pdsh).
    if num == 1:
        # Call train.py directly and pretend we were launched by torchrun.
        # Without RANK/WORLD_SIZE, deepspeed.init_distributed() falls back to
        # mpi4py, which is not in the image. The Deepspeed launcher itself can
        # hang after "Unable to find hostfile..." on Runpod serverless.
        cmd = [sys.executable, "-u", script, "--deepspeed", "--config", train_toml]
    else:
        cmd = [
            "deepspeed", f"--num_gpus={num}",
            "--master_addr=127.0.0.1", "--master_port=29500",
            script, "--deepspeed", "--config", train_toml,
        ]
    extra = {
        "PYTHONUNBUFFERED": "1",
        # Default Docker/serverless shm is often 64MiB; NCCL SHM init can hang.
        "NCCL_SHM_DISABLE": "1",
    }
    if num == 1:
        extra.update({
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        })
    if is_rtx_40xx(gpu_name) or os.environ.get("FORCE_NCCL_DISABLE") == "1":
        extra["NCCL_P2P_DISABLE"] = "1"
        extra["NCCL_IB_DISABLE"] = "1"
    return cmd, extra


def _read_tb_loss(run_dir: str) -> float | None:
    """Best-effort last train/loss from TensorBoard events (Deepspeed often prints little)."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return None
    events = sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True))
    if not events:
        # also check the run_dir itself and immediate children
        events = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
        for sub in glob.glob(os.path.join(run_dir, "*")):
            events += sorted(glob.glob(os.path.join(sub, "events.out.tfevents.*")))
    last = None
    for ev in events[-3:]:
        try:
            acc = EventAccumulator(os.path.dirname(ev))
            acc.Reload()
            if "train/loss" in acc.Tags().get("scalars", []):
                scalars = acc.Scalars("train/loss")
                if scalars:
                    last = float(scalars[-1].value)
        except Exception:
            continue
    return last


def run_command(cmd: list[str], log_path: str, progress, stage: str, total_steps: int | None = None,
                timeout_s: int | None = None, cwd: str | None = None,
                extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run Deepspeed / train.py, stream its log, parse steps and loss."""
    log.info("launching: %s", redact(" ".join(shlex.quote(c) for c in cmd)))
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra_env:
        env.update(extra_env)
    t0 = time.time()
    state = {"step": 0, "loss": None, "last_report": 0.0, "first_step_t": None, "tail": []}
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n# --- {stage} ---\n")
        proc = subprocess.Popen(
            cmd, cwd=cwd or PIPE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, bufsize=1, errors="replace",
        )

        def watchdog():
            if timeout_s:
                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    log.error("%s exceeded %ss, terminating", stage, timeout_s)
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()

        threading.Thread(target=watchdog, daemon=True).start()

        stop_hb = threading.Event()

        def heartbeat():
            while not stop_hb.wait(60):
                if proc.poll() is not None:
                    return
                now = time.time()
                if now - state["last_report"] < 55:
                    continue
                state["last_report"] = now
                payload = {"stage": stage, "heartbeat_s": round(now - t0),
                           "last_log": (state["tail"] or [""])[-1][:300]}
                if state["step"]:
                    payload["step"] = state["step"]
                if total_steps:
                    payload["total_steps"] = total_steps
                if state["loss"] is not None:
                    payload["loss"] = state["loss"]
                try:
                    progress(payload)
                except Exception:
                    log.debug("heartbeat progress failed", exc_info=True)

        threading.Thread(target=heartbeat, daemon=True).start()
        buf = ""
        try:
            while True:
                ch = proc.stdout.read(1)
                if ch == "" and proc.poll() is not None:
                    break
                if ch not in ("\n", "\r"):
                    buf += ch
                    continue
                line, buf = redact(buf), ""
                if not line.strip():
                    continue
                logf.write(line + "\n")
                logf.flush()
                state["tail"] = (state["tail"] + [line])[-80:]
                now = time.time()
                lm = LOSS_RE.search(line)
                if lm:
                    try:
                        state["loss"] = float(lm.group(1))
                    except ValueError:
                        pass
                # Prefer Deepspeed "step=N". Ignore caching/tqdm counters (e.g. "2/20")
                # unless the line clearly refers to training steps and matches total_steps.
                sm = re.search(r"(?:^|\[).*?\bstep[s]?\s*=\s*(\d+)", line, re.I)
                if not sm:
                    sm = re.search(r"\bstep[s]?\s*=\s*(\d+)", line, re.I)
                tm = re.search(r"\|\s*(\d+)/(\d+)\s*\[", line)
                step = None
                if sm:
                    step = int(sm.group(1))
                elif tm and total_steps and int(tm.group(2)) == int(total_steps) and re.search(
                        r"train|loss|it/s|s/it", line, re.I):
                    step = int(tm.group(1))
                if step and step > 0:
                    if state["first_step_t"] is None:
                        state["first_step_t"] = time.time()
                    state["step"] = step
                    if now - state["last_report"] > 15:
                        state["last_report"] = now
                        payload = {"stage": stage, "step": step}
                        if total_steps:
                            payload["total_steps"] = total_steps
                        if state["loss"] is not None:
                            payload["loss"] = state["loss"]
                        progress(payload)
                else:
                    log.info("pipe[%s]: %s", stage, line[:500])
                # Heartbeat so a silent hang is visible in progress_update.
                if now - state["last_report"] > 60:
                    state["last_report"] = now
                    payload = {"stage": stage, "heartbeat_s": round(now - t0),
                               "last_log": (state["tail"] or [""])[-1][:300]}
                    if state["step"]:
                        payload["step"] = state["step"]
                    if total_steps:
                        payload["total_steps"] = total_steps
                    if state["loss"] is not None:
                        payload["loss"] = state["loss"]
                    progress(payload)
        finally:
            stop_hb.set()
        rc = proc.wait()
    elapsed = time.time() - t0
    if rc != 0:
        tail = "\n".join(state["tail"][-25:])
        raise TrainingError(f"{stage} exited with code {rc}. Last log lines:\n{tail[-3000:]}")
    train_s = (time.time() - state["first_step_t"]) if state["first_step_t"] else None
    return {"global_step": state["step"], "final_loss": state["loss"],
            "total_seconds": round(elapsed, 1),
            "train_loop_seconds": round(train_s, 1) if train_s else None}


def find_outputs(out_dir: str) -> dict[str, Any]:
    """Locate the latest LoRA .safetensors under diffusion-pipe's timestamped run folder."""
    # output_dir / <YYYYMMDD_HH-MM-SS> / stepN|epochN / *.safetensors
    candidates: list[str] = []
    for pattern in ("**/step*/**/*.safetensors", "**/epoch*/**/*.safetensors",
                    "**/step*/*.safetensors", "**/epoch*/*.safetensors",
                    "**/*.safetensors"):
        candidates += glob.glob(os.path.join(out_dir, pattern), recursive=True)
    # Drop Deepspeed / optimizer shards if any slipped in, prefer lora.safetensors / adapter.
    uniq = []
    seen = set()
    for p in candidates:
        rp = os.path.realpath(p)
        if rp in seen:
            continue
        parts = rp.split(os.sep)
        if any(part.startswith("global_step") for part in parts):
            continue
        # Skip diffusion-pipe's in-progress save dir (…/stepN/tmp/*.bin), not a host path like /tmp/…
        if os.path.basename(os.path.dirname(rp)) == "tmp":
            continue
        if not os.path.isfile(rp):
            continue
        seen.add(rp)
        uniq.append(rp)

    def sort_key(p: str) -> tuple:
        parent = os.path.basename(os.path.dirname(p))
        m = re.search(r"(?:step|epoch)(\d+)", parent)
        n = int(m.group(1)) if m else 0
        prefer = 0 if os.path.basename(p) in ("lora.safetensors", "adapter_model.safetensors") else 1
        return (n, -os.path.getmtime(p), prefer, p)

    uniq.sort(key=sort_key)
    lora = uniq[-1] if uniq else None
    ckpts = [p for p in uniq if p != lora]
    return {"lora_path": lora, "checkpoints": ckpts, "samples": []}


def copy_final_lora(lora_path: str, dest_dir: str, output_name: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{output_name}.safetensors")
    if os.path.realpath(lora_path) != os.path.realpath(dest):
        shutil.copy2(lora_path, dest)
    # Also copy a PEFT adapter_config.json if it sits next to the original file.
    side = os.path.join(os.path.dirname(lora_path), "adapter_config.json")
    if os.path.isfile(side):
        shutil.copy2(side, os.path.join(dest_dir, "adapter_config.json"))
    return dest


def cleanup_deepspeed_checkpoints(out_dir: str) -> int:
    """Remove Deepspeed global_step* folders (optimizer state) after a successful run."""
    removed = 0
    for d in glob.glob(os.path.join(out_dir, "**", "global_step*"), recursive=True):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


def fill_loss_from_tensorboard(train_info: dict[str, Any], out_dir: str) -> None:
    if train_info.get("final_loss") is not None:
        return
    loss = _read_tb_loss(out_dir)
    if loss is not None:
        train_info["final_loss"] = loss
