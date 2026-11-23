"""Base model files: download at job start, cache on the network volume if attached.

Official default paths use bf16 weights (SDXL 0.9vae, Wan-AI catalogs, FLUX.1-dev).
We never download Comfy ``fp8_scaled`` files for these v1 defaults.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Any

from .paths import has_volume, model_cache_dir
from .secret_utils import redact

log = logging.getLogger(__name__)

# Single-file layouts
SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_FILE = "sd_xl_base_1.0_0.9vae.safetensors"
SDXL_SIZE = 6_938_078_334

# Official catalogs (directories) used as ckpt_path / diffusers_path.
SNAPSHOTS: dict[str, dict[str, Any]] = {
    "sdxl": {
        "kind": "file",
        "repo": SDXL_REPO,
        "filename": SDXL_FILE,
        "size": SDXL_SIZE,
        "gated": False,
        "path_key": "checkpoint_path",
    },
    "wan-t2v-1.3b": {
        "kind": "snapshot",
        "repo": "Wan-AI/Wan2.1-T2V-1.3B",
        "gated": False,
        "path_key": "ckpt_path",
    },
    "wan-t2v-14b": {
        "kind": "snapshot",
        "repo": "Wan-AI/Wan2.1-T2V-14B",
        "gated": False,
        "path_key": "ckpt_path",
        "transformer_dtype": "float8",
    },
    "wan-i2v-14b": {
        "kind": "snapshot",
        "repo": "Wan-AI/Wan2.1-I2V-14B-720P",
        "gated": False,
        "path_key": "ckpt_path",
        "transformer_dtype": "float8",
    },
    "wan22-t2v-a14b-low": {
        "kind": "snapshot",
        "repo": "Wan-AI/Wan2.2-T2V-A14B",
        "gated": False,
        "path_key": "ckpt_path",
        "transformer_subdir": "low_noise_model",
        "transformer_dtype": "float8",
        "min_t": 0.0,
        "max_t": 0.875,
    },
    "wan22-t2v-a14b-high": {
        "kind": "snapshot",
        "repo": "Wan-AI/Wan2.2-T2V-A14B",
        "gated": False,
        "path_key": "ckpt_path",
        "transformer_subdir": "high_noise_model",
        "transformer_dtype": "float8",
        "min_t": 0.875,
        "max_t": 1.0,
    },
    "wan22-i2v-a14b-low": {
        "kind": "snapshot",
        "repo": "Wan-AI/Wan2.2-I2V-A14B",
        "gated": False,
        "path_key": "ckpt_path",
        "transformer_subdir": "low_noise_model",
        "transformer_dtype": "float8",
        "min_t": 0.0,
        "max_t": 0.9,
    },
    "wan22-i2v-a14b-high": {
        "kind": "snapshot",
        "repo": "Wan-AI/Wan2.2-I2V-A14B",
        "gated": False,
        "path_key": "ckpt_path",
        "transformer_subdir": "high_noise_model",
        "transformer_dtype": "float8",
        "min_t": 0.9,
        "max_t": 1.0,
    },
    "flux-dev": {
        "kind": "snapshot",
        "repo": "black-forest-labs/FLUX.1-dev",
        "gated": True,
        "path_key": "diffusers_path",
        "transformer_dtype": "float8",
        "flux_shift": True,
    },
    "ltx-video": {
        "kind": "snapshot",
        "repo": "Lightricks/LTX-Video",
        "gated": False,
        "path_key": "diffusers_path",
        # single_file_path is resolved after the snapshot (latest *2b*.safetensors or similar).
        "single_file_glob": "*.safetensors",
    },
    "qwen-image": {
        "kind": "snapshot",
        "repo": "Qwen/Qwen-Image",
        "gated": False,  # may become gated; check_access will say so
        "path_key": "diffusers_path",
        "transformer_dtype": "float8",
    },
    "hunyuan-video": {
        "kind": "stub",
        "repo": None,
        "gated": True,
        "reason": (
            "hunyuan-video is a v1 should-have without a single public catalog pin in this worker. "
            "Use sdxl, wan-t2v-1.3b / 14b, or flux-dev. A later release will add the ComfyUI file set."
        ),
    },
}

FORBIDDEN_SUBSTR = ("fp8_scaled", "fp8scaled")


class ModelError(RuntimeError):
    pass


def spec_for(cfg) -> dict[str, Any]:
    try:
        return SNAPSHOTS[cfg.base_model]
    except KeyError as exc:
        raise ModelError(f"no download recipe for {cfg.base_model}") from exc


def _assert_not_fp8_scaled(name: str) -> None:
    low = name.lower()
    if any(s in low for s in FORBIDDEN_SUBSTR):
        raise ModelError(
            f"refusing {name}: v1 default paths do not use ready-made Comfy fp8_scaled weights. "
            "Official catalogs are bf16; transformer_dtype=float8 is applied at training time."
        )


def _hf_headers(token: str | None) -> dict[str, str]:
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def check_access(cfg, token: str | None) -> dict[str, Any]:
    """Metadata-only check (no weights) that the official catalog or file is reachable."""
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

    spec = spec_for(cfg)
    if spec["kind"] == "stub":
        return {"model": {"ok": False, "error": spec["reason"]}}
    repo = spec["repo"]
    try:
        if spec["kind"] == "file":
            _assert_not_fp8_scaled(spec["filename"])
            meta = get_hf_file_metadata(hf_hub_url(repo, spec["filename"]), token=token)
            size = meta.size or 0
            expected = spec.get("size")
            out = {"ok": True, "repo": repo, "file": spec["filename"],
                   "size_gb": round(size / 1e9, 2)}
            if expected and abs(size - expected) > 1024:
                out["warning"] = f"size {size} != expected {expected}"
            return {"checkpoint": out}
        api = HfApi(token=token)
        info = api.model_info(repo, files_metadata=False)
        gated = bool(getattr(info, "gated", False) or spec.get("gated"))
        if gated and not token:
            return {"model": {
                "ok": False, "repo": repo,
                "error": (f"{repo} is gated: accept the license on huggingface.co "
                          "and set HF_TOKEN with read access as a Runpod secret"),
            }}
        siblings = list(getattr(info, "siblings", None) or [])
        total = 0
        for s in siblings:
            name = getattr(s, "rfilename", "") or ""
            _assert_not_fp8_scaled(name)
            total += int(getattr(s, "size", 0) or 0)
        return {"model": {"ok": True, "repo": repo, "files": len(siblings),
                          "size_gb": round(total / 1e9, 2) if total else None, "gated": gated}}
    except Exception as exc:
        msg = redact(str(exc)).splitlines()[0][:240]
        hint = ""
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            hint = " (gated: accept the model license on huggingface.co and set HF_TOKEN with read access)"
        return {"model": {"ok": False, "repo": repo, "error": msg + hint}}


def _local_dir(repo: str) -> str:
    return os.path.join(model_cache_dir(), repo.replace("/", "--"))


def _hf_file(repo: str, filename: str, dest_dir: str, expected: int | None, token: str | None) -> str:
    from huggingface_hub import hf_hub_download

    _assert_not_fp8_scaled(filename)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, filename)
    if expected and os.path.isfile(dest) and os.path.getsize(dest) == expected:
        log.info("cache hit %s", dest)
        return os.path.realpath(dest)
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            path = os.path.realpath(hf_hub_download(
                repo_id=repo, filename=filename, token=token, local_dir=dest_dir,
            ))
            if expected and os.path.getsize(path) != expected:
                log.warning("size mismatch for %s: got %s expected %s (attempt %s)",
                            filename, os.path.getsize(path), expected, attempt)
                try:
                    os.remove(path)
                except OSError:
                    pass
                last_err = ModelError(f"{filename}: unexpected size")
                continue
            return path
        except Exception as exc:
            last_err = exc
            log.warning("download of %s failed (attempt %s): %s", filename, attempt, redact(str(exc))[:200])
            time.sleep(min(8, 2 ** attempt))
    raise ModelError(f"download of {repo}/{filename} failed: {redact(str(last_err)).splitlines()[0][:300]}")


def _snapshot(repo: str, dest_dir: str, token: str | None, progress=None) -> str:
    from huggingface_hub import snapshot_download

    os.makedirs(dest_dir, exist_ok=True)
    marker = os.path.join(dest_dir, ".snapshot_complete")
    if os.path.isdir(dest_dir) and os.path.isfile(marker):
        log.info("cache hit snapshot %s", dest_dir)
        return dest_dir
    if progress:
        progress({"stage": "downloading_model", "repo": repo})
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            path = snapshot_download(repo_id=repo, token=token, local_dir=dest_dir)
            # Refuse if the snapshot only contains fp8_scaled transformers we would use as defaults.
            for root, _dirs, files in os.walk(path):
                for f in files:
                    # Presence of fp8_scaled files next to bf16 is OK; we just will not point at them.
                    if any(s in f.lower() for s in FORBIDDEN_SUBSTR):
                        log.info("snapshot contains %s (will not be used as a default path)", f)
            open(marker, "w").close()
            return path
        except Exception as exc:
            last_err = exc
            log.warning("snapshot of %s failed (attempt %s): %s", repo, attempt, redact(str(exc))[:200])
            time.sleep(min(8, 2 ** attempt))
    raise ModelError(f"download of {repo} failed: {redact(str(last_err)).splitlines()[0][:300]}")


def _pick_single_file(root: str, pattern: str) -> str | None:
    import glob as _glob
    hits = [p for p in _glob.glob(os.path.join(root, pattern)) if os.path.isfile(p)]
    hits += [p for p in _glob.glob(os.path.join(root, "**", pattern), recursive=True) if os.path.isfile(p)]
    # Prefer names that look like a main transformer / single-file checkpoint, skip tiny files.
    uniq = sorted({os.path.realpath(p) for p in hits}, key=lambda p: (-os.path.getsize(p), p))
    for p in uniq:
        _assert_not_fp8_scaled(os.path.basename(p))
        if os.path.getsize(p) > 50 * 2**20:
            return p
    return uniq[0] if uniq else None


def ensure_models(cfg, token: str | None, progress=None) -> tuple[dict[str, str], dict[str, Any]]:
    """Download the official catalog (cached). Returns path map and timing info."""
    spec = spec_for(cfg)
    if spec["kind"] == "stub":
        raise ModelError(spec["reason"])
    if spec.get("gated") and not token:
        raise ModelError(
            f"{spec['repo']} is gated: accept the license on huggingface.co and set HF_TOKEN "
            "with read access as a Runpod secret (never in the job input)"
        )
    cache = model_cache_dir()
    os.makedirs(cache, exist_ok=True)
    t0 = time.time()
    paths: dict[str, str] = {}
    repo = spec["repo"]
    if progress:
        progress({"stage": "downloading_model", "repo": repo})
    try:
        if spec["kind"] == "file":
            dest = _local_dir(repo)
            paths["checkpoint_path"] = _hf_file(repo, spec["filename"], dest, spec.get("size"), token)
        else:
            dest = _local_dir(repo)
            root = _snapshot(repo, dest, token, progress)
            paths[spec["path_key"]] = root
            sub = spec.get("transformer_subdir")
            if sub:
                tpath = os.path.join(root, sub)
                if not os.path.isdir(tpath):
                    raise ModelError(f"{repo} is missing transformer subfolder {sub}")
                paths["transformer_path"] = tpath
            if spec.get("single_file_glob"):
                picked = _pick_single_file(root, spec["single_file_glob"])
                if picked:
                    paths["single_file_path"] = picked
    except ModelError:
        raise
    except Exception as exc:
        msg = redact(str(exc)).splitlines()[0][:300]
        if spec.get("gated") or "401" in msg or "403" in msg:
            msg += " (gated model: accept the license on huggingface.co and set HF_TOKEN)"
        raise ModelError(f"download of {repo} failed: {msg}") from exc

    sizes = {}
    for k, v in paths.items():
        if os.path.isfile(v):
            sizes[k] = round(os.path.getsize(v) / 1e9, 2)
        elif os.path.isdir(v):
            total = 0
            for root, _dirs, files in os.walk(v):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            sizes[k] = round(total / 1e9, 2)
    free_gb = round(shutil.disk_usage(cache).free / 1e9, 1)
    info = {"download_seconds": round(time.time() - t0, 1), "sizes_gb": sizes,
            "cache_dir": cache, "disk_free_gb": free_gb, "on_volume": has_volume(),
            "dtype": cfg.normalized_dtype}
    if not has_volume():
        info["warning"] = (
            "no network volume: weights were stored under /tmp and will be re-downloaded on the next worker"
        )
    return paths, info
