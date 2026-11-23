"""Input schema for the diffusion-pipe multi-model LoRA trainer.

Every parameter comes from (highest priority first):

1. the job ``input`` object,
2. an environment variable (what the Runpod Hub deploy form sets),
3. the preset defaults,
4. the built-in default.

Dependency-free so it can be unit-tested and used for a dry run without a GPU.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
URL_RE = re.compile(r"^https?://\S+$")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
TRIGGER_RE = re.compile(r"^[^\n\r,]{1,64}$")

TRUE_STRINGS = {"1", "true", "yes", "y", "on", "enabled"}
FALSE_STRINGS = {"0", "false", "no", "n", "off", "disabled", ""}

# v1 must-support + should-support. Coming-soon IDs are recognized so the error is explicit.
MUST_MODELS = ("sdxl", "wan-t2v-1.3b", "wan-t2v-14b", "flux-dev")
SHOULD_MODELS = (
    "wan-i2v-14b",
    "wan22-t2v-a14b-low",
    "wan22-t2v-a14b-high",
    "wan22-i2v-a14b-low",
    "wan22-i2v-a14b-high",
    "hunyuan-video",
    "ltx-video",
    "qwen-image",
)
SUPPORTED_MODELS = MUST_MODELS + SHOULD_MODELS

# Upstream supports many more architectures; v1 does not ship presets for these.
COMING_SOON_MODELS = (
    "flux-2", "flux-kontext", "chroma", "hidream", "sd3", "cosmos", "cosmos-predict2",
    "lumina", "omnigen2", "auraflow", "z-image", "hunyuan-video-1.5", "hunyuan-image-2.1",
    "anima", "ernie-image", "ltx-2.3", "ideogram4", "krea-2", "minimax-h3",
    "minimax-h3-image", "moss-image", "qwen-image-edit", "qwen-image-2.1",
    "wan22-t2v-5b",
)
BASE_MODELS = SUPPORTED_MODELS + COMING_SOON_MODELS

# diffusion-pipe [model].type
PIPE_TYPE = {
    "sdxl": "sdxl",
    "wan-t2v-1.3b": "wan",
    "wan-t2v-14b": "wan",
    "wan-i2v-14b": "wan",
    "wan22-t2v-a14b-low": "wan",
    "wan22-t2v-a14b-high": "wan",
    "wan22-i2v-a14b-low": "wan",
    "wan22-i2v-a14b-high": "wan",
    "flux-dev": "flux",
    "hunyuan-video": "hunyuan-video",
    "ltx-video": "ltx-video",
    "qwen-image": "qwen_image",
}

LORA_FORMAT = {
    "sdxl": "kohya",
    "wan-t2v-1.3b": "comfyui",
    "wan-t2v-14b": "comfyui",
    "wan-i2v-14b": "comfyui",
    "wan22-t2v-a14b-low": "comfyui",
    "wan22-t2v-a14b-high": "comfyui",
    "wan22-i2v-a14b-low": "comfyui",
    "wan22-i2v-a14b-high": "comfyui",
    "flux-dev": "diffusers",
    "hunyuan-video": "diffusers",
    "ltx-video": "comfyui",
    "qwen-image": "comfyui",
}

I2V_MODELS = {
    "wan-i2v-14b",
    "wan22-i2v-a14b-low",
    "wan22-i2v-a14b-high",
}
VIDEO_MODELS = I2V_MODELS | {
    "wan-t2v-1.3b", "wan-t2v-14b",
    "wan22-t2v-a14b-low", "wan22-t2v-a14b-high",
    "hunyuan-video", "ltx-video",
}
IS_14B = {
    "wan-t2v-14b", "wan-i2v-14b",
    "wan22-t2v-a14b-low", "wan22-t2v-a14b-high",
    "wan22-i2v-a14b-low", "wan22-i2v-a14b-high",
}
WAN22_MODELS = {
    "wan22-t2v-a14b-low", "wan22-t2v-a14b-high",
    "wan22-i2v-a14b-low", "wan22-i2v-a14b-high",
}
# models that need a Hugging Face token for the official weights
GATED_MODELS = {"flux-dev"}
# Official weights are bfloat16; do not default to fp16 (lesson from the musubi worker).
DEFAULT_DTYPE = "bfloat16"
DTYPE_CHOICES = ("bfloat16", "float16", "fp16", "bf16")

# Preset defaults applied only when the field was not set by input or env.
# Learning rates follow diffusion-pipe examples (~2e-5), not musubi's 2e-4.
PRESETS: dict[str, dict[str, Any]] = {
    "sdxl": {"steps": 2000, "rank": 16, "learning_rate": 2e-5, "resolution": (1024, 1024)},
    "wan-t2v-1.3b": {"steps": 2000, "rank": 16, "learning_rate": 2e-5, "resolution": (832, 480)},
    "wan-t2v-14b": {"steps": 2000, "rank": 16, "learning_rate": 2e-5, "resolution": (512, 512)},
    "wan-i2v-14b": {"steps": 2000, "rank": 16, "learning_rate": 2e-5, "resolution": (512, 512)},
    "wan22-t2v-a14b-low": {"steps": 2000, "rank": 16, "learning_rate": 2e-5, "resolution": (512, 512)},
    "wan22-t2v-a14b-high": {"steps": 2000, "rank": 16, "learning_rate": 2e-5, "resolution": (512, 512)},
    "wan22-i2v-a14b-low": {"steps": 2000, "rank": 16, "learning_rate": 2e-5, "resolution": (512, 512)},
    "wan22-i2v-a14b-high": {"steps": 2000, "rank": 16, "learning_rate": 2e-5, "resolution": (512, 512)},
    "flux-dev": {"steps": 2000, "rank": 16, "learning_rate": 2e-5, "resolution": (1024, 1024)},
    "hunyuan-video": {"steps": 2000, "rank": 16, "learning_rate": 2e-5, "resolution": (512, 512)},
    "ltx-video": {"steps": 2000, "rank": 16, "learning_rate": 2e-5, "resolution": (512, 512)},
    "qwen-image": {"steps": 2000, "rank": 32, "learning_rate": 2e-5, "resolution": (640, 640)},
    "quick": {"steps": 20, "rank": 8, "learning_rate": 2e-5, "resolution": (512, 512)},
}


@dataclass(frozen=True)
class Param:
    name: str
    type: type
    default: Any
    env: str | None = None
    min: float | None = None
    max: float | None = None
    choices: tuple | None = None
    help: str = ""


class ResolutionType:
    pass


class FramesType:
    pass


class BlocksType:
    pass


PARAMS: list[Param] = [
    Param("base_model", str, "sdxl", "BASE_MODEL", choices=BASE_MODELS,
          help="sdxl is the default (no token, ~6.5 GiB, 24 GB). Also Wan 2.1/2.2 and Flux.1-dev."),
    Param("dataset_url", str, "", "DATASET_URL",
          help="https URL to a .zip / .tar(.gz) of images and/or videos with optional same-name .txt captions."),
    Param("trigger_word", str, "", "TRIGGER_WORD", help="Unique token for the concept, e.g. 'zxowl'."),
    Param("class_word", str, "", "CLASS_WORD", help="Generic class, e.g. 'figurine'."),
    Param("caption_mode", str, "auto", "CAPTION_MODE", choices=("auto", "existing", "trigger"),
          help="auto = use .txt captions when present, else trigger+class; trigger = ignore captions; "
               "existing = prefer .txt files."),
    Param("prepend_trigger", bool, True, "PREPEND_TRIGGER",
          help="Make sure every caption starts with the trigger word."),
    Param("preset", str, "sdxl", "PRESET", choices=tuple(PRESETS),
          help="Named defaults for steps/rank/lr/resolution. Explicit settings win."),
    Param("steps", int, None, "STEPS", min=10, max=20000),
    Param("rank", int, None, "LORA_RANK", min=1, max=256),
    Param("learning_rate", float, None, "LEARNING_RATE", min=1e-7, max=1e-2),
    Param("resolution", ResolutionType, None, "RESOLUTION",
          help="Training resolution: 512 or '1024x1024' or [w, h]. A single number is square."),
    Param("batch_size", int, 1, "BATCH_SIZE", min=1, max=8),
    Param("frames", FramesType, None, None,
          help="Alias of frame_buckets: int or list. Images always also get a 1-frame bucket."),
    Param("frame_buckets", FramesType, None, "FRAME_BUCKETS",
          help="Video frame bucket lengths. Default [1] for image models, [1, 17] for video models."),
    Param("pipeline_stages", int, 1, "PIPELINE_STAGES", min=1, max=8,
          help="Deepspeed pipeline parallelism degree. GPU count must be divisible by this. Default 1."),
    Param("blocks_to_swap", BlocksType, "auto", "BLOCKS_TO_SWAP",
          help="'auto' picks a value from the model and VRAM, 0 disables swap, or an integer. "
               "Incompatible with pipeline_stages > 1."),
    Param("dtype", str, DEFAULT_DTYPE, "DTYPE",
          help="Must match the dtype of the weight files. Official default paths are bfloat16."),
    Param("seed", int, 42, "SEED", min=0),
    Param("save_every_n_steps", int, 0, "SAVE_EVERY_N_STEPS", min=0,
          help="Intermediate LoRA files (kept on the network volume). 0 = only the final LoRA."),
    Param("num_samples", int, 0, "NUM_SAMPLES", min=0, max=4,
          help="Accepted for compatibility; this worker does not generate sample images/videos in v1."),
    Param("return_lora_base64", bool, False, "RETURN_LORA_BASE64",
          help="Include the LoRA as base64 in the job output (only if the file is small)."),
    Param("output_name", str, "", "OUTPUT_NAME", help="LoRA file name (default: '<trigger>_<base_model>_lora')."),
    Param("run_name", str, "", "RUN_NAME", help="Folder name for outputs. Default: the Runpod job id."),
    Param("hf_repo_id", str, "", "HF_OUTPUT_REPO", help="Upload the LoRA here (needs HF_TOKEN with write)."),
    Param("hf_private", bool, True, "HF_PRIVATE"),
    Param("upload_url", str, "", None, help="Presigned HTTP PUT URL for the .safetensors file."),
    Param("upload_to_bucket", bool, True, "UPLOAD_TO_BUCKET"),
    Param("dry_run", bool, False, "DRY_RUN",
          help="Download + validate the dataset, check model URLs, return the plan. No GPU work."),
    Param("max_media", int, 200, "MAX_MEDIA", min=1, max=5000,
          help="Max number of images + videos to use from the archive."),
]

PARAM_BY_NAME = {p.name: p for p in PARAMS}

ALIASES = {
    "model": "base_model",
    "dataset": "dataset_url",
    "zip_url": "dataset_url",
    "trigger": "trigger_word",
    "lr": "learning_rate",
    "max_train_steps": "steps",
    "network_dim": "rank",
    "lora_rank": "rank",
    "output_repo": "hf_repo_id",
    "presigned_url": "upload_url",
    "target_frames": "frame_buckets",
}


class ConfigError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class ResolvedConfig:
    values: dict[str, Any]
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __getattr__(self, item: str) -> Any:
        try:
            return self.values[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    @property
    def pipe_type(self) -> str:
        return PIPE_TYPE[self.values["base_model"]]

    @property
    def lora_format(self) -> str:
        return LORA_FORMAT[self.values["base_model"]]

    @property
    def is_i2v(self) -> bool:
        return self.values["base_model"] in I2V_MODELS

    @property
    def is_video_model(self) -> bool:
        return self.values["base_model"] in VIDEO_MODELS

    @property
    def is_wan22(self) -> bool:
        return self.values["base_model"] in WAN22_MODELS

    @property
    def is_14b(self) -> bool:
        return self.values["base_model"] in IS_14B

    @property
    def is_gated(self) -> bool:
        return self.values["base_model"] in GATED_MODELS

    @property
    def resolution_wh(self) -> tuple[int, int]:
        return self.values["resolution"]

    @property
    def normalized_dtype(self) -> str:
        d = str(self.values["dtype"]).lower()
        if d in ("fp16", "float16"):
            return "float16"
        if d in ("bf16", "bfloat16"):
            return "bfloat16"
        return d

    def public_dict(self) -> dict[str, Any]:
        out = dict(self.values)
        if out.get("upload_url"):
            out["upload_url"] = "<presigned url>"
        if isinstance(out.get("resolution"), tuple):
            out["resolution"] = list(out["resolution"])
        return out


def _parse_resolution(raw: Any) -> tuple[int, int]:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return int(raw[0]), int(raw[1])
    if isinstance(raw, bool):
        raise ValueError("resolution: expected a number or WxH, got boolean")
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"resolution: expected integer or WxH, got {raw!r}")
        n = int(raw)
        return n, n
    s = str(raw).strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d+)[x,;](\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    if re.fullmatch(r"\d+", s):
        n = int(s)
        return n, n
    raise ValueError(f"resolution: expected a number or WxH (e.g. 1024 or 832x480), got {raw!r}")


def _parse_frames(raw: Any) -> list[int]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, bool):
        raise ValueError("frame_buckets: expected int or list, got boolean")
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"frame_buckets: expected integers, got {raw!r}")
        return [int(raw)]
    if isinstance(raw, (list, tuple)):
        out = []
        for x in raw:
            out.extend(_parse_frames(x))
        return out
    s = str(raw).strip()
    if not s:
        return []
    parts = re.split(r"[,\s]+", s)
    try:
        return [int(p) for p in parts if p]
    except ValueError as exc:
        raise ValueError(f"frame_buckets: expected integers, got {raw!r}") from exc


def _parse_blocks(raw: Any) -> Any:
    if raw is None or raw == "":
        return "auto"
    if isinstance(raw, bool):
        raise ValueError("blocks_to_swap: expected 'auto' or an integer, got boolean")
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"blocks_to_swap: expected integer or 'auto', got {raw!r}")
        return int(raw)
    s = str(raw).strip().lower()
    if s == "auto":
        return "auto"
    try:
        return int(s)
    except ValueError as exc:
        raise ValueError(f"blocks_to_swap: expected 'auto' or an integer, got {raw!r}") from exc


def _coerce(param: Param, raw: Any) -> Any:
    t = param.type
    if t is ResolutionType:
        return _parse_resolution(raw)
    if t is FramesType:
        return _parse_frames(raw)
    if t is BlocksType:
        return _parse_blocks(raw)
    if t is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)) and raw in (0, 1):
            return bool(raw)
        s = str(raw).strip().lower()
        if s in TRUE_STRINGS:
            return True
        if s in FALSE_STRINGS:
            return False
        raise ValueError(f"{param.name}: expected boolean, got {raw!r}")
    if t is int:
        if isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected integer, got boolean")
        if isinstance(raw, float):
            if not raw.is_integer():
                raise ValueError(f"{param.name}: expected integer, got {raw!r}")
            return int(raw)
        s = str(raw).strip()
        try:
            return int(float(s)) if re.fullmatch(r"-?\d+(\.0+)?", s) else int(s)
        except ValueError as exc:
            raise ValueError(f"{param.name}: expected integer, got {raw!r}") from exc
    if t is float:
        if isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected number, got boolean")
        try:
            return float(str(raw).strip())
        except ValueError as exc:
            raise ValueError(f"{param.name}: expected number, got {raw!r}") from exc
    if t is list:
        if isinstance(raw, list):
            return [str(x) for x in raw]
        if isinstance(raw, str):
            return [x.strip() for x in raw.split("|") if x.strip()]
        raise ValueError(f"{param.name}: expected a list of strings")
    if t is str:
        if not isinstance(raw, (str, int, float)) or isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected string")
        return str(raw).strip()
    raise TypeError(t)  # pragma: no cover


def normalize_dtype(raw: str) -> str:
    d = str(raw).lower().strip()
    if d in ("fp16", "float16"):
        return "float16"
    if d in ("bf16", "bfloat16"):
        return "bfloat16"
    raise ValueError(f"dtype: must be bfloat16 or float16, got {raw!r}")


def resolve_config(job_input: Mapping[str, Any] | None,
                   env: Mapping[str, str] | None = None) -> ResolvedConfig:
    env = os.environ if env is None else env
    job_input = dict(job_input or {})
    errors: list[str] = []
    warnings: list[str] = []

    for alias, canonical in ALIASES.items():
        if alias in job_input:
            if canonical in job_input:
                errors.append(f"both '{alias}' and '{canonical}' given")
            else:
                job_input[canonical] = job_input.pop(alias)

    if "frames" in job_input and "frame_buckets" in job_input:
        errors.append("both 'frames' and 'frame_buckets' given")
    elif "frames" in job_input and "frame_buckets" not in job_input:
        job_input["frame_buckets"] = job_input["frames"]

    for secret_like in ("hf_token", "HF_TOKEN", "token", "aws_secret_access_key", "secret_key"):
        if secret_like in job_input:
            errors.append("do not send tokens/keys in the job input; set HF_TOKEN / BUCKET_* as Runpod secrets")
            job_input.pop(secret_like, None)

    unknown = sorted(k for k in job_input if k not in PARAM_BY_NAME)
    if unknown:
        errors.append(f"unknown input field(s): {', '.join(unknown)}")

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for p in PARAMS:
        raw, src = None, "default"
        if p.name in job_input and job_input[p.name] is not None:
            raw, src = job_input[p.name], "input"
        elif p.env and env.get(p.env) not in (None, ""):
            raw, src = env.get(p.env), "env"
        val = p.default
        if src != "default":
            try:
                val = _coerce(p, raw)
            except ValueError as exc:
                errors.append(str(exc))
        values[p.name], sources[p.name] = val, src

    preset_name = values.get("preset") or "sdxl"
    preset = PRESETS.get(preset_name)
    if preset is None:
        errors.append(f"preset: must be one of {list(PRESETS)}, got {preset_name!r}")
        preset = PRESETS["sdxl"]
    for k, v in preset.items():
        if values.get(k) is None:
            values[k], sources[k] = (list(v) if isinstance(v, list) else v), "preset"

    if values.get("steps") is None:
        values["steps"], sources["steps"] = 2000, "default"
    if values.get("rank") is None:
        values["rank"], sources["rank"] = 16, "default"
    if values.get("learning_rate") is None:
        values["learning_rate"], sources["learning_rate"] = 2e-5, "default"
    if values.get("resolution") is None:
        values["resolution"], sources["resolution"] = (1024, 1024), "default"

    bm = values.get("base_model") or "sdxl"
    if values.get("frame_buckets") is None:
        default_frames = [1, 17] if bm in VIDEO_MODELS else [1]
        values["frame_buckets"], sources["frame_buckets"] = default_frames, "default"
    if not values.get("frame_buckets"):
        values["frame_buckets"] = [1]

    for p in PARAMS:
        val = values[p.name]
        if p.type in (int, float) and isinstance(val, (int, float)) and not isinstance(val, bool):
            if p.min is not None and val < p.min:
                errors.append(f"{p.name}: must be >= {p.min}, got {val}")
            if p.max is not None and val > p.max:
                errors.append(f"{p.name}: must be <= {p.max}, got {val}")
        if p.choices and val not in p.choices and val is not None and p.type is not BlocksType:
            errors.append(f"{p.name}: must be one of {list(p.choices)}, got {val!r}")

    if bm in COMING_SOON_MODELS:
        errors.append(
            f"{bm} is supported by upstream diffusion-pipe but is not in this worker's v1 presets; "
            f"v1 trains {', '.join(SUPPORTED_MODELS)}"
        )

    url = values["dataset_url"]
    if not url:
        errors.append("dataset: set 'dataset_url' to an https URL of a .zip / .tar(.gz) of images and/or videos")
    elif not URL_RE.match(url):
        errors.append("dataset_url: expected an http(s) URL")

    tw = values["trigger_word"]
    if tw and not TRIGGER_RE.match(tw):
        errors.append("trigger_word: one short phrase, no commas or newlines (max 64 chars)")
    if not tw:
        if values["caption_mode"] == "trigger":
            errors.append("caption_mode 'trigger' needs trigger_word")
        else:
            warnings.append("no trigger_word: the LoRA will be driven by your captions only")

    res = values["resolution"]
    if not (isinstance(res, tuple) and len(res) == 2):
        errors.append("resolution: expected a number or [width, height]")
    else:
        w, h = res
        if min(w, h) < 256 or max(w, h) > 2048:
            errors.append(f"resolution: each side must be 256–2048, got {w}x{h}")
        if w % 16 or h % 16:
            errors.append(f"resolution must be a multiple of 16, got {w}x{h}")

    frames = values["frame_buckets"]
    if not isinstance(frames, list) or not frames:
        errors.append("frame_buckets: provide at least one length (images use 1)")
    else:
        bad = [n for n in frames if not isinstance(n, int) or n < 1 or n > 257]
        if bad:
            errors.append(f"frame_buckets: each length must be an integer 1–257; invalid: {bad}")
        # Images always need the 1-frame bucket; keep it present.
        if 1 not in frames:
            values["frame_buckets"] = [1, *frames]
            warnings.append("added frame bucket 1 (required for still images)")

    try:
        values["dtype"] = normalize_dtype(values["dtype"])
    except ValueError as exc:
        errors.append(str(exc))

    if values["dtype"] == "float16" and sources.get("dtype") in ("input", "env"):
        warnings.append(
            "dtype=float16: only use this when the weight files themselves are fp16 "
            "(e.g. Comfy fp16 safetensors). Official SDXL / Wan-AI / FLUX.1-dev weights are bfloat16."
        )

    blocks = values["blocks_to_swap"]
    if blocks != "auto":
        if not isinstance(blocks, int) or blocks < 0:
            errors.append("blocks_to_swap: must be 'auto' or an integer >= 0")
        elif values["pipeline_stages"] > 1 and blocks > 0:
            errors.append("blocks_to_swap is incompatible with pipeline_stages > 1; set one of them to 1 / 0")

    if values["pipeline_stages"] > 1:
        warnings.append(
            f"pipeline_stages={values['pipeline_stages']}: the endpoint must have a GPU count "
            "divisible by this (e.g. 2 GPUs with pipeline_stages=2). Default Hub deploys use 1 GPU."
        )

    if values["hf_repo_id"] and not HF_REPO_RE.match(values["hf_repo_id"]):
        errors.append("hf_repo_id: expected 'owner/name'")
    if values["upload_url"] and not values["upload_url"].startswith("https://"):
        errors.append("upload_url: expected an https presigned PUT URL")
    if values["run_name"] and not RUN_NAME_RE.match(values["run_name"]):
        errors.append("run_name: letters, digits, '.', '_' or '-' (max 100 chars)")
    if values["output_name"] and not RUN_NAME_RE.match(values["output_name"]):
        errors.append("output_name: letters, digits, '.', '_' or '-' (max 100 chars)")

    has_token = bool((env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
                     and not (env.get("HF_TOKEN") or "").startswith("{{"))
    if values["hf_repo_id"] and not has_token:
        msg = "hf_repo_id is set but HF_TOKEN is not; add HF_TOKEN (write access) as a Runpod secret"
        (warnings if values["dry_run"] else errors).append(msg)
    if bm in GATED_MODELS and not has_token:
        msg = (f"{bm} is a gated model: accept the license on huggingface.co and set HF_TOKEN "
               "(read access) as a Runpod secret. Do not put the token in the job input.")
        (warnings if values["dry_run"] else errors).append(msg)

    if bm in IS_14B or bm == "flux-dev":
        warnings.append(
            f"{bm} is a large model; a 48 GB GPU is recommended (24 GB may work with "
            "blocks_to_swap / transformer_dtype=float8). The default smoke path is sdxl on 24 GB."
        )

    values.pop("frames", None)

    if errors:
        raise ConfigError(errors)
    return ResolvedConfig(values=values, sources=sources, warnings=warnings)
