import pytest

from src.config import ConfigError, resolve_config


def test_sdxl_defaults():
    cfg = resolve_config({"dataset_url": "https://example.com/d.zip", "trigger_word": "zxowl"}, env={})
    assert cfg.base_model == "sdxl"
    assert cfg.steps == 2000 and cfg.rank == 16
    assert cfg.learning_rate == 2e-5
    assert cfg.resolution_wh == (1024, 1024)
    assert cfg.pipeline_stages == 1
    assert cfg.normalized_dtype == "bfloat16"
    assert cfg.lora_format == "kohya"
    assert cfg.pipe_type == "sdxl"
    assert cfg.frame_buckets == [1]


def test_quick_preset_and_resolution_aliases():
    cfg = resolve_config({
        "dataset_url": "https://example.com/d.zip", "preset": "quick",
        "trigger_word": "zxowl", "resolution": "512",
    }, env={})
    assert cfg.steps == 20 and cfg.rank == 8 and cfg.resolution_wh == (512, 512)


def test_rejects_tokens_in_input():
    with pytest.raises(ConfigError) as ei:
        resolve_config({"dataset_url": "https://x/y.zip", "hf_token": "should-not-be-here"}, env={})
    assert "do not send tokens" in str(ei.value)


def test_flux_requires_token_unless_dry_run():
    with pytest.raises(ConfigError) as ei:
        resolve_config({"dataset_url": "https://x/y.zip", "base_model": "flux-dev"}, env={})
    assert "gated" in str(ei.value).lower()
    cfg = resolve_config({"dataset_url": "https://x/y.zip", "base_model": "flux-dev", "dry_run": True}, env={})
    assert any("gated" in w.lower() or "HF_TOKEN" in w for w in cfg.warnings)


def test_coming_soon_is_explicit():
    with pytest.raises(ConfigError) as ei:
        resolve_config({"dataset_url": "https://x/y.zip", "base_model": "z-image"}, env={})
    assert "not in this worker's v1" in str(ei.value)


def test_blocks_incompatible_with_pipeline():
    with pytest.raises(ConfigError) as ei:
        resolve_config({
            "dataset_url": "https://x/y.zip", "pipeline_stages": 2, "blocks_to_swap": 8,
        }, env={})
    assert "incompatible" in str(ei.value)


def test_dtype_aliases_and_lr_from_env():
    cfg = resolve_config(
        {"dataset_url": "https://x/y.zip", "dtype": "bf16", "resolution": [832, 480]},
        env={"LEARNING_RATE": "4e-5", "STEPS": "100"},
    )
    assert cfg.normalized_dtype == "bfloat16"
    assert cfg.learning_rate == 4e-5 and cfg.steps == 100
    assert cfg.resolution_wh == (832, 480)


def test_unknown_field_and_missing_dataset():
    with pytest.raises(ConfigError) as ei:
        resolve_config({"foo": 1}, env={})
    msg = str(ei.value)
    assert "unknown input field" in msg and "dataset" in msg
