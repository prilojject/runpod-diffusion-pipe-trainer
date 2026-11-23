import os

from src.config import resolve_config
from src.secret_utils import redact
from src.train import (auto_blocks_to_swap, build_train_command, find_outputs,
                       is_rtx_40xx, write_dataset_toml, write_train_toml)


def _cfg(**kw):
    base = {"dataset_url": "https://example.com/d.zip", "trigger_word": "zxowl", "steps": 20, "rank": 8,
            "resolution": 512, "preset": "quick"}
    base.update(kw)
    return resolve_config(base, env={})


def test_sdxl_toml_dtype_and_paths(tmp_path):
    cfg = _cfg()
    media = str(tmp_path / "images")
    os.makedirs(media)
    ds = str(tmp_path / "dataset.toml")
    tr = str(tmp_path / "train.toml")
    write_dataset_toml(cfg, [media], ds, 3)
    write_train_toml(cfg, {"checkpoint_path": "/models/sd_xl_base_1.0_0.9vae.safetensors"},
                     ds, str(tmp_path / "out"), tr, blocks=0)
    text = open(tr).read()
    ds_text = open(ds).read()
    assert "type = 'sdxl'" in text
    assert "dtype = 'bfloat16'" in text
    assert "save_dtype = 'bfloat16'" in text
    assert "rank = 8" in text
    assert "max_steps = 20" in text
    assert "enable_wandb = false" in text
    assert "pipeline_stages = 1" in text
    assert "checkpoint_path =" in text
    assert "alpha" not in text  # diffusion-pipe forbids alpha in the adapter table
    assert "fp8_scaled" not in text
    assert "skip_final_checkpoint = true" in text
    assert "skip_empty_caption = true" in ds_text
    assert media in ds_text
    assert "frame_buckets = [1]" in ds_text


def test_wan14_min_vram_and_nccl_on_4090(tmp_path):
    cfg = _cfg(base_model="wan-t2v-14b", preset="wan-t2v-14b")
    assert auto_blocks_to_swap(cfg, 23.5) == 32
    write_train_toml(cfg, {"ckpt_path": "/models/Wan2.1-T2V-14B"},
                     "/tmp/ds.toml", "/tmp/out", str(tmp_path / "train.toml"), blocks=32)
    text = open(tmp_path / "train.toml").read()
    assert "type = 'wan'" in text
    assert "transformer_dtype = 'float8'" in text
    assert "blocks_to_swap = 32" in text
    assert "AdamW8bitKahan" in text
    assert "activation_checkpointing = 'unsloth'" in text
    cmd, env = build_train_command(cfg, str(tmp_path / "train.toml"), gpu_count=1,
                                   gpu_name="NVIDIA GeForce RTX 4090")
    assert cmd[0] == "deepspeed" and "--num_gpus=1" in cmd and "--deepspeed" in cmd
    assert env["NCCL_P2P_DISABLE"] == "1" and env["NCCL_IB_DISABLE"] == "1"


def test_no_nccl_disable_on_a100():
    cfg = _cfg()
    cmd, env = build_train_command(cfg, "/tmp/t.toml", gpu_count=1, gpu_name="NVIDIA A100 80GB")
    assert "NCCL_P2P_DISABLE" not in env
    assert cmd[0] == "deepspeed"


def test_pipeline_stages_two_gpus():
    cfg = _cfg(pipeline_stages=2)
    cmd, env = build_train_command(cfg, "/tmp/t.toml", gpu_count=2, gpu_name="NVIDIA GeForce RTX 4090")
    assert "--num_gpus=2" in cmd
    assert env.get("NCCL_P2P_DISABLE") == "1"


def test_wan22_timestep_range(tmp_path):
    cfg = _cfg(base_model="wan22-t2v-a14b-low", preset="wan22-t2v-a14b-low")
    write_train_toml(cfg, {"ckpt_path": "/m/Wan2.2", "transformer_path": "/m/Wan2.2/low_noise_model"},
                     "/tmp/ds.toml", "/tmp/out", str(tmp_path / "t.toml"), blocks=32)
    text = open(tmp_path / "t.toml").read()
    assert "min_t = 0.0" in text and "max_t = 0.875" in text
    assert "transformer_path =" in text


def test_find_outputs_picks_latest_step(tmp_path):
    run = tmp_path / "20261123_01-02-03"
    (run / "step10").mkdir(parents=True)
    (run / "step20").mkdir()
    (run / "global_step20").mkdir()
    (run / "step10" / "lora.safetensors").write_bytes(b"a")
    (run / "step20" / "lora.safetensors").write_bytes(b"bbbb")
    (run / "global_step20" / "x.safetensors").write_bytes(b"nope")
    out = find_outputs(str(tmp_path))
    assert out["lora_path"].endswith("step20/lora.safetensors")


def test_redact_secrets():
    token = "hf_this_is_a_fake_token_value"
    os.environ["HF_TOKEN"] = token
    try:
        assert token not in redact(f"header Authorization: Bearer {token}")
        assert "***" in redact(token)
    finally:
        del os.environ["HF_TOKEN"]


def test_is_rtx_40xx():
    assert is_rtx_40xx("NVIDIA GeForce RTX 4090")
    assert not is_rtx_40xx("NVIDIA A100-SXM4-80GB")
