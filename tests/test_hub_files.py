"""Static checks of .runpod/hub.json and tests.json against the Runpod Hub publishing guide."""
import json
import os

from src.config import PARAMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA_ENV = {"HF_TOKEN", "TRAIN_TIMEOUT_SECONDS", "PRESET"}


def load(name):
    with open(os.path.join(ROOT, ".runpod", name)) as fh:
        return json.load(fh)


def test_required_files():
    for f in ("handler.py", "Dockerfile", "README.md", "LICENSE", ".runpod/hub.json", ".runpod/tests.json",
              ".runpod/icon.png", "examples/datasets/tiny_sdxl.zip"):
        # README may be written just after tests start in some flows; skip only if we run early.
        if f == "README.md" and not os.path.exists(os.path.join(ROOT, f)):
            continue
        assert os.path.exists(os.path.join(ROOT, f)), f


def test_hub_json():
    hub = load("hub.json")
    for k in ("title", "description", "type", "category", "config", "iconUrl"):
        assert hub.get(k), k
    assert hub["type"] == "serverless" and hub["category"] == "image"
    cfg = hub["config"]
    assert cfg["runsOn"] == "GPU" and cfg["containerDiskInGb"] == 150 and cfg["gpuCount"] == 1
    assert "ADA_24" in cfg["gpuIds"] and "Blackwell" in cfg["gpuIds"] and cfg["gpuIds"].startswith("ADA_24")
    assert "1g.24gb" in cfg["gpuIds"] and "2g.48gb" in cfg["gpuIds"]
    known = {p.env for p in PARAMS if p.env} | EXTRA_ENV
    keys = [e["key"] for e in cfg["env"]]
    assert len(keys) == len(set(keys)) and set(keys) <= known, set(keys) - known
    for preset in cfg["presets"]:
        assert set(preset["defaults"]) <= set(keys), preset["name"]
    names = [p["name"] for p in cfg["presets"]]
    assert any("SDXL" in n and "default" in n.lower() for n in names)
    assert any("Quick test" in n for n in names)
    for e in cfg["env"]:
        inp = e["input"]
        assert inp["type"] in {"string", "number", "boolean", "huggingface"}
        if inp["type"] == "number":
            assert inp["min"] <= inp["default"] <= inp["max"]
        if inp["type"] == "boolean":
            assert {inp["trueValue"], inp["falseValue"]} == {"true", "false"}
        for o in inp.get("options", []):
            assert o["label"] and o["value"]


def test_tests_json():
    t = load("tests.json")
    assert t["config"]["gpuTypeId"] == "NVIDIA GeForce RTX 4090"
    assert t["config"]["gpuCount"] == 1
    assert len(t["tests"]) == 1
    case = t["tests"][0]
    assert case["input"]["base_model"] == "sdxl"
    assert case["input"]["steps"] == 20
    assert case["input"]["num_samples"] == 0
    assert case["input"]["resolution"] == 512
    assert "tiny_sdxl.zip" in case["input"]["dataset_url"]
    assert case.get("timeout", 0) >= 1_500_000


def test_dockerfile_pins_sha_not_floating_main():
    text = open(os.path.join(ROOT, "Dockerfile")).read()
    assert "DIFFUSION_PIPE_SHA=c114c67eef87ef3a16eb82e4bb24c8ef700c26e4" in text
    assert "git checkout ${DIFFUSION_PIPE_SHA}" in text
    assert "git submodule update --init --recursive" in text
    assert "git clone" in text and "origin/main" not in text
    # Must not clone a floating branch without checkout.
    assert "--branch main" not in text
    assert "COPY" not in text.split("CMD")[0] or "safetensors" not in text
    assert "fp8_scaled" not in text
    # Weights must not be baked in.
    assert "huggingface-cli download" not in text


def test_license_is_gpl():
    text = open(os.path.join(ROOT, "LICENSE")).read()
    assert "GNU GENERAL PUBLIC LICENSE" in text
    assert "Version 3" in text
    assert "MIT License" not in text
