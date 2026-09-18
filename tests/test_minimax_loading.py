"""Run with PYTHONPATH=/path/to/ComfyUI python -m pytest -q tests."""

import importlib
import importlib.util
from pathlib import Path
import sys

import gguf
import pytest
import torch

from comfy.cli_args import args

args.cpu = True
spec = importlib.util.spec_from_file_location("comfyui_gguf", Path(__file__).resolve().parents[1] / "__init__.py")
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
loader = importlib.import_module("comfyui_gguf.loader")
convert = importlib.import_module("comfyui_gguf.tools.convert")


def test_minimax_architecture():
    keys = {"video_patch_proj.weight", "audio_patch_proj.weight", "final_layer.video_out.weight", "final_layer.audio_out.weight"}
    assert convert.detect_arch(keys).arch == "minimax_h3"
    assert "minimax_h3" in loader.IMG_ARCH_LIST
    assert not convert.is_model_arch(convert.ModelMiniMaxH3, keys - {"audio_patch_proj.weight"})


def test_qwen3_vision_mapping(tmp_path, monkeypatch):
    encoder = tmp_path / "Qwen3VL-32B-Thinking-Q4_K_M.gguf"
    projector = tmp_path / "Qwen3VL-32B-Thinking-mmproj-Q8_0.gguf"
    projector.touch()
    (tmp_path / "Qwen3VL-8B-Instruct-mmproj-Q8_0.gguf").touch()
    first = torch.ones(2, 3, 4, 4)
    second = first * 2
    weight = torch.randn(2, 2)
    sd = {
        "v.patch_embd.weight": first,
        "v.patch_embd.weight.1": second,
        "v.patch_embd.bias": weight,
        "v.position_embd.weight": weight,
        "v.blk.0.attn_qkv.weight": weight,
        "v.blk.0.attn_out.weight": weight,
        "v.blk.0.ffn_up.weight": weight,
        "v.blk.0.ffn_down.weight": weight,
        "v.blk.0.ln1.weight": weight,
        "v.blk.0.ln2.weight": weight,
        "v.post_ln.weight": weight,
        "mm.0.weight": weight,
        "mm.2.weight": weight,
    }
    for block in (24, 8, 16):
        for name in ("norm", "fc1", "fc2"):
            sd[f"v.deepstack.{block}.{name}.weight"] = torch.full((2, 2), float(block))

    def load(path, is_text_model):
        assert path == str(projector)
        assert is_text_model
        return sd, {}

    monkeypatch.setattr(loader, "gguf_sd_loader", load)
    result = loader.gguf_mmproj_loader(str(encoder))
    patch = result.pop("visual.patch_embed.proj.weight")
    torch.testing.assert_close(patch, torch.stack((first, second), dim=2))
    expected = {
        "visual.patch_embed.proj.bias", "visual.pos_embed.weight",
        "visual.blocks.0.attn.qkv.weight", "visual.blocks.0.attn.proj.weight",
        "visual.blocks.0.mlp.linear_fc1.weight", "visual.blocks.0.mlp.linear_fc2.weight",
        "visual.blocks.0.norm1.weight", "visual.blocks.0.norm2.weight",
        "visual.merger.norm.weight", "visual.merger.linear_fc1.weight", "visual.merger.linear_fc2.weight",
    }
    expected.update(f"visual.deepstack_merger_list.{i}.{name}.weight" for i in range(3) for name in ("norm", "linear_fc1", "linear_fc2"))
    assert set(result) == expected
    for name, value in result.items():
        if not name.startswith("visual.deepstack_merger_list."):
            assert value is weight
    for i, block in enumerate((8, 16, 24)):
        for name in ("norm", "linear_fc1", "linear_fc2"):
            torch.testing.assert_close(result[f"visual.deepstack_merger_list.{i}.{name}.weight"], torch.full((2, 2), float(block)))


@pytest.mark.parametrize("width,prefix", [(5120, ""), (4096, "model."), (2560, "model.")])
def test_qwen3_clip_includes_vision(monkeypatch, width, prefix):
    sd = {"blk.0.ffn_norm.weight": torch.empty(width)}
    vision = {"visual.deepstack_merger_list.0.norm.weight": torch.empty(4)}
    monkeypatch.setattr(loader, "gguf_sd_loader", lambda *a, **k: (sd, {"arch_str": "qwen3vl"}))
    monkeypatch.setattr(loader, "gguf_mmproj_loader", lambda path: vision)
    result = loader.gguf_clip_loader("qwen3vl.gguf")
    assert prefix + "visual.deepstack_merger_list.0.norm.weight" in result
    language_prefix = "model.language_model." if prefix else "model."
    assert language_prefix + "layers.0.post_attention_layernorm.weight" in result


def test_qwen3_missing_projector_is_an_error(monkeypatch):
    monkeypatch.setattr(loader, "gguf_sd_loader", lambda *a, **k: ({"blk.0.ffn_norm.weight": torch.empty(5120)}, {"arch_str": "qwen3vl"}))
    monkeypatch.setattr(loader, "gguf_mmproj_loader", lambda path: {})
    with pytest.raises(ValueError, match="Missing Qwen3-VL mmproj"):
        loader.gguf_clip_loader("qwen3vl.gguf")


def test_qwen2_vision_mapping_is_preserved(tmp_path, monkeypatch):
    (tmp_path / "Qwen2VL-mmproj.gguf").touch()
    weight = torch.randn(2, 2)
    sd = {"mm.0.weight": weight, "v.post_ln.weight": weight, "v.blk.0.ffn_up.weight": weight}
    monkeypatch.setattr(loader, "gguf_sd_loader", lambda *a, **k: (sd, {}))
    result = loader.gguf_mmproj_loader(str(tmp_path / "Qwen2VL-Q4_K_M.gguf"))
    assert set(result) == {"visual.merger.mlp.0.weight", "visual.merger.ln_q.weight", "visual.blocks.0.mlp.up_proj.weight"}


@pytest.mark.parametrize("width", [2560, 4096])
def test_smaller_qwen3_text_only_loading_is_preserved(monkeypatch, width):
    sd = {"blk.0.ffn_norm.weight": torch.empty(width)}
    monkeypatch.setattr(loader, "gguf_sd_loader", lambda *a, **k: (sd, {"arch_str": "qwen3vl"}))
    monkeypatch.setattr(loader, "gguf_mmproj_loader", lambda path: {})
    result = loader.gguf_clip_loader("qwen3vl.gguf")
    assert set(result) == {"model.layers.0.post_attention_layernorm.weight"}


@pytest.mark.parametrize("arch", [None, "minimax_h3"])
def test_minimax_gguf_loading(tmp_path, arch):
    path = tmp_path / "minimax.gguf"
    writer = gguf.GGUFWriter(path, arch="minimax_h3")
    if arch is None:
        writer.kv_data[0].pop("general.architecture")
    keys = ("video_patch_proj.weight", "audio_patch_proj.weight", "final_layer.video_out.weight", "final_layer.audio_out.weight")
    weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    for key in keys:
        writer.add_tensor(key, weight.numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sd, extra = loader.gguf_sd_loader(path)
    assert extra["arch_str"] == "minimax_h3"
    assert set(sd) == set(keys)
    for value in sd.values():
        torch.testing.assert_close(value.as_subclass(torch.Tensor), weight)
