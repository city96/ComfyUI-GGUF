import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import gguf
import torch


def load_patch_state():
    root = Path(__file__).parents[1]
    package_name = "comfyui_gguf_patch_state_test"
    package = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(package_name, loader=None)
    )
    package.__path__ = [str(root)]
    sys.modules[package_name] = package
    module_name = f"{package_name}.patch_state"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name, root / "patch_state.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def quantized_weight(shape=(2, 256), qtype=None):
    qtype = qtype or gguf.GGMLQuantizationType.Q5_K
    _, type_size = gguf.GGML_QUANT_SIZES[qtype]
    rows, cols = shape
    raw = torch.zeros(rows * (cols // 256) * type_size, dtype=torch.uint8)
    return SimpleNamespace(tensor_type=qtype, data=raw)


def plain_weight(shape=(4, 4)):
    return SimpleNamespace(
        tensor_type=gguf.GGMLQuantizationType.F16, data=torch.zeros(*shape)
    )


class PatchStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ps = load_patch_state()

    def make_patcher(self, patches, installed=False, uuid="uuid-1", current="uuid-1"):
        layer = SimpleNamespace(weight=quantized_weight(), bias=None)
        model = SimpleNamespace(
            attn=SimpleNamespace(q_proj=layer),
            current_weight_patches_uuid=current,
        )
        # Keys that do not resolve are ignored.
        model.plain = SimpleNamespace(weight=plain_weight(), bias=None)
        return SimpleNamespace(
            patches=patches,
            patches_uuid=uuid,
            model=model,
            _gguf_patches_installed=installed,
        )

    def test_no_patches_never_forces(self):
        patcher = self.make_patcher({})
        self.assertFalse(self.ps.requires_patch_reinstall(patcher))

    def test_plain_weights_never_force(self):
        patcher = self.make_patcher({"plain.weight": [("x", "k")]})
        self.assertFalse(self.ps.requires_patch_reinstall(patcher))

    def test_quantized_first_load_forces(self):
        patcher = self.make_patcher(
            {"attn.q_proj.weight": [("x", "k")]}, installed=False,
            uuid="uuid-1", current=None,
        )
        self.assertTrue(self.ps.requires_patch_reinstall(patcher))

    def test_quantized_changed_patches_force(self):
        patcher = self.make_patcher(
            {"attn.q_proj.weight": [("x", "k")]}, installed=True,
            uuid="uuid-2", current="uuid-1",
        )
        self.assertTrue(self.ps.requires_patch_reinstall(patcher))

    def test_quantized_steady_state_does_not_force(self):
        patcher = self.make_patcher(
            {"attn.q_proj.weight": [("x", "k")]}, installed=True,
            uuid="uuid-1", current="uuid-1",
        )
        self.assertFalse(self.ps.requires_patch_reinstall(patcher))

    def test_quantized_unpatched_state_forces(self):
        patcher = self.make_patcher(
            {"attn.q_proj.weight": [("x", "k")]}, installed=False,
            uuid="uuid-1", current="uuid-1",
        )
        self.assertTrue(self.ps.requires_patch_reinstall(patcher))

    def test_missing_model_is_conservative(self):
        patcher = SimpleNamespace(
            patches={"attn.q_proj.weight": [("x", "k")]},
            patches_uuid="uuid-1",
            model=None,
        )
        self.assertTrue(self.ps.requires_patch_reinstall(patcher))

    def test_unresolvable_keys_are_ignored(self):
        patcher = self.make_patcher({"nope.missing.weight": [("x", "k")]})
        self.assertFalse(self.ps.requires_patch_reinstall(patcher))


if __name__ == "__main__":
    unittest.main()
