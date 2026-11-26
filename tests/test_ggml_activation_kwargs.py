import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import gguf
import torch
import torch.nn.functional as F


# Regression tests for https://github.com/molbal/ComfyUI-GGUF/issues/27
#
# ComfyUI forwards activation-quantization kwargs (input_act, act_weight,
# residual, ...) through forward_comfy_cast_weights on every Linear call.
# The GGML path dequantizes weights eagerly and never uses them, so they
# must be filtered against each forward_ggml_cast_weights signature instead
# of raising TypeError.

def ops_module_factory():
    """Load the repo's ops.py as a package module for operation-level tests."""
    package_name = "comfyui_gguf_test"
    if package_name not in sys.modules:
        package = importlib.util.module_from_spec(
            importlib.util.spec_from_loader(package_name, loader=None)
        )
        package.__path__ = [str(Path(__file__).parents[1])]
        sys.modules[package_name] = package
    module_name = f"{package_name}.ops"
    if module_name in sys.modules:
        return sys.modules[module_name]
    ops_path = Path(__file__).parents[1] / "ops.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        ops_path,
        submodule_search_locations=[str(ops_path.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_q8_weight(ops, rows, cols, seed):
    rng = np.random.default_rng(seed)
    packed = gguf.quants.quantize(
        rng.standard_normal((rows, cols), dtype=np.float32),
        gguf.GGMLQuantizationType.Q8_0,
    ).copy()
    return ops.GGMLTensor(
        torch.from_numpy(packed),
        tensor_type=gguf.GGMLQuantizationType.Q8_0,
        tensor_shape=(rows, cols),
    )


# kwargs that recent ComfyUI versions pass into forward_comfy_cast_weights
# on every Linear call (None values: standard compute, no activation quant).
CORE_KWARGS = dict(
    input_act=None, act_weight=None, residual=None, residual_scale=None
)


class GGMLActivationKwargsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ops = ops_module_factory()

    def _make_q8_linear(self, in_features=64, out_features=16, seed=0):
        layer = self.ops.GGMLOps.Linear(
            in_features, out_features, bias=False, device="cpu", dtype=torch.float32
        )
        layer.load_state_dict({
            "weight": make_q8_weight(self.ops, out_features, in_features, seed)
        }, strict=False)
        return layer

    def _assert_close_to_dequantized(self, layer, x, out, atol=1e-3):
        dequantized = self.ops.dequantize_tensor(layer.weight, dtype=torch.float32)
        reference = F.linear(x, dequantized)
        self.assertTrue(
            torch.allclose(out, reference, atol=atol),
            f"max diff {(out - reference).abs().max().item()}",
        )

    def test_ggml_linear_accepts_core_activation_kwargs(self):
        layer = self._make_q8_linear(seed=0)
        x = torch.randn(2, 64, dtype=torch.float32)
        out = layer.forward_comfy_cast_weights(x, **CORE_KWARGS)
        self._assert_close_to_dequantized(layer, x, out)

    def test_ggml_linear_full_forward_accepts_core_activation_kwargs(self):
        layer = self._make_q8_linear(seed=1)
        x = torch.randn(3, 64, dtype=torch.float32)
        out = layer(x)
        self._assert_close_to_dequantized(layer, x, out)

    def test_ggml_linear_keeps_legacy_call_style(self):
        layer = self._make_q8_linear(seed=2)
        x = torch.randn(2, 64, dtype=torch.float32)
        out = layer.forward_comfy_cast_weights(x)
        self._assert_close_to_dequantized(layer, x, out)

    def test_ggml_embedding_keeps_out_dtype_kwarg(self):
        num_embeddings, embedding_dim = 32, 32
        embedding = self.ops.GGMLOps.Embedding(
            num_embeddings, embedding_dim, device="cpu", dtype=torch.float32
        )
        embedding.load_state_dict({
            "weight": make_q8_weight(
                self.ops, num_embeddings, embedding_dim, seed=3
            )
        }, strict=False)
        indices = torch.tensor([0, 5, 7, 31])
        out = embedding.forward_comfy_cast_weights(indices, out_dtype=torch.float16)
        dequantized = self.ops.dequantize_tensor(embedding.weight, dtype=torch.float16)
        reference = F.embedding(indices, dequantized)
        self.assertEqual(out.dtype, torch.float16)
        self.assertTrue(
            torch.allclose(out, reference, atol=1e-2),
            f"max diff {(out - reference).abs().max().item()}",
        )


if __name__ == "__main__":
    unittest.main()
