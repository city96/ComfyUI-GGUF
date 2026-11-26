import importlib.util
import sys
import unittest
from pathlib import Path

import gguf
import torch

from dequant import ggml_packed_nbytes


def _comfy_importable():
    try:
        import comfy.ops  # noqa: F401
        return True
    except Exception:
        return False


def ops_module_factory():
    """Load the repo's ops.py as a package module for operation-level tests."""
    package_name = "comfyui_gguf_state_size_test"
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


class PackedBytesTests(unittest.TestCase):
    def test_k_quant_storage_sizes(self):
        cases = {
            "Q3_K": (256, 110),
            "Q4_K": (256, 144),
            "Q5_K": (256, 176),
            "Q6_K": (256, 210),
        }
        for name, (block_size, type_size) in cases.items():
            with self.subTest(qtype=name):
                qtype = gguf.GGMLQuantizationType[name]
                self.assertEqual(ggml_packed_nbytes(qtype, (4096, 4096)), 65536 * type_size)
                self.assertEqual(ggml_packed_nbytes(qtype, (2, 256)), 2 * type_size)
                self.assertEqual(block_size, 256)

    def test_reporter_shapes_match_packed_reality(self):
        q5 = gguf.GGMLQuantizationType.Q5_K
        self.assertEqual(ggml_packed_nbytes(q5, (4096, 4096)), 65536 * 176)
        q6 = gguf.GGMLQuantizationType.Q6_K
        self.assertEqual(ggml_packed_nbytes(q6, (4096, 16384)), 262144 * 210)

    def test_legacy_and_iq_storage_sizes(self):
        self.assertEqual(
            ggml_packed_nbytes(gguf.GGMLQuantizationType.Q8_0, (16, 32)),
            (512 // 32) * 34,
        )
        self.assertEqual(
            ggml_packed_nbytes(gguf.GGMLQuantizationType.IQ4_XS, (2, 256)),
            2 * 136,
        )
        self.assertEqual(
            ggml_packed_nbytes(gguf.GGMLQuantizationType.F16, (4, 8)), 64
        )
        self.assertEqual(
            ggml_packed_nbytes(gguf.GGMLQuantizationType.BF16, (4, 8)), 64
        )

    def test_unknown_shapes_return_none(self):
        q5 = gguf.GGMLQuantizationType.Q5_K
        self.assertIsNone(ggml_packed_nbytes(None, (2, 256)))
        self.assertIsNone(ggml_packed_nbytes(q5, (3, 100)))
        self.assertIsNone(ggml_packed_nbytes(q5, (0, 256)))
        self.assertIsNone(ggml_packed_nbytes("Q5_K", (2, 256)))


@unittest.skipUnless(_comfy_importable(), "ComfyUI modules are unavailable")
class GGUFStateSizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ops = ops_module_factory()

    def _state_weight(self, qtype, shape):
        _, type_size = gguf.GGML_QUANT_SIZES[qtype]
        rows, cols = shape
        raw = torch.zeros(rows * (cols // 256) * type_size, dtype=torch.uint8)
        weight = self.ops.GGMLTensor(
            raw, tensor_type=qtype, tensor_shape=shape, compute_dtype=torch.float16
        )
        layer = self.ops.GGMLOps.Linear(cols, rows, bias=False, dtype=torch.float16)
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        state = layer.state_dict()
        return state["weight"], raw.numel()

    def test_state_dict_reports_packed_bytes(self):
        for qtype_name, shape in (
            ("Q5_K", (2, 256)),
            ("Q6_K", (2, 256)),
            ("Q4_K", (2, 256)),
            ("Q3_K", (2, 256)),
            ("Q8_0", (16, 32)),
            ("IQ4_XS", (2, 256)),
        ):
            with self.subTest(qtype=qtype_name):
                qtype = gguf.GGMLQuantizationType[qtype_name]
                fake, packed = self._state_weight(qtype, shape)
                self.assertEqual(fake.nbytes, packed)
                self.assertLess(fake.nbytes, fake.numel() * 2)

    def test_state_dict_matches_storage_for_reporter_shapes(self):
        fake, packed = self._state_weight(
            gguf.GGMLQuantizationType.Q5_K, (64, 4096)
        )
        self.assertEqual(fake.nbytes, packed)
        self.assertEqual(packed, 64 * 16 * 176)


if __name__ == "__main__":
    unittest.main()
