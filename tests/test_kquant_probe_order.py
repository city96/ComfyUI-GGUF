import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import gguf
import torch


def load_backend_module():
    root = Path(__file__).parents[1]
    package_name = "comfyui_gguf_probe_test"
    package = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(package_name, loader=None)
    )
    package.__path__ = [str(root)]
    sys.modules[package_name] = package
    module_name = f"{package_name}.kquant_backend"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name, root / "kquant_backend.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class RecordingBackend:
    supports_dequant_dtype = True

    def __init__(self):
        self.calls = []

    def supports(self, **kwargs):
        self.calls.append(("supports", kwargs))
        return True

    def should_use(self, **kwargs):
        self.calls.append(("should_use", kwargs))
        return True

    def linear(self, **kwargs):
        self.calls.append(("linear", kwargs))
        weight = torch.ones(
            kwargs["weight_shape"],
            dtype=kwargs["input"].dtype,
            device=kwargs["input"].device,
        )
        return torch.nn.functional.linear(kwargs["input"], weight, kwargs["bias"])


class KQuantProbeOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend_module = load_backend_module()

    def tearDown(self):
        self.backend_module.register_kquant_backend(None)
        self.backend_module._failed_routes.clear()

    @staticmethod
    def storage(qtype, out_features=2, in_features=256):
        _, type_size = gguf.GGML_QUANT_SIZES[qtype]
        return torch.zeros(
            out_features * (in_features // 256) * type_size, dtype=torch.uint8
        )

    def test_gates_use_shapes_and_dtypes_only(self):
        backend = RecordingBackend()
        self.backend_module.register_kquant_backend(backend)
        qtype = gguf.GGMLQuantizationType.Q5_K
        raw = self.storage(qtype)
        output = self.backend_module.try_kquant_linear(
            torch.ones((1, 256)), raw, qtype, (2, 256)
        )
        self.assertIsNotNone(output)
        supports = next(value for name, value in backend.calls if name == "supports")
        self.assertEqual(supports["weight_shape"], (2, 256))
        self.assertEqual(supports["weight_dtype"], torch.uint8)
        linear = next(value for name, value in backend.calls if name == "linear")
        self.assertEqual(tuple(linear["weight"].shape), (2, 176))
        self.assertTrue(linear["weight"].is_contiguous())
        self.assertEqual(linear["weight"].untyped_storage().nbytes(), raw.numel())

    def test_disabled_backend_allocates_no_repack(self):
        self.backend_module._backend = None
        self.backend_module._backend_checked = True
        qtype = gguf.GGMLQuantizationType.Q5_K
        raw = self.storage(qtype)
        try:
            with mock.patch.object(
                torch.Tensor, "reshape", autospec=True,
                side_effect=lambda self, *args, **kwargs: type(self).reshape(
                    self, *args, **kwargs
                ),
            ) as reshape, mock.patch.object(
                torch.Tensor, "contiguous", autospec=True,
                side_effect=lambda self, *args, **kwargs: type(self).contiguous(
                    self, *args, **kwargs
                ),
            ) as contiguous:
                result = self.backend_module.try_kquant_linear(
                    torch.ones((1, 256)), raw, qtype, (2, 256)
                )
        finally:
            self.backend_module._backend_checked = False
        self.assertIsNone(result)
        self.assertEqual(reshape.call_count, 0)
        self.assertEqual(contiguous.call_count, 0)


if __name__ == "__main__":
    unittest.main()
