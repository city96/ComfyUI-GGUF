import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import gguf
import torch


def load_runtime_modules():
    root = Path(__file__).parents[1]
    package_name = "comfyui_gguf_kquant_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package

    loaded = {}
    for module_name in ("quant_ops", "ops"):
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.{module_name}", root / f"{module_name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[module_name] = module
    loaded["backend"] = sys.modules[f"{package_name}.kquant_backend"]
    return loaded


class FakeKQuantBackend:
    supports_dequant_dtype = True

    def __init__(self, fail=False, should_use=True):
        self.fail = fail
        self._should_use = should_use
        self.calls = []

    def supports(self, **kwargs):
        self.calls.append(("supports", kwargs))
        return True

    def should_use(self, **kwargs):
        self.calls.append(("should_use", kwargs))
        return self._should_use

    def linear(self, **kwargs):
        self.calls.append(("linear", kwargs))
        if self.fail:
            raise RuntimeError("synthetic kernel failure")
        weight = torch.ones(
            kwargs["weight_shape"],
            dtype=kwargs["input"].dtype,
            device=kwargs["input"].device,
        )
        return torch.nn.functional.linear(kwargs["input"], weight, kwargs["bias"])


class KQuantBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = load_runtime_modules()
        cls.ops = modules["ops"]
        cls.quant_ops = modules["quant_ops"]
        cls.backend_module = modules["backend"]

    def tearDown(self):
        self.backend_module.register_kquant_backend(None)

    @staticmethod
    def quantized_storage(qtype, out_features=2):
        _, type_size = gguf.GGML_QUANT_SIZES[qtype]
        return torch.zeros(out_features * type_size, dtype=torch.uint8)

    def static_linear(self, qtype):
        raw = self.quantized_storage(qtype)
        weight = self.ops.GGMLTensor(
            raw,
            tensor_type=qtype,
            tensor_shape=(2, 256),
            compute_dtype=torch.float32,
        )
        linear = self.ops.GGMLOps.Linear(
            256, 2, bias=False, dtype=torch.float32
        )
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        return linear, raw

    def test_static_linear_sends_all_supported_k_quants_directly_to_backend(self):
        for qtype in (
            gguf.GGMLQuantizationType.Q3_K,
            gguf.GGMLQuantizationType.Q4_K,
            gguf.GGMLQuantizationType.Q5_K,
            gguf.GGMLQuantizationType.Q6_K,
        ):
            with self.subTest(qtype=qtype.name):
                backend = FakeKQuantBackend()
                self.backend_module.register_kquant_backend(backend)
                linear, raw = self.static_linear(qtype)
                input_tensor = torch.ones((2, 3, 256), dtype=torch.float32)

                output = linear.forward_ggml_cast_weights(input_tensor)

                self.assertEqual(tuple(output.shape), (2, 3, 2))
                self.assertTrue(torch.equal(output, torch.full_like(output, 256)))
                call = next(value for name, value in backend.calls if name == "linear")
                self.assertEqual(call["qtype"], qtype.name)
                self.assertEqual(call["weight_shape"], (2, 256))
                self.assertEqual(call["dequant_dtype"], torch.float16)
                self.assertEqual(call["weight"].untyped_storage().nbytes(), raw.numel())

    def test_static_linear_forwards_its_decode_precision(self):
        for setting, expected in (
            (None, torch.float16),
            ("target", torch.float32),
            (torch.bfloat16, torch.bfloat16),
            (torch.float32, torch.float32),
        ):
            with self.subTest(setting=setting):
                backend = FakeKQuantBackend()
                self.backend_module.register_kquant_backend(backend)
                linear, _ = self.static_linear(gguf.GGMLQuantizationType.Q5_K)
                linear.dequant_dtype = setting
                linear.forward_ggml_cast_weights(torch.ones((1, 256)))
                call = next(value for name, value in backend.calls if name == "linear")
                self.assertEqual(call["dequant_dtype"], expected)

    def test_old_backend_declines_different_decode_precision_without_disabling_route(self):
        backend = FakeKQuantBackend()
        backend.supports_dequant_dtype = False
        self.backend_module.register_kquant_backend(backend)
        raw = self.quantized_storage(gguf.GGMLQuantizationType.Q5_K)
        input_tensor = torch.ones((1, 256))
        self.assertIsNone(self.backend_module.try_kquant_linear(
            input_tensor, raw, gguf.GGMLQuantizationType.Q5_K, (2, 256),
            dequant_dtype=None,
        ))
        self.assertEqual(backend.calls, [])
        output = self.backend_module.try_kquant_linear(
            input_tensor, raw, gguf.GGMLQuantizationType.Q5_K, (2, 256)
        )
        self.assertTrue(torch.equal(output, torch.full_like(output, 256)))
        self.assertNotIn("dequant_dtype", backend.calls[-1][1])

    def test_backend_shape_decline_uses_portable_fallback_without_disabling_route(self):
        backend = FakeKQuantBackend(should_use=False)
        self.backend_module.register_kquant_backend(backend)
        linear, _ = self.static_linear(gguf.GGMLQuantizationType.Q4_K)
        input_tensor = torch.ones((1, 256), dtype=torch.float32)

        first = linear.forward_ggml_cast_weights(input_tensor)
        second = linear.forward_ggml_cast_weights(input_tensor)

        self.assertTrue(torch.equal(first, torch.zeros_like(first)))
        self.assertTrue(torch.equal(second, torch.zeros_like(second)))
        self.assertEqual(
            sum(name == "should_use" for name, _ in backend.calls),
            2,
        )
        self.assertFalse(any(name == "linear" for name, _ in backend.calls))

    def test_backend_failure_is_disabled_and_uses_dequantized_fallback(self):
        backend = FakeKQuantBackend(fail=True)
        self.backend_module.register_kquant_backend(backend)
        linear, _ = self.static_linear(gguf.GGMLQuantizationType.Q4_K)
        input_tensor = torch.ones((1, 256), dtype=torch.float32)

        first = linear.forward_ggml_cast_weights(input_tensor)
        second = linear.forward_ggml_cast_weights(input_tensor)

        self.assertTrue(torch.equal(first, torch.zeros_like(first)))
        self.assertTrue(torch.equal(second, torch.zeros_like(second)))
        self.assertEqual(
            sum(name == "linear" for name, _ in backend.calls),
            1,
        )

    def test_dynamic_layout_dispatches_linear_and_transposed_mm_without_dequant(self):
        backend = FakeKQuantBackend()
        self.backend_module.register_kquant_backend(backend)
        qtype = gguf.GGMLQuantizationType.Q5_K
        weight = self.quant_ops.make_quantized(
            self.quantized_storage(qtype),
            qtype,
            (2, 256),
            orig_dtype=torch.float32,
        )
        input_tensor = torch.ones((3, 256), dtype=torch.float32)
        bias = torch.tensor([1.0, 2.0])

        linear_output = torch.nn.functional.linear(input_tensor, weight, bias)
        mm_output = torch.mm(input_tensor, weight.t())

        self.assertTrue(
            torch.equal(
                linear_output,
                torch.tensor([[257.0, 258.0]]).expand(3, 2),
            )
        )
        self.assertTrue(torch.equal(mm_output, torch.full((3, 2), 256.0)))
        self.assertEqual(
            sum(name == "linear" for name, _ in backend.calls),
            2,
        )
        for name, call in backend.calls:
            if name == "linear":
                self.assertEqual(call["dequant_dtype"], torch.float16)

    def test_dynamic_layout_does_not_change_mismatched_weight_dtype(self):
        backend = FakeKQuantBackend()
        self.backend_module.register_kquant_backend(backend)
        qtype = gguf.GGMLQuantizationType.Q5_K
        weight = self.quant_ops.make_quantized(
            self.quantized_storage(qtype), qtype, (2, 256),
            orig_dtype=torch.float16,
        )
        with self.assertRaises(RuntimeError):
            torch.nn.functional.linear(torch.ones((1, 256)), weight)
        self.assertEqual(backend.calls, [])

    def test_default_cpu_linear_imports_backend_then_uses_portable_fallback(self):
        self.backend_module.register_kquant_backend(None)
        linear, _ = self.static_linear(gguf.GGMLQuantizationType.Q6_K)
        input_tensor = torch.ones((1, 256), dtype=torch.float32)

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.object(
                self.backend_module.importlib,
                "import_module",
                wraps=self.backend_module.importlib.import_module,
            ) as import_module:
                output = linear.forward_ggml_cast_weights(input_tensor)

        import_module.assert_called_once_with(
            self.backend_module._LOCAL_BACKEND_MODULE,
            package=self.backend_module.__package__,
        )
        self.assertEqual(output.device.type, "cpu")
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_explicit_opt_out_backend_uses_cpu_dequantized_fallback(self):
        self.backend_module.register_kquant_backend(None)
        qtype = gguf.GGMLQuantizationType.Q6_K
        weight = self.quant_ops.make_quantized(
            self.quantized_storage(qtype),
            qtype,
            (2, 256),
            orig_dtype=torch.float32,
        )
        input_tensor = torch.ones((1, 256), dtype=torch.float32)

        with mock.patch.dict("os.environ", {self.backend_module._BACKEND_ENV: "none"}):
            with mock.patch.object(
                self.backend_module.importlib, "import_module"
            ) as import_module:
                output = torch.nn.functional.linear(input_tensor, weight)

        import_module.assert_not_called()
        self.assertEqual(output.device.type, "cpu")
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_static_linear_uses_portable_fallback_when_backend_is_opted_out(self):
        self.backend_module.register_kquant_backend(None)
        linear, _ = self.static_linear(gguf.GGMLQuantizationType.Q5_K)
        input_tensor = torch.ones((1, 256), dtype=torch.float32)

        with mock.patch.dict(
            "os.environ", {self.backend_module._BACKEND_ENV: "none"}
        ):
            with mock.patch.object(
                self.backend_module.importlib, "import_module"
            ) as import_module:
                output = linear.forward_ggml_cast_weights(input_tensor)

        import_module.assert_not_called()
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_backend_can_be_opted_out_with_explicit_off_values(self):
        for setting in ("none", "off", "disabled", "false"):
            with self.subTest(setting=setting):
                self.backend_module.register_kquant_backend(None)
                with mock.patch.dict(
                    "os.environ", {self.backend_module._BACKEND_ENV: setting}
                ):
                    with mock.patch.object(
                        self.backend_module.importlib, "import_module"
                    ) as import_module:
                        self.assertIsNone(self.backend_module._load_backend())
                import_module.assert_not_called()

    def test_default_and_explicit_bundled_selection(self):
        backend = FakeKQuantBackend()
        module = types.SimpleNamespace(
            __name__="test_bundled_backend", backend=backend
        )

        for environment in (
            {},
            {self.backend_module._BACKEND_ENV: ""},
            {self.backend_module._BACKEND_ENV: "bundled"},
        ):
            with self.subTest(
                setting=environment.get(self.backend_module._BACKEND_ENV, "unset")
            ):
                self.backend_module.register_kquant_backend(None)
                with mock.patch.dict("os.environ", environment, clear=True):
                    with mock.patch.object(
                        self.backend_module.importlib,
                        "import_module",
                        return_value=module,
                    ) as import_module:
                        self.assertIs(backend, self.backend_module._load_backend())

                import_module.assert_called_once_with(
                    self.backend_module._LOCAL_BACKEND_MODULE,
                    package=self.backend_module.__package__,
                )

    def test_invalid_k_quant_storage_never_reaches_backend(self):
        backend = FakeKQuantBackend()
        self.backend_module.register_kquant_backend(backend)
        raw = torch.zeros((2, 143), dtype=torch.uint8)

        output = self.backend_module.try_kquant_linear(
            torch.ones((1, 256)),
            raw,
            gguf.GGMLQuantizationType.Q4_K,
            (2, 256),
        )

        self.assertIsNone(output)
        self.assertEqual(backend.calls, [])

    @mock.patch("torch.cuda.get_device_capability", return_value=(8, 6))
    def test_bundled_triton_launch_config_respects_device_dtype_and_row_limits(self, capability):
        module = importlib.import_module(
            ".kquant_triton_backend", package=self.backend_module.__package__
        )
        device = torch.device("cuda")
        cases = (
            ("Q3_K", (6144, 6144), 4096, (256, 4)),
            ("Q3_K", (6144, 6144), 8192, (256, 4)),
            ("Q3_K", (6144, 6144), 8193, None),
            ("Q5_K", (16384, 4096), 16, (256, 4)),
            ("Q5_K", (16384, 4096), 64, (256, 4)),
            ("Q5_K", (16384, 4096), 128, (256, 4)),
            ("Q5_K", (16384, 4096), 256, (256, 4)),
            ("Q5_K", (16384, 4096), 1024, (256, 4)),
            ("Q5_K", (16384, 4096), 1025, None),
            ("Q5_K", (4096, 4096), 512, (256, 4)),
            ("Q5_K", (4096, 4096), 1024, (256, 4)),
            ("Q6_K", (4096, 16384), 512, (256, 4)),
            ("Q6_K", (4096, 16384), 1024, (256, 4)),
            ("Q6_K", (4096, 16384), 1025, None),
            ("Q4_K", (16384, 4096), 128, None),
        )
        for qtype, weight_shape, m_size, expected in cases:
            with self.subTest(qtype=qtype, weight_shape=weight_shape, m_size=m_size):
                config = module._select_launch_config(
                    qtype=qtype,
                    input_dtype=torch.bfloat16,
                    input_shape=(m_size, weight_shape[1]),
                    weight_shape=weight_shape,
                    device=device,
                )
                self.assertEqual(config, expected)

        self.assertIsNone(
            module._select_launch_config(
                qtype="Q5_K",
                input_dtype=torch.float32,
                input_shape=(128, 4096),
                weight_shape=(4096, 4096),
                device=device,
            )
        )
        self.assertIsNone(
            module._select_launch_config(
                qtype="Q5_K",
                input_dtype=torch.bfloat16,
                input_shape=(128, 4096),
                weight_shape=(4096, 4096),
                device=torch.device("cpu"),
            )
        )
        capability.return_value = (12, 0)
        self.assertIsNone(module._select_launch_config(
            qtype="Q5_K", input_dtype=torch.bfloat16,
            input_shape=(128, 4096), weight_shape=(4096, 4096), device=device,
        ))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the bundled Triton backend")
    @mock.patch.dict("os.environ", {"COMFYUI_GGUF_KQUANT_BACKEND": "bundled"})
    def test_bundled_triton_backend_decodes_k_quant_weights_exactly(self):
        self.backend_module.register_kquant_backend(None)
        module = importlib.import_module(
            ".kquant_triton_backend", package=self.backend_module.__package__
        )
        from dequant import dequantize

        torch.manual_seed(26)
        for qtype in (
            gguf.GGMLQuantizationType.Q3_K,
            gguf.GGMLQuantizationType.Q4_K,
            gguf.GGMLQuantizationType.Q5_K,
            gguf.GGMLQuantizationType.Q6_K,
        ):
            with self.subTest(qtype=qtype.name):
                raw = self.random_k_storage(qtype, (3, 512))

                for input_dtype in (torch.float16, torch.bfloat16, torch.float32):
                    with self.subTest(input_dtype=input_dtype):
                        columns = torch.randint(512, (32,), device="cuda")
                        input_tensor = torch.zeros(
                            (32, 512), device="cuda", dtype=input_dtype
                        )
                        input_tensor[
                            torch.arange(32, device="cuda"), columns
                        ] = 1
                        with mock.patch.object(
                            module,
                            "_select_launch_config",
                            return_value=(256, 4),
                        ):
                            output = self.backend_module.try_kquant_linear(
                                input_tensor, raw, qtype, (3, 512)
                            )
                        reference_weight = dequantize(
                            raw.cpu(), qtype, (3, 512), dtype=input_dtype
                        ).cuda()
                        reference = reference_weight[:, columns].T.contiguous()

                        self.assertTrue(torch.equal(output, reference))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the bundled Triton backend")
    @mock.patch.dict("os.environ", {"COMFYUI_GGUF_KQUANT_BACKEND": "bundled"})
    def test_bundled_q6_triton_backend_large_one_hot_decode_is_stable(self):
        self.backend_module.register_kquant_backend(None)
        module = importlib.import_module(
            ".kquant_triton_backend", package=self.backend_module.__package__
        )
        from dequant import dequantize

        torch.manual_seed(2619)
        qtype = gguf.GGMLQuantizationType.Q6_K
        n_size, k_size, m_size = 2560, 9728, 224
        _, type_size = gguf.GGML_QUANT_SIZES[qtype]
        raw = torch.zeros(
            (n_size, k_size // 256 * type_size), device="cuda", dtype=torch.uint8
        )
        for offset in range(0, raw.shape[1], type_size):
            raw[:, offset : offset + 128] = torch.randint(
                0, 256, (n_size, 128), device="cuda", dtype=torch.uint8
            )
            raw[:, offset + 128 : offset + 192] = torch.randint(
                0, 256, (n_size, 64), device="cuda", dtype=torch.uint8
            )
            raw[:, offset + 192 : offset + 208] = torch.randint(
                -4,
                5,
                (n_size, 16),
                device="cuda",
                dtype=torch.int16,
            ).to(torch.uint8)
            raw[:, offset + 208] = 0
            raw[:, offset + 209] = 56

        for input_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(input_dtype=input_dtype):
                columns = torch.randint(k_size, (m_size,), device="cuda")
                input_tensor = torch.zeros(
                    (m_size, k_size), device="cuda", dtype=input_dtype
                )
                input_tensor[torch.arange(m_size, device="cuda"), columns] = 1
                reference_weight = dequantize(
                    raw.cpu(), qtype, (n_size, k_size), dtype=input_dtype
                ).cuda()
                reference = reference_weight[:, columns].T.contiguous()

                with mock.patch.object(
                    module,
                    "_select_launch_config",
                    return_value=(256, 4),
                ):
                    for _ in range(5):
                        output = self.backend_module.try_kquant_linear(
                            input_tensor, raw, qtype, (n_size, k_size)
                        )
                        self.assertTrue(torch.equal(output, reference))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the bundled Triton backend")
    @mock.patch.dict("os.environ", {"COMFYUI_GGUF_KQUANT_BACKEND": "bundled"})
    def test_bundled_triton_backend_matches_reference(self):
        self.backend_module.register_kquant_backend(None)
        module = importlib.import_module(
            ".kquant_triton_backend", package=self.backend_module.__package__
        )
        from dequant import dequantize

        for qtype in (
            gguf.GGMLQuantizationType.Q4_K,
            gguf.GGMLQuantizationType.Q5_K,
            gguf.GGMLQuantizationType.Q6_K,
        ):
            with self.subTest(qtype=qtype.name):
                _, type_size = gguf.GGML_QUANT_SIZES[qtype]
                raw = torch.randint(
                    0, 256, (3, 2 * type_size), device="cuda", dtype=torch.uint8
                )
                for offset in (0, type_size):
                    if qtype == gguf.GGMLQuantizationType.Q6_K:
                        raw[:, offset + 192 : offset + 208] = torch.randint(
                            0, 10, (3, 16), device="cuda", dtype=torch.uint8
                        )
                        raw[:, offset + 208] = 0
                        raw[:, offset + 209] = 60
                    else:
                        raw[:, offset] = 0
                        raw[:, offset + 1] = 60
                        raw[:, offset + 2 : offset + 4] = 0
                        raw[:, offset + 4 : offset + 16] = torch.randint(
                            0, 64, (3, 12), device="cuda", dtype=torch.uint8
                        )
                input_tensor = torch.randn(
                    2, 5, 512, device="cuda", dtype=torch.float16
                )
                bias = torch.randn(3, device="cuda", dtype=input_tensor.dtype)
                with mock.patch.object(
                    module,
                    "_select_launch_config",
                    return_value=(256, 4),
                ):
                    output = self.backend_module.try_kquant_linear(
                        input_tensor, raw, qtype, (3, 512), bias=bias
                    )
                reference_weight = dequantize(
                    raw.cpu(), qtype, (3, 512), dtype=input_tensor.dtype
                ).cuda()
                reference = torch.nn.functional.linear(
                    input_tensor, reference_weight, bias
                )
                self.assertTrue(torch.equal(output, reference))

    @staticmethod
    def random_k_storage(qtype, shape):
        n_size, k_size = shape
        _, type_size = gguf.GGML_QUANT_SIZES[qtype]
        raw = torch.randint(
            0, 256, (n_size, k_size // 256 * type_size),
            device="cuda", dtype=torch.uint8,
        )
        blocks = raw.reshape(-1, type_size)
        if qtype == gguf.GGMLQuantizationType.Q6_K:
            d_offset = 208
        elif qtype == gguf.GGMLQuantizationType.Q3_K:
            d_offset = 108
        else:
            d_offset = 0
        scales = torch.linspace(
            0.000171, 0.00137, blocks.shape[0], device="cuda", dtype=torch.float32
        ).to(torch.float16)
        if not torch.isfinite(scales).all():
            raise ValueError("Synthetic K-quant scales must be finite")
        blocks[:, d_offset:d_offset + 2] = scales.view(torch.uint8).reshape(-1, 2)
        if qtype in (
            gguf.GGMLQuantizationType.Q4_K,
            gguf.GGMLQuantizationType.Q5_K,
        ):
            minimum = (scales * 0.713).contiguous()
            blocks[:, 2:4] = minimum.view(torch.uint8).reshape(-1, 2)
        return raw

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the bundled Triton backend")
    @mock.patch.dict("os.environ", {"COMFYUI_GGUF_KQUANT_BACKEND": "bundled"})
    def test_bundled_decode_precision_bias_and_noncontiguous_inputs_are_exact(self):
        self.backend_module.register_kquant_backend(None)
        module = importlib.import_module(
            ".kquant_triton_backend", package=self.backend_module.__package__
        )
        from dequant import dequantize

        torch.manual_seed(2626)
        for qtype in (
            gguf.GGMLQuantizationType.Q3_K,
            gguf.GGMLQuantizationType.Q4_K,
            gguf.GGMLQuantizationType.Q5_K,
            gguf.GGMLQuantizationType.Q6_K,
        ):
            raw = self.random_k_storage(qtype, (5, 768))
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                x = torch.randn(2, 3, 1536, device="cuda", dtype=dtype)[..., ::2]
                bias = torch.randn(5, device="cuda", dtype=dtype)
                for decode_dtype in (None, "target", torch.float16, torch.bfloat16, torch.float32):
                    with self.subTest(qtype=qtype.name, dtype=dtype, decode_dtype=decode_dtype):
                        reference_weight = dequantize(
                            raw, qtype, (5, 768),
                            dtype=dtype if decode_dtype == "target" else decode_dtype,
                        ).to(dtype)
                        for b in (None, bias):
                            with mock.patch.object(
                                module, "_select_launch_config", return_value=(1024, 4)
                            ):
                                output = self.backend_module.try_kquant_linear(
                                    x, raw, qtype, (5, 768), b, dequant_dtype=decode_dtype
                                )
                            reference = torch.nn.functional.linear(x, reference_weight, b)
                            self.assertIsNotNone(output)
                            self.assertTrue(torch.equal(output, reference))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the bundled Triton backend")
    @mock.patch.dict("os.environ", {"COMFYUI_GGUF_KQUANT_BACKEND": "bundled"})
    def test_bundled_production_routes_match_static_and_dynamic_precision(self):
        self.backend_module.register_kquant_backend(None)
        if torch.cuda.get_device_capability() != (8, 6):
            self.skipTest("Production launch configs currently require SM 8.6")
        from dequant import dequantize

        torch.manual_seed(2609)
        for qtype, shape, m_sizes in (
            (gguf.GGMLQuantizationType.Q3_K, (5, 768), (4096, 8192)),
            (gguf.GGMLQuantizationType.Q5_K, (4096, 4096), (65, 257, 1024)),
            (gguf.GGMLQuantizationType.Q5_K, (16384, 4096), (65, 257, 1024)),
            (gguf.GGMLQuantizationType.Q6_K, (4096, 16384), (65, 257, 1024)),
        ):
            raw = self.random_k_storage(qtype, shape)
            for dtype in (torch.float16, torch.bfloat16):
                for decode_dtype in (None, torch.bfloat16, torch.float32):
                    reference_weight = dequantize(raw, qtype, shape, dtype=decode_dtype).to(dtype)
                    for m_size in m_sizes:
                        with self.subTest(qtype=qtype.name, shape=shape, dtype=dtype,
                                          decode_dtype=decode_dtype, m_size=m_size):
                            x = torch.randn(m_size, shape[1], device="cuda", dtype=dtype)
                            bias = torch.randn(shape[0], device="cuda", dtype=dtype)
                            output = self.backend_module.try_kquant_linear(
                                x, raw, qtype, shape, bias, dequant_dtype=decode_dtype
                            )
                            reference = torch.nn.functional.linear(x, reference_weight, bias)
                            self.assertIsNotNone(output)
                            self.assertTrue(torch.equal(output, reference))
            del raw

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the bundled Triton backend")
    @mock.patch.dict("os.environ", {"COMFYUI_GGUF_KQUANT_BACKEND": "bundled"})
    def test_bundled_static_and_dynamic_layers_match_their_portable_operations(self):
        self.backend_module.register_kquant_backend(None)
        module = importlib.import_module(
            ".kquant_triton_backend", package=self.backend_module.__package__
        )
        torch.manual_seed(2630)
        qtype = gguf.GGMLQuantizationType.Q5_K
        shape = (5, 768)
        raw = self.random_k_storage(qtype, shape)
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(65, shape[1], device="cuda", dtype=dtype)
            bias = torch.randn(shape[0], device="cuda", dtype=dtype)
            layer = self.ops.GGMLOps.Linear(shape[1], shape[0], dtype=dtype)
            layer.weight = torch.nn.Parameter(
                self.ops.GGMLTensor(
                    raw, tensor_type=qtype, tensor_shape=shape, compute_dtype=dtype
                ), requires_grad=False,
            )
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)
            for decode_dtype in (None, "target", torch.bfloat16, torch.float32):
                with self.subTest(dtype=dtype, decode_dtype=decode_dtype):
                    layer.dequant_dtype = decode_dtype
                    weight_fp, bias_fp = layer.cast_bias_weight(x)
                    reference = torch.nn.functional.linear(x, weight_fp, bias_fp)
                    with mock.patch.object(
                        module, "_select_launch_config", return_value=(256, 4)
                    ):
                        output = layer.forward_ggml_cast_weights(x)
                    self.assertTrue(torch.equal(output, reference))

            weight = self.quant_ops.make_quantized(raw, qtype, shape, orig_dtype=dtype)
            reference = torch.nn.functional.linear(x, weight.dequantize(), bias)
            with mock.patch.object(module, "_select_launch_config", return_value=(256, 4)):
                self.assertTrue(torch.equal(torch.nn.functional.linear(x, weight, bias), reference))
                self.assertTrue(torch.equal(
                    torch.mm(x, weight.t()), torch.mm(x, weight.dequantize().t())
                ))
                self.assertTrue(torch.equal(
                    torch.addmm(bias, x, weight.t()),
                    torch.addmm(bias, x, weight.dequantize().t()),
                ))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the bundled Triton backend")
    @mock.patch.dict("os.environ", {"COMFYUI_GGUF_KQUANT_BACKEND": "bundled"})
    def test_bundled_q3_static_and_dynamic_layers_match_portable_operations(self):
        self.backend_module.register_kquant_backend(None)
        torch.manual_seed(2631)
        qtype = gguf.GGMLQuantizationType.Q3_K
        shape = (5, 768)
        raw = self.random_k_storage(qtype, shape)
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(4096, shape[1], device="cuda", dtype=dtype)
            bias = torch.randn(shape[0], device="cuda", dtype=dtype)
            layer = self.ops.GGMLOps.Linear(shape[1], shape[0], dtype=dtype)
            layer.weight = torch.nn.Parameter(
                self.ops.GGMLTensor(
                    raw, tensor_type=qtype, tensor_shape=shape, compute_dtype=dtype
                ), requires_grad=False,
            )
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)
            for decode_dtype in (None, "target", torch.bfloat16, torch.float32):
                with self.subTest(dtype=dtype, decode_dtype=decode_dtype):
                    layer.dequant_dtype = decode_dtype
                    weight_fp, bias_fp = layer.cast_bias_weight(x)
                    reference = torch.nn.functional.linear(x, weight_fp, bias_fp)
                    output = layer.forward_ggml_cast_weights(x)
                    self.assertTrue(torch.equal(output, reference))

            weight = self.quant_ops.make_quantized(raw, qtype, shape, orig_dtype=dtype)
            reference_weight = weight.dequantize()
            reference = torch.nn.functional.linear(x, reference_weight, bias)
            self.assertTrue(torch.equal(torch.nn.functional.linear(x, weight, bias), reference))
            self.assertTrue(torch.equal(
                torch.mm(x, weight.t()), torch.mm(x, reference_weight.t())
            ))
            self.assertTrue(torch.equal(
                torch.addmm(bias, x, weight.t()), torch.addmm(bias, x, reference_weight.t())
            ))


if __name__ == "__main__":
    unittest.main()
