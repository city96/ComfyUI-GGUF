import unittest
import json
import gc
import os
import subprocess
import sys
from contextlib import ExitStack, nullcontext
from collections import OrderedDict
from types import SimpleNamespace
import importlib.util
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import gguf
import torch
import comfy.sd
import comfy.quant_ops
from safetensors.torch import save_file

import tools.convert as converter
from tools.convert import (
    MEBIBYTE,
    K_QUANT_PRESETS,
    K_QUANT_TYPE_NAMES,
    QUANT_TYPE_MAP,
    ModelLTXV,
    ModelLTXVUpsampler,
    ModelMinimaxH3,
    ModelMinimaxH3VAE,
    ModelMiniMaxMusic3DiT,
    ModelMiniMaxMusic3TextEncoder,
    ModelFlux3,
    ModelFibo2,
    ModelMOSSImage,
    ModelQwen3,
    ModelQwen3VL,
    ModelQwenImage21,
    ModelTemplate,
    normalize_qwen3vl_state_dict,
    normalize_source_state_dict,
    QWEN3VL_SOURCE_PREFIX,
    QWEN3VL_TEXT_ONLY_METADATA,
    QWEN3VL_VARIANT_METADATA,
    QWEN3VL_VISUAL_PREFIX,
    SourceStateDict,
    _streamed_safetensors_layout,
    convert_file,
    convert_state_dict,
    detect_arch,
    plan_k_quantization,
    plan_target_size_quantization,
    quantize_k,
    quantize_int8_convrot,
    quantize_int4_cr_w4a4,
    quantize_int1_cr_xl,
    resolve_quantization_device,
)
from dequant import dequantize, dequantize_functions, dequantize_tensor
from lora import (
    fuse_targets_into_state_dict,
    load_gguf_lora,
    load_lora,
    materialize_int8_source_weights,
    resolve_fusion_targets,
)


def load_gguf_loader():
    loader_path = Path(__file__).parents[1] / "loader.py"
    package_name = "comfyui_gguf_test"
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.loader",
        loader_path,
        submodule_search_locations=[str(loader_path.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[package_name] = module
    sys.modules[f"{package_name}.loader"] = module
    spec.loader.exec_module(module)
    return module


def ops_module_factory():
    """Load the repo's ops.py as a package module for operation-level tests."""
    import sys
    ops_path = Path(__file__).parents[1] / "ops.py"
    package_name = "comfyui_gguf_test"
    if package_name not in sys.modules:
        package = importlib.util.module_from_spec(
            importlib.util.spec_from_loader(package_name, loader=None)
        )
        package.__path__ = [str(ops_path.parent)]
        sys.modules[package_name] = package
    module_name = f"{package_name}.ops"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        ops_path,
        submodule_search_locations=[str(ops_path.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ops_factory(default_linear_dtype="int4"):
    """Load the repo's ops.py and return its ConvRot INT4 operations class."""
    return ops_module_factory().get_gguf_q4_w4a4_ops(
        torch.bfloat16,
        default_linear_dtype=default_linear_dtype,
    )


def nodes_factory():
    """Load the repo's nodes.py without requiring ComfyUI to load this plugin."""
    import sys

    nodes_path = Path(__file__).parents[1] / "nodes.py"
    repository_path = nodes_path.parent.resolve()
    original_sys_path = sys.path[:]
    sys.path = [path for path in sys.path if Path(path).resolve() != repository_path]
    package_name = "comfyui_gguf_nodes_test"
    try:
        package = sys.modules.get(package_name)
        if package is None:
            package = importlib.util.module_from_spec(
                importlib.util.spec_from_loader(package_name, loader=None)
            )
            package.__path__ = [str(nodes_path.parent)]
            sys.modules[package_name] = package
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.nodes",
            nodes_path,
            submodule_search_locations=[str(nodes_path.parent)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path = original_sys_path


class KQuantizationPlanTests(unittest.TestCase):
    def setUp(self):
        self.model_arch = ModelTemplate()
        self.model_arch.keys_hiprec = ["hiprec"]
        self.model_arch.keys_noquant = ["protected"]
        self.state_dict = OrderedDict((
            ("blocks.0.attn.to_v.weight", torch.ones((8, 256))),
            ("blocks.0.attn.to_out.0.weight", torch.ones((8, 256))),
            ("blocks.0.attn.qkv.weight", torch.ones((8, 256))),
            ("blocks.0.ff.net.2.weight", torch.ones((8, 256))),
            ("token_embedding.weight", torch.ones((8, 256))),
            ("blocks.0.other.weight", torch.ones((8, 256))),
            ("protected.weight", torch.ones((8, 256), dtype=torch.bfloat16)),
            ("hiprec.weight", torch.ones((8, 256))),
            ("conv.weight", torch.ones((8, 256), dtype=torch.bfloat16)),
            ("normalization.weight", torch.ones(256)),
        ))

    def test_public_k_selections_have_matching_file_types(self):
        expected = {
            "Q4_K": gguf.LlamaFileType.MOSTLY_Q4_K_S,
            "Q5_K": gguf.LlamaFileType.MOSTLY_Q5_K_S,
            "Q6_K": gguf.LlamaFileType.MOSTLY_Q6_K,
            "Q4_K_S": gguf.LlamaFileType.MOSTLY_Q4_K_S,
            "Q4_K_M": gguf.LlamaFileType.MOSTLY_Q4_K_M,
            "Q5_K_S": gguf.LlamaFileType.MOSTLY_Q5_K_S,
            "Q5_K_M": gguf.LlamaFileType.MOSTLY_Q5_K_M,
        }
        self.assertEqual(
            K_QUANT_TYPE_NAMES | K_QUANT_PRESETS,
            set(expected),
        )
        for name, file_type in expected.items():
            with self.subTest(name=name):
                self.assertEqual(QUANT_TYPE_MAP[name][1], file_type)

    def test_uniform_plans_preserve_architecture_and_linear_safety_rules(self):
        uniform_types = {
            "Q4_K": gguf.GGMLQuantizationType.Q4_K,
            "Q5_K": gguf.GGMLQuantizationType.Q5_K,
            "Q6_K": gguf.GGMLQuantizationType.Q6_K,
        }
        for name, qtype in uniform_types.items():
            with self.subTest(name=name):
                plan = plan_k_quantization(self.state_dict, self.model_arch, name)
                for key in (
                    "blocks.0.attn.to_v.weight",
                    "blocks.0.attn.to_out.0.weight",
                    "blocks.0.attn.qkv.weight",
                    "blocks.0.ff.net.2.weight",
                    "token_embedding.weight",
                    "blocks.0.other.weight",
                ):
                    self.assertEqual(plan[key], qtype)
                self.assertEqual(plan["protected.weight"], gguf.GGMLQuantizationType.BF16)
                self.assertEqual(plan["hiprec.weight"], gguf.GGMLQuantizationType.F32)
                self.assertEqual(plan["conv.weight"], gguf.GGMLQuantizationType.BF16)
                self.assertEqual(plan["normalization.weight"], gguf.GGMLQuantizationType.F32)

    def test_q4_k_m_promotes_sensitive_categories(self):
        plan = plan_k_quantization(self.state_dict, self.model_arch, "Q4_K_M")

        self.assertEqual(plan["blocks.0.attn.qkv.weight"], gguf.GGMLQuantizationType.Q5_K)
        for key in (
            "blocks.0.attn.to_v.weight",
            "blocks.0.attn.to_out.0.weight",
            "blocks.0.ff.net.2.weight",
            "token_embedding.weight",
        ):
            self.assertEqual(plan[key], gguf.GGMLQuantizationType.Q6_K)
        self.assertEqual(plan["blocks.0.other.weight"], gguf.GGMLQuantizationType.Q4_K)

    def test_small_and_medium_preset_assignments(self):
        q4_small = plan_k_quantization(self.state_dict, self.model_arch, "Q4_K_S")
        for key in (
            "blocks.0.attn.to_v.weight",
            "blocks.0.attn.to_out.0.weight",
            "blocks.0.ff.net.2.weight",
        ):
            self.assertEqual(q4_small[key], gguf.GGMLQuantizationType.Q5_K)
        self.assertEqual(q4_small["token_embedding.weight"], gguf.GGMLQuantizationType.Q6_K)
        self.assertEqual(q4_small["blocks.0.attn.qkv.weight"], gguf.GGMLQuantizationType.Q4_K)

        q5_small = plan_k_quantization(self.state_dict, self.model_arch, "Q5_K_S")
        self.assertEqual(q5_small["token_embedding.weight"], gguf.GGMLQuantizationType.Q6_K)
        self.assertEqual(q5_small["blocks.0.attn.to_v.weight"], gguf.GGMLQuantizationType.Q5_K)

        q5_medium = plan_k_quantization(self.state_dict, self.model_arch, "Q5_K_M")
        for key in (
            "blocks.0.attn.to_v.weight",
            "blocks.0.attn.to_out.0.weight",
            "blocks.0.attn.qkv.weight",
            "blocks.0.ff.net.2.weight",
            "token_embedding.weight",
        ):
            self.assertEqual(q5_medium[key], gguf.GGMLQuantizationType.Q6_K)
        self.assertEqual(q5_medium["blocks.0.other.weight"], gguf.GGMLQuantizationType.Q5_K)

    def test_mixed_plan_is_independent_of_checkpoint_order(self):
        forward = plan_k_quantization(self.state_dict, self.model_arch, "Q5_K_M")
        reverse = plan_k_quantization(
            OrderedDict(reversed(tuple(self.state_dict.items()))),
            self.model_arch,
            "Q5_K_M",
        )
        self.assertEqual(forward, reverse)

    def test_rejects_non_aligned_k_quant_rows(self):
        state_dict = {
            "blocks.0.weight": torch.ones((8, 384), dtype=torch.float32),
        }
        with self.assertRaisesRegex(
            ValueError,
            "blocks.0.weight.*384.*256-value K-quant block size",
        ):
            plan_k_quantization(state_dict, self.model_arch, "Q5_K")

    def test_shape_fix_models_can_rearrange_a_block_aligned_total(self):
        self.model_arch.shape_fix = True
        state_dict = {
            "blocks.0.weight": torch.ones((512, 4), dtype=torch.float32),
        }
        plan = plan_k_quantization(state_dict, self.model_arch, "Q6_K")
        self.assertEqual(plan["blocks.0.weight"], gguf.GGMLQuantizationType.Q6_K)

    def test_honors_established_wildcard_prefix_protections(self):
        model_arch = ModelTemplate()
        model_arch.keys_hiprec = ["^blocks.*.attn.qkv_proj."]
        plan = plan_k_quantization(
            {
                "blocks.0.attn.qkv_proj.weight": torch.ones((8, 256)),
                "blocks.0.mlp.fc1.weight": torch.ones((8, 256)),
            },
            model_arch,
            "Q4_K_M",
        )
        self.assertEqual(
            plan["blocks.0.attn.qkv_proj.weight"],
            gguf.GGMLQuantizationType.F32,
        )
        self.assertEqual(
            plan["blocks.0.mlp.fc1.weight"],
            gguf.GGMLQuantizationType.Q4_K,
        )

    def test_file_type_metadata_matches_every_k_selection(self):
        state_dict = {
            "double_blocks.0.img_attn.proj.weight": torch.ones((16, 16)),
        }
        for name in sorted(K_QUANT_TYPE_NAMES | K_QUANT_PRESETS):
            with self.subTest(name=name), TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / f"flux-{name}.gguf"
                converted_path, _ = convert_state_dict(
                    state_dict,
                    str(output_path),
                    quant_type_name=name,
                )
                reader = gguf.GGUFReader(converted_path)
                file_type = reader.get_field("general.file_type").contents()
                reader.tensors.clear()
                reader.fields.clear()
                reader.data._mmap.close()
                del reader
                self.assertEqual(file_type, QUANT_TYPE_MAP[name][1].value)

    def test_streamed_conversion_uses_same_k_selection_and_metadata(self):
        state_dict = {
            "double_blocks.0.img_attn.proj.weight": torch.ones((16, 16)),
        }
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "flux.safetensors"
            output_path = Path(temp_dir) / "flux-Q4_K_M.gguf"
            save_file(state_dict, str(source_path))
            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q4_K_M",
                streamed=True,
            )
            reader = gguf.GGUFReader(converted_path)
            file_type = reader.get_field("general.file_type").contents()
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader
        self.assertEqual(file_type, gguf.LlamaFileType.MOSTLY_Q4_K_M.value)

    def test_conversion_restores_convrot_weights_from_checkpoint_metadata(self):
        source = torch.tensor(
            [[2.0, 2.0, 2.0, 2.0], [2.0, -2.0, 2.0, -2.0]],
            dtype=torch.float32,
        )
        qdata, scale, quant_conf, _ = quantize_int8_convrot(
            source, convrot_groupsize=4, device=torch.device("cpu")
        )
        state_dict = {
            "blocks.0.attn.wq.weight": qdata,
            "blocks.0.attn.wq.weight_scale": scale,
            "txtfusion.projector.weight": torch.ones((2, 2), dtype=torch.float16),
            "first.weight": torch.ones((2, 2), dtype=torch.float16),
        }
        metadata = {
            "_quantization_metadata": json.dumps({
                "format": "int8_convrot",
                "layers": {"blocks.0.attn.wq": quant_conf},
            })
        }
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "krea2-convrot.safetensors"
            output_path = Path(temp_dir) / "krea2-convrot.gguf"
            save_file(state_dict, str(source_path), metadata=metadata)
            for streamed in (False, True):
                output_path = Path(temp_dir) / f"krea2-convrot-{streamed}.gguf"
                converted_path, _ = convert_file(
                    str(source_path),
                    str(output_path),
                    interact=False,
                    quant_type_name="F16",
                    streamed=streamed,
                )
                reader = gguf.GGUFReader(converted_path)
                tensor = next(
                    tensor for tensor in reader.tensors
                    if tensor.name == "blocks.0.attn.wq.weight"
                )
                actual = torch.from_numpy(tensor.data.copy()).view(torch.float16).reshape(
                    tuple(reversed(tensor.shape))
                )
                reader.tensors.clear()
                reader.fields.clear()
                reader.data._mmap.close()
                del reader
                with self.subTest(streamed=streamed):
                    self.assertTrue(torch.equal(actual, source.to(torch.float16)))

    def test_streamed_conversion_restores_scaled_int8_source_weights(self):
        state_dict = {
            "blocks.0.attn.wq.weight": torch.tensor(
                [[10, -20], [30, -40]], dtype=torch.int8
            ),
            "blocks.0.attn.wq.weight_scale": torch.tensor([0.1, 0.01]),
            "txtfusion.projector.weight": torch.ones((2, 2), dtype=torch.float16),
            "first.weight": torch.ones((2, 2), dtype=torch.float16),
        }
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "krea2-int8.safetensors"
            output_path = Path(temp_dir) / "krea2-f16.gguf"
            save_file(state_dict, str(source_path))
            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="F16",
                streamed=True,
            )
            reader = gguf.GGUFReader(converted_path)
            tensor = next(
                tensor for tensor in reader.tensors
                if tensor.name == "blocks.0.attn.wq.weight"
            )
            actual = torch.from_numpy(tensor.data.copy()).view(torch.float16).reshape(
                tuple(reversed(tensor.shape))
            )
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

        self.assertTrue(
            torch.equal(
                actual,
                torch.tensor([[1.0, -2.0], [0.3, -0.4]], dtype=torch.float16),
            )
        )

    def test_uniform_k_conversions_work_without_gguf_k_encoder(self):
        expected = {
            "Q4_K": gguf.GGMLQuantizationType.Q4_K,
            "Q5_K": gguf.GGMLQuantizationType.Q5_K,
            "Q6_K": gguf.GGMLQuantizationType.Q6_K,
        }
        for name, qtype in expected.items():
            with self.subTest(name=name), TemporaryDirectory() as temp_dir:
                state_dict = {
                    "double_blocks.0.img_attn.proj.weight": torch.ones((8, 256)),
                }
                output_path = Path(temp_dir) / f"flux-{name}.gguf"
                converted_path, _ = convert_state_dict(
                    state_dict,
                    str(output_path),
                    quant_type_name=name,
                )
                reader = gguf.GGUFReader(converted_path)
                tensor_type = reader.tensors[0].tensor_type
                reader.tensors.clear()
                reader.fields.clear()
                reader.data._mmap.close()
                del reader
                self.assertEqual(tensor_type, qtype)


    def test_local_k_encoders_round_trip_through_reference_dequantizer(self):
        source = (np.sin(np.linspace(0.0, 40.0, 512)) * 2.0).astype(np.float32).reshape(2, 256)
        tolerances = {
            gguf.GGMLQuantizationType.Q4_K: 0.20,
            gguf.GGMLQuantizationType.Q5_K: 0.10,
            gguf.GGMLQuantizationType.Q6_K: 0.06,
        }
        for qtype, tolerance in tolerances.items():
            with self.subTest(qtype=qtype.name):
                packed = quantize_k(source, qtype)
                decoded = dequantize(
                    torch.from_numpy(packed), qtype, source.shape, dtype=torch.float32
                ).numpy()
                self.assertEqual(
                    packed.shape[-1], gguf.GGML_QUANT_SIZES[qtype][1]
                )
                self.assertLess(float(np.max(np.abs(decoded - source))), tolerance)


class TargetSizeQuantizationTests(unittest.TestCase):
    def setUp(self):
        self.model_arch = ModelTemplate()
        self.state_dict = OrderedDict(
            (f"blocks.{index}.weight", torch.ones((4096, 32), dtype=torch.float32))
            for index in range(3)
        )
        self.state_dict["normalization.weight"] = torch.ones((4096,), dtype=torch.float32)

    def test_reduces_center_core_layers_before_outer_layers(self):
        plan, _, selected_size = plan_target_size_quantization(
            self.state_dict, self.model_arch, 0.4
        )

        self.assertEqual(plan["blocks.1.weight"], gguf.GGMLQuantizationType.Q5_0)
        self.assertEqual(plan["blocks.0.weight"], gguf.GGMLQuantizationType.I8)
        self.assertEqual(plan["blocks.2.weight"], gguf.GGMLQuantizationType.I8)
        self.assertLessEqual(selected_size, int(0.4 * MEBIBYTE))

    def test_reduces_one_dimensional_weights_only_after_all_core_layers(self):
        plan, _, selected_size = plan_target_size_quantization(
            self.state_dict, self.model_arch, 0.222
        )

        for index in range(3):
            self.assertEqual(plan[f"blocks.{index}.weight"], gguf.GGMLQuantizationType.Q4_0)
        self.assertEqual(plan["normalization.weight"], gguf.GGMLQuantizationType.BF16)
        self.assertLessEqual(selected_size, int(0.222 * MEBIBYTE))

    def test_supports_standard_q8_baseline(self):
        plan, _, _ = plan_target_size_quantization(
            self.state_dict,
            self.model_arch,
            2.0,
            target_size_q8_type="Q8_0",
        )

        for index in range(3):
            self.assertEqual(plan[f"blocks.{index}.weight"], gguf.GGMLQuantizationType.Q8_0)

    def test_reports_minimum_when_target_is_unsupported(self):
        with self.assertRaisesRegex(ValueError, "smallest supported TARGET_SIZE output"):
            plan_target_size_quantization(self.state_dict, self.model_arch, 0.1)


class Q8CRConversionDeviceTests(unittest.TestCase):
    def test_auto_uses_cpu_without_cuda(self):
        with mock.patch("tools.convert.torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_quantization_device("auto").type, "cpu")

    def test_cuda_requires_available_device(self):
        with mock.patch("tools.convert.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires an available CUDA device"):
                resolve_quantization_device("cuda")

    def test_cpu_quantization_stays_on_cpu(self):
        qdata, scale, quant_conf, _ = quantize_int8_convrot(
            torch.arange(256, dtype=torch.float32).reshape(1, 256),
            device=torch.device("cpu"),
        )

        self.assertEqual(qdata.device.type, "cpu")
        self.assertEqual(scale.device.type, "cpu")
        self.assertTrue(quant_conf["weight_rotated"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_quantization_has_cpu_equivalent_decode_error(self):
        torch.manual_seed(0)
        weight = torch.randn((32, 256), dtype=torch.float32)
        cpu_qdata, cpu_scale, _, _ = quantize_int8_convrot(weight, device=torch.device("cpu"))
        cuda_qdata, cuda_scale, _, _ = quantize_int8_convrot(weight, device=torch.device("cuda"))

        cpu_decoded = cpu_qdata.to(torch.float32) * cpu_scale
        cuda_decoded = cuda_qdata.cpu().to(torch.float32) * cuda_scale.cpu()
        self.assertTrue(torch.allclose(cpu_decoded, cuda_decoded, atol=1e-5, rtol=0))

    def test_auto_device_serializes_q8_cr_layout(self):
        state_dict = {
            "video_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "audio_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "blocks.0.attn.qkv_proj.weight": torch.ones((96, 32), dtype=torch.float32),
            "blocks.0.mlp.fc1.weight": torch.ones((128, 512), dtype=torch.float32),
            "final_layer.video_out.weight": torch.ones((96, 32), dtype=torch.float32),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            output_path = Path(temp_dir) / "minimax_h3-Q8_CR.gguf"
            save_file(state_dict, str(source_path))

            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q8_CR",
                quantization_device="auto",
            )

            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

        self.assertEqual(
            tensor_types["blocks.0.attn.qkv_proj.weight"],
            gguf.GGMLQuantizationType.F32,
        )
        self.assertEqual(
            tensor_types["blocks.0.mlp.fc1.weight"],
            gguf.GGMLQuantizationType.I8,
        )
        self.assertEqual(
            tensor_types["blocks.0.mlp.fc1.weight_scale"],
            gguf.GGMLQuantizationType.F32,
        )

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "PyTorch does not support FP8")
    def test_streamed_layout_supports_safetensors_f8_e4m3(self):
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "fp8.safetensors"
            save_file(
                {"weight": torch.ones((2, 2), dtype=torch.float8_e4m3fn)},
                str(source_path),
            )
            state_dict, source_keys = _streamed_safetensors_layout(str(source_path))

        self.assertEqual(state_dict["weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(source_keys["weight"], "weight")


class Flux3ConversionTests(unittest.TestCase):
    def _source_state_dict(self):
        return {
            "content_mode_blocks.image.0.attn_out.weight": torch.ones((32, 64), dtype=torch.bfloat16),
            "content_mode_blocks.image_cond.0.attn_out.weight": torch.ones((32, 64), dtype=torch.bfloat16),
            "single_stream_modulations.txt.lin.weight": torch.ones((96, 64), dtype=torch.bfloat16),
            "content_mode_blocks.image.0.norm.query_norm.scale": torch.ones(4, dtype=torch.bfloat16),
            "emb_in.image.weight": torch.ones((32, 128), dtype=torch.bfloat16),
            "emb_in.image_cond.weight": torch.ones((32, 128), dtype=torch.bfloat16),
        }

    def test_detects_flux3_and_comfyui_runtime_config(self):
        from comfy.model_detection import detect_unet_config

        state_dict = self._source_state_dict()
        self.assertIsInstance(detect_arch(state_dict), ModelFlux3)
        self.assertEqual(detect_unet_config(state_dict, "")["image_model"], "flux3")

    def test_k_quants_preserve_flux3_input_projections(self):
        state_dict = {
            "emb_in.image.weight": torch.ones((3072, 128), dtype=torch.bfloat16),
            "emb_in.image_cond.weight": torch.ones((3072, 128), dtype=torch.bfloat16),
        }
        for quant_type in ("Q4_K_M", "Q6_K"):
            plan = plan_k_quantization(state_dict, ModelFlux3(), quant_type)
            with self.subTest(quant_type=quant_type):
                self.assertEqual(
                    plan["emb_in.image.weight"], gguf.GGMLQuantizationType.BF16
                )
                self.assertEqual(
                    plan["emb_in.image_cond.weight"], gguf.GGMLQuantizationType.BF16
                )

    def test_q8_cr_conversion_loads_as_flux3_in_static_and_dynamic_modes(self):
        state_dict = self._source_state_dict()
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "flux3.safetensors"
            output_path = Path(temp_dir) / "flux3-Q8_CR.gguf"
            save_file(state_dict, str(source_path))

            converted_path, model_arch = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q8_CR",
                quantization_device="cpu",
            )

            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            architecture = reader.get_field("general.architecture").contents()
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

            loader = load_gguf_loader()
            loaded_architectures = []
            for dynamic in (False, True):
                loaded, extra = loader.gguf_sd_loader(
                    converted_path,
                    handle_prefix=None,
                    dynamic=dynamic,
                )
                loaded_architectures.append(extra["arch_str"])
                self.assertEqual(extra["gguf_quant_mode"], "int8_convrot")
                self.assertEqual(
                    tuple(loaded["content_mode_blocks.image.0.attn_out.weight"].shape),
                    (32, 64),
                )
                del loaded
                gc.collect()

        self.assertIsInstance(model_arch, ModelFlux3)
        self.assertEqual(architecture, "flux3")
        self.assertEqual(loaded_architectures, ["flux3", "flux3"])
        self.assertEqual(
            tensor_types["content_mode_blocks.image.0.attn_out.weight"],
            gguf.GGMLQuantizationType.I8,
        )
        self.assertEqual(
            tensor_types["content_mode_blocks.image.0.attn_out.weight_scale"],
            gguf.GGMLQuantizationType.F32,
        )
        self.assertEqual(
            tensor_types["content_mode_blocks.image.0.norm.query_norm.scale"],
            gguf.GGMLQuantizationType.F32,
        )

    def test_q4_cr_and_q4_cr_l_promote_sensitive_flux3_layers_and_unrotate_q8_cr(self):
        torch.manual_seed(7)
        orig_txt_in = torch.randn((16, 256), dtype=torch.bfloat16)
        orig_attn_out = torch.randn((16, 256), dtype=torch.bfloat16)
        state_dict = {
            "content_mode_blocks.image.0.attn_out.weight": orig_attn_out,
            "content_mode_blocks.image.0.v_proj.weight": torch.randn((16, 256), dtype=torch.bfloat16),
            "content_mode_blocks.image_cond.0.attn_out.weight": torch.randn((16, 256), dtype=torch.bfloat16),
            "txt_mode_blocks.0.attn_out.weight": torch.randn((16, 256), dtype=torch.bfloat16),
            "content_mode_blocks.image.0.norm.query_norm.scale": torch.ones(64, dtype=torch.bfloat16),
            "emb_in.0.weight": torch.ones((16, 128), dtype=torch.bfloat16),
            "time_in.in_layer.weight": torch.randn((16, 256), dtype=torch.bfloat16),
            "ref_time_in.in_layer.weight": torch.randn((16, 256), dtype=torch.bfloat16),
            "txt_in.weight": orig_txt_in,
            "early_stream_modulations.img.lin.weight": torch.randn((16, 256), dtype=torch.bfloat16),
            "single_stream_modulations.txt.lin.weight": torch.randn((16, 256), dtype=torch.bfloat16),
            "single_stream_blocks.0.linear1.weight": torch.randn((16, 256), dtype=torch.bfloat16),
            "final_layer.image.adaLN_modulation.1.weight": torch.randn((16, 256), dtype=torch.bfloat16),
            "final_layer.image.linear.weight": torch.randn((8, 256), dtype=torch.bfloat16),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "flux3.safetensors"
            q4_cr_path = Path(temp_dir) / "flux3-Q4_CR.gguf"
            q4_cr_l_path = Path(temp_dir) / "flux3-Q4_CR_L.gguf"
            save_file(state_dict, str(source_path))

            convert_file(
                str(source_path),
                str(q4_cr_path),
                interact=False,
                quant_type_name="Q4_CR",
                quantization_device="cpu",
            )
            convert_file(
                str(source_path),
                str(q4_cr_l_path),
                interact=False,
                quant_type_name="Q4_CR_L",
                quantization_device="cpu",
            )

            self.assertGreater(q4_cr_l_path.stat().st_size, q4_cr_path.stat().st_size)

            loader = load_gguf_loader()
            sd_cr, extra_cr = loader.gguf_sd_loader(str(q4_cr_path), dynamic=False)
            sd_crl, extra_crl = loader.gguf_sd_loader(str(q4_cr_l_path), dynamic=False)

            self.assertEqual(extra_cr["gguf_quant_mode"], "int4_cr_w4a4")
            self.assertEqual(extra_crl["gguf_quant_mode"], "int4_cr_w4a8")

            for sd in (sd_cr, sd_crl):
                self.assertEqual(
                    sd["content_mode_blocks.image.0.norm.query_norm.scale"].tensor_type,
                    gguf.GGMLQuantizationType.F32,
                )
                self.assertEqual(
                    sd["final_layer.image.linear.weight"].tensor_type,
                    gguf.GGMLQuantizationType.F32,
                )
                self.assertEqual(sd["emb_in.0.weight"].dtype, torch.bfloat16)
                txt_conf = json.loads(bytes(sd["txt_in.comfy_quant"].tolist()).decode("utf-8"))
                self.assertEqual(txt_conf["format"], "int8_tensorwise")

            img_attn_conf_cr = json.loads(
                bytes(sd_cr["content_mode_blocks.image.0.attn_out.comfy_quant"].tolist()).decode("utf-8")
            )
            v_conf_cr = json.loads(
                bytes(sd_cr["content_mode_blocks.image.0.v_proj.comfy_quant"].tolist()).decode("utf-8")
            )
            v_conf_crl = json.loads(
                bytes(sd_crl["content_mode_blocks.image.0.v_proj.comfy_quant"].tolist()).decode("utf-8")
            )
            txt_attn_conf_cr = json.loads(
                bytes(sd_cr["txt_mode_blocks.0.attn_out.comfy_quant"].tolist()).decode("utf-8")
            )
            txt_attn_conf_crl = json.loads(
                bytes(sd_crl["txt_mode_blocks.0.attn_out.comfy_quant"].tolist()).decode("utf-8")
            )
            self.assertEqual(img_attn_conf_cr["format"], "int8_tensorwise")
            self.assertEqual(v_conf_cr["format"], "int4_cr")
            self.assertEqual(v_conf_cr["backing"], "w4a4")
            self.assertEqual(v_conf_crl["format"], "int8_tensorwise")
            self.assertEqual(txt_attn_conf_cr["format"], "int4_cr")
            self.assertEqual(txt_attn_conf_crl["format"], "int8_tensorwise")

            # Verify GGUFQ4W4A4Ops.Linear un-rotates mixed int8_tensorwise (Q8_CR) layers on load
            ops = ops_factory(default_linear_dtype="int8")
            lin = ops.Linear(256, 16, bias=False)
            lin._load_from_state_dict(
                sd_crl,
                prefix="txt_in.",
                local_metadata={},
                strict=True,
                missing_keys=[],
                unexpected_keys=[],
                error_msgs=[],
            )
            rel_err = (lin.weight.float() - orig_txt_in.float()).norm() / orig_txt_in.float().norm()
            self.assertLess(rel_err.item(), 0.02)
            del sd, sd_cr, sd_crl, lin
            gc.collect()


class QwenImage21ConversionTests(unittest.TestCase):
    def test_detects_qwen_image_21_from_distinctive_dit_keys(self):
        state_dict = {
            "txt_in.text_norm.weight": torch.ones(4096, dtype=torch.bfloat16),
            "modulation.1.weight": torch.ones((16384, 4096), dtype=torch.bfloat16),
            "transformer_blocks.0.attn.norm_q.weight": torch.ones(128, dtype=torch.bfloat16),
            "img_in.weight": torch.ones((4096, 64), dtype=torch.bfloat16),
            "proj_out.weight": torch.ones((64, 4096), dtype=torch.bfloat16),
        }
        self.assertIsInstance(detect_arch(state_dict), ModelQwenImage21)

    def test_qwen_image_21_conversion_writes_architecture_and_protects_inputs(self):
        state_dict = {
            "txt_in.text_norm.weight": torch.ones(4096, dtype=torch.bfloat16),
            "modulation.1.weight": torch.ones((16384, 4096), dtype=torch.bfloat16),
            "transformer_blocks.0.attn.norm_q.weight": torch.ones(128, dtype=torch.bfloat16),
            "transformer_blocks.0.attn.to_q.weight": torch.ones((4096, 4096), dtype=torch.bfloat16),
            "img_in.weight": torch.ones((4096, 64), dtype=torch.bfloat16),
            "proj_out.weight": torch.ones((64, 4096), dtype=torch.bfloat16),
        }
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "qwen-image-2.1-Q4_0.gguf"
            converted_path, model_arch = convert_state_dict(
                state_dict, str(output_path), quant_type_name="Q4_0"
            )
            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            architecture = reader.get_field("general.architecture").contents()
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

        self.assertIsInstance(model_arch, ModelQwenImage21)
        self.assertEqual(architecture, "qwen_image21")
        self.assertEqual(tensor_types["transformer_blocks.0.attn.to_q.weight"], gguf.GGMLQuantizationType.Q4_0)
        self.assertEqual(tensor_types["img_in.weight"], gguf.GGMLQuantizationType.BF16)

    def test_q4_cr_keeps_attention_projections_at_source_precision(self):
        state_dict = {
            "txt_in.text_norm.weight": torch.ones(4096, dtype=torch.bfloat16),
            "modulation.1.weight": torch.ones((16384, 4096), dtype=torch.bfloat16),
            "transformer_blocks.0.attn.norm_q.weight": torch.ones(128, dtype=torch.bfloat16),
            "transformer_blocks.0.attn.to_q.weight": torch.ones((4096, 4096), dtype=torch.bfloat16),
            "transformer_blocks.0.img_mlp.gate_up.weight": torch.ones((24576, 4096), dtype=torch.bfloat16),
            "img_in.weight": torch.ones((4096, 64), dtype=torch.bfloat16),
            "proj_out.weight": torch.ones((64, 4096), dtype=torch.bfloat16),
        }
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "qwen-image-2.1-Q4_CR.gguf"
            converted_path, _ = convert_state_dict(
                state_dict, str(output_path), quant_type_name="Q4_CR", quantization_device="cpu"
            )
            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

        self.assertEqual(tensor_types["transformer_blocks.0.attn.to_q.weight"], gguf.GGMLQuantizationType.I8)
        self.assertEqual(tensor_types["transformer_blocks.0.attn.to_q.weight_scale"], gguf.GGMLQuantizationType.F32)
        self.assertEqual(tensor_types["transformer_blocks.0.img_mlp.gate_up.weight"], gguf.GGMLQuantizationType.I8)

    def test_q4_cr_l_serializes_w4a8_and_loads_mixed_q8_layers(self):
        state_dict = {
            "txt_in.text_norm.weight": torch.ones(16, dtype=torch.bfloat16),
            "modulation.1.weight": torch.ones((64, 16), dtype=torch.bfloat16),
            "transformer_blocks.0.attn.norm_q.weight": torch.ones(4, dtype=torch.bfloat16),
            "transformer_blocks.0.attn.to_q.weight": torch.randn((64, 512), dtype=torch.bfloat16),
            "transformer_blocks.0.mlp.fc1.weight": torch.randn((64, 512), dtype=torch.bfloat16),
            "transformer_blocks.0.mlp.fc2.weight": torch.randn((64, 300), dtype=torch.bfloat16),
            "img_in.weight": torch.ones((64, 128), dtype=torch.bfloat16),
            "proj_out.weight": torch.ones((64, 512), dtype=torch.bfloat16),
        }
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "qwen-image-2.1-Q4_CR_L.gguf"
            converted_path, _ = convert_state_dict(
                state_dict,
                str(output_path),
                quant_type_name="Q4_CR_L",
                quantization_device="cpu",
            )
            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            q4_conf_field = reader.get_field(
                "comfy.gguf.quant.transformer_blocks.0.mlp.fc1.weight"
            )
            q4_conf = json.loads(str(q4_conf_field.parts[q4_conf_field.data[-1]], "utf-8"))
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

            loader = load_gguf_loader()
            loaded, extra = loader.gguf_sd_loader(converted_path, dynamic=False)
            nodes_module = nodes_factory()
            ops_type = nodes_module._legacy_gguf_ops(extra)
            q4_layer = ops_type.Linear(512, 64, bias=False)
            q4_prefix = "transformer_blocks.0.mlp.fc1."
            q4_layer._load_from_state_dict(
                loaded,
                prefix=q4_prefix,
                local_metadata={},
                strict=True,
                missing_keys=[],
                unexpected_keys=[],
                error_msgs=[],
            )
            q8_layer = ops_type.Linear(512, 64, bias=False)
            q8_prefix = "transformer_blocks.0.attn.to_q."
            q8_layer._load_from_state_dict(
                loaded,
                prefix=q8_prefix,
                local_metadata={},
                strict=True,
                missing_keys=[],
                unexpected_keys=[],
                error_msgs=[],
            )
            x = torch.randn((3, 512), dtype=torch.float32)
            self.assertEqual(q4_layer._linear_dtype_for_input(x), "int8")
            qt = q4_layer._get_cached_quantized_weight(torch.device("cpu"))
            self.assertEqual(qt._params.linear_dtype, "int8")
            q4_layer.evict_quantized_caches()
            base_out = q4_layer(x)
            dense_reference = torch.nn.functional.linear(
                x, q4_layer._dequantized_weight(torch.device("cpu"), x.dtype)
            )
            torch.testing.assert_close(base_out, dense_reference, atol=1e-4, rtol=1e-3)
            from comfy.weight_adapter import LoRAAdapter
            rank = 4
            up = torch.randn((64, rank), dtype=torch.float32)
            down = torch.randn((rank, 512), dtype=torch.float32)
            adapter = LoRAAdapter(set(), (up, down, float(rank), None, None, None))
            key = "transformer_blocks.0.mlp.fc1.weight"
            patches = {key: [(0.5, adapter, 1.0, None, None)]}

            class FakeLowVramPatch:
                is_lowvram_patch = True
                def __init__(self, key, patches):
                    self.key = key
                    self.patches = patches
                    self.prepared_patches = None
                def __call__(self, value):
                    return value

            q4_layer.weight_function = [FakeLowVramPatch(key, patches)]
            native_out = q4_layer(x)
            expected = base_out + 0.5 * torch.nn.functional.linear(
                torch.nn.functional.linear(x, down), up
            )
            torch.testing.assert_close(native_out, expected, atol=1e-4, rtol=1e-3)
            self.assertIsNone(q4_layer._fused_weight)

            q4_layer.weight_function = [lambda value, *args, **kwargs: value + 0.125]
            fused_out = q4_layer(x)
            self.assertIsNotNone(q4_layer._fused_weight)
            self.assertIsNone(q4_layer._quantized_weight)
            torch.testing.assert_close(
                fused_out,
                torch.nn.functional.linear(
                    x, q4_layer._fused_weight.to(dtype=x.dtype)
                ),
                atol=1e-4,
                rtol=1e-3,
            )
            dynamic_sd, dynamic_extra = loader.gguf_sd_loader(
                converted_path, dynamic=True
            )
            self.assertEqual(dynamic_extra["gguf_quant_mode"], "int4_cr_w4a8")
            self.assertIn("transformer_blocks.0.mlp.fc1.comfy_quant", dynamic_sd)
            del dynamic_sd, dynamic_extra
            q4_default_dtype = q4_layer._default_linear_dtype
            q4_is_quantized = q4_layer._quantized
            q8_is_quantized = q8_layer._quantized
            del qt, q4_layer, q8_layer, loaded
            gc.collect()

        self.assertEqual(q4_conf["backing"], "w4a8")
        self.assertEqual(q4_conf["linear_dtype"], "int8")
        self.assertEqual(extra["gguf_quant_mode"], "int4_cr_w4a8")
        self.assertEqual(q4_default_dtype, "int8")
        self.assertTrue(q4_is_quantized)
        self.assertFalse(q8_is_quantized)
        self.assertEqual(
            tensor_types["transformer_blocks.0.attn.to_q.weight"],
            gguf.GGMLQuantizationType.I8,
        )
        self.assertEqual(
            tensor_types["transformer_blocks.0.mlp.fc1.weight"],
            gguf.GGMLQuantizationType.I8,
        )
        self.assertEqual(
            tensor_types["transformer_blocks.0.mlp.fc2.weight"],
            gguf.GGMLQuantizationType.F16,
        )
        self.assertEqual(
            tensor_types["transformer_blocks.0.attn.to_q.weight_scale"],
            gguf.GGMLQuantizationType.F32,
        )
        self.assertEqual(
            tensor_types["img_in.weight"],
            gguf.GGMLQuantizationType.BF16,
        )


class MOSSImageConversionTests(unittest.TestCase):
    @staticmethod
    def _source_state_dict():
        dtype = torch.bfloat16
        dim = 512
        ffn_dim = 1024
        return {
            "rope.bands": torch.ones(16, dtype=torch.float32),
            "final_linear_video.weight": torch.randn((128, dim), dtype=dtype),
            "final_norm_video.weight": torch.ones(dim, dtype=dtype),
            "layers.0.k_norm.weight": torch.ones(128, dtype=dtype),
            "layers.0.norm1.weight": torch.ones(dim, dtype=dtype),
            "layers.0.norm2.weight": torch.ones(dim, dtype=dtype),
            "layers.0.proj.ffn_down.weight": torch.randn((dim, ffn_dim), dtype=dtype),
            "layers.0.proj.ffn_up_gate.weight": torch.randn((2 * ffn_dim, dim), dtype=dtype),
            "layers.0.proj.o.weight": torch.randn((dim, dim), dtype=dtype),
            "layers.0.proj.qkv.weight": torch.randn((3 * dim, dim), dtype=dtype),
            "layers.0.q_norm.weight": torch.ones(128, dtype=dtype),
            "shared_block_modulation.weight": torch.randn((6 * dim, dim), dtype=dtype),
            "text_embedder.proj.weight": torch.randn((dim, 4096), dtype=dtype),
            "text_embedder.blocks.0.qkv.weight": torch.randn((3 * dim, dim), dtype=dtype),
            "time_embedding.0.weight": torch.randn((dim, 256), dtype=dtype),
            "video_embedder.weight": torch.randn((dim, 128), dtype=dtype),
            "video_head_modulation.weight": torch.randn((2 * dim, dim), dtype=dtype),
        }

    def test_detects_moss_image_from_distinctive_transformer_keys(self):
        model_arch = detect_arch(self._source_state_dict())

        self.assertIsInstance(model_arch, ModelMOSSImage)
        self.assertEqual(model_arch.arch, "moss_image")
        self.assertIn("^text_embedder.", model_arch.keys_noquant)
        self.assertIn("^rope.bands", model_arch.keys_ignore)

    def test_converts_and_loads_moss_image_gguf_in_static_and_dynamic_modes(self):
        state_dict = self._source_state_dict()
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "moss-image.safetensors"
            output_path = Path(temp_dir) / "moss-image-Q4_0.gguf"
            save_file(state_dict, str(source_path))
            converted_path, model_arch = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q4_0",
                streamed=True,
            )

            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            architecture = reader.get_field("general.architecture").contents()
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

            loader = load_gguf_loader()
            for dynamic in (False, True):
                loaded, extra = loader.gguf_sd_loader(
                    converted_path,
                    handle_prefix=None,
                    dynamic=dynamic,
                )
                self.assertEqual(extra["arch_str"], "moss_image")
                weight = loaded["layers.0.proj.ffn_up_gate.weight"]
                if dynamic:
                    loaded_shape = tuple(weight._params.orig_shape)
                else:
                    loaded_shape = tuple(weight.tensor_shape)
                self.assertEqual(loaded_shape, (2048, 512))
                del loaded, weight
                gc.collect()

        self.assertIsInstance(model_arch, ModelMOSSImage)
        self.assertEqual(architecture, "moss_image")
        self.assertNotIn("rope.bands", tensor_types)
        self.assertEqual(
            tensor_types["layers.0.proj.ffn_up_gate.weight"],
            gguf.GGMLQuantizationType.Q4_0,
        )
        self.assertEqual(
            tensor_types["text_embedder.blocks.0.qkv.weight"],
            gguf.GGMLQuantizationType.BF16,
        )
        self.assertEqual(
            tensor_types["shared_block_modulation.weight"],
            gguf.GGMLQuantizationType.BF16,
        )
        self.assertEqual(
            tensor_types["final_linear_video.weight"],
            gguf.GGMLQuantizationType.BF16,
        )

    def test_q4_cr_keeps_moss_attention_in_q8_cr(self):
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "moss-image-Q4_CR.gguf"
            converted_path, _ = convert_state_dict(
                self._source_state_dict(),
                str(output_path),
                quant_type_name="Q4_CR",
                quantization_device="cpu",
            )

            reader = gguf.GGUFReader(converted_path)
            qkv_field = reader.get_field(
                "comfy.gguf.quant.layers.0.proj.qkv.weight"
            )
            qkv_conf = json.loads(str(qkv_field.parts[qkv_field.data[-1]], "utf-8"))
            mlp_field = reader.get_field(
                "comfy.gguf.quant.layers.0.proj.ffn_up_gate.weight"
            )
            mlp_conf = json.loads(str(mlp_field.parts[mlp_field.data[-1]], "utf-8"))
            del qkv_field, mlp_field
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

            loader = load_gguf_loader()
            for dynamic in (False, True):
                loaded, extra = loader.gguf_sd_loader(
                    converted_path,
                    handle_prefix=None,
                    dynamic=dynamic,
                )
                self.assertEqual(extra["gguf_quant_mode"], "int4_cr_w4a4")
                self.assertIn("layers.0.proj.qkv.comfy_quant", loaded)
                del loaded
                gc.collect()

        self.assertEqual(qkv_conf["format"], "int8_tensorwise")
        self.assertEqual(mlp_conf["format"], "int4_cr")
        self.assertEqual(mlp_conf["backing"], "w4a4")

    def test_installed_comfyui_core_detects_moss_image_when_supported(self):
        from comfy.model_detection import detect_unet_config

        config = detect_unet_config(self._source_state_dict(), "")
        if config is None:
            self.skipTest(
                "Installed ComfyUI predates core MOSS-Image support "
                "(upstream commit e087d8f6)."
            )
        self.assertEqual(config["image_model"], "moss_image")

    def test_standalone_converter_fuses_split_qkv_lora_without_pythonpath(self):
        from comfy.model_detection import detect_unet_config

        source = self._source_state_dict()
        if detect_unet_config(source, "") is None:
            self.skipTest(
                "Installed ComfyUI predates core MOSS-Image support "
                "(upstream commit e087d8f6)."
            )

        dim = source["layers.0.proj.qkv.weight"].shape[1]
        source["layers.0.proj.qkv.weight"] = torch.zeros_like(
            source["layers.0.proj.qkv.weight"]
        )
        lora = {}
        for projection in ("q", "k", "v"):
            target = f"diffusion_model.layers.0.proj.{projection}"
            down = torch.zeros((1, dim), dtype=torch.float16)
            down[0, 0] = 1
            up = torch.zeros((dim, 1), dtype=torch.float16)
            up[0, 0] = 1
            lora[f"{target}.lora_down.weight"] = down
            lora[f"{target}.lora_up.weight"] = up
            lora[f"{target}.alpha"] = torch.tensor(1.0)

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "moss-image.safetensors"
            lora_path = Path(temp_dir) / "moss-image-lora.safetensors"
            output_path = Path(temp_dir) / "moss-image-lora.gguf"
            save_file(source, str(source_path))
            save_file(lora, str(lora_path))

            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            repo_root = Path(__file__).parents[1]
            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "tools" / "convert.py"),
                    "--src",
                    str(source_path),
                    "--dst",
                    str(output_path),
                    "--lora",
                    str(lora_path),
                    "--lora-strength",
                    "0.7",
                    "--quantization-device",
                    "cpu",
                    "--quant-type",
                    "F16",
                ],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"Converter failed without PYTHONPATH:\n{result.stdout}\n{result.stderr}",
            )

            reader = gguf.GGUFReader(str(output_path))
            try:
                tensor = next(
                    tensor
                    for tensor in reader.tensors
                    if tensor.name == "layers.0.proj.qkv.weight"
                )
                merged = torch.from_numpy(tensor.data.copy()).view(torch.float16).reshape(
                    tuple(reversed(tensor.shape))
                )
            finally:
                reader.tensors.clear()
                reader.fields.clear()
                reader.data._mmap.close()

        self.assertTrue(
            torch.equal(
                merged[[0, dim, 2 * dim], 0],
                torch.full(
                    (3,),
                    torch.tensor(0.7, dtype=torch.bfloat16).to(torch.float16).item(),
                    dtype=torch.float16,
                ),
            ),
            f"Unexpected fused Q/K/V values: {merged[[0, dim, 2 * dim], 0]}",
        )
        self.assertEqual(torch.count_nonzero(merged).item(), 3)


class Fibo2ConversionTests(unittest.TestCase):
    @staticmethod
    def _source_state_dict():
        dtype = torch.bfloat16
        dim = 512
        ffn_dim = 1024
        context_dim = 1024

        def block_tensors(prefix, with_cross_attention=False):
            tensors = {
                f"{prefix}.attention.qkv.weight": torch.randn((3 * dim, dim), dtype=dtype),
                f"{prefix}.attention.out.weight": torch.randn((dim, dim), dtype=dtype),
                f"{prefix}.attention.q_norm.weight": torch.ones(128, dtype=dtype),
                f"{prefix}.attention.k_norm.weight": torch.ones(128, dtype=dtype),
                f"{prefix}.feed_forward.w13.weight": torch.randn((2 * ffn_dim, dim), dtype=dtype),
                f"{prefix}.feed_forward.w2.weight": torch.randn((dim, ffn_dim), dtype=dtype),
                f"{prefix}.attention_norm1.weight": torch.ones(dim, dtype=dtype),
                f"{prefix}.attention_norm2.weight": torch.ones(dim, dtype=dtype),
                f"{prefix}.ffn_norm1.weight": torch.ones(dim, dtype=dtype),
                f"{prefix}.ffn_norm2.weight": torch.ones(dim, dtype=dtype),
                f"{prefix}.adaLN_modulation.0.weight": torch.randn((4 * dim, 256), dtype=dtype),
                f"{prefix}.adaLN_modulation.0.bias": torch.randn((4 * dim,), dtype=dtype),
            }
            if with_cross_attention:
                tensors.update({
                    f"{prefix}.cross_attn_norm1.weight": torch.ones(dim, dtype=dtype),
                    f"{prefix}.cross_attention.to_q.weight": torch.randn((dim, dim), dtype=dtype),
                    f"{prefix}.cross_attention.to_kv.weight": torch.randn((2 * dim, context_dim), dtype=dtype),
                    f"{prefix}.cross_attention.to_out.weight": torch.randn((dim, dim), dtype=dtype),
                    f"{prefix}.cross_attention.norm_q.weight": torch.ones(128, dtype=dtype),
                    f"{prefix}.cross_attention.norm_k.weight": torch.ones(128, dtype=dtype),
                    f"{prefix}.cross_attn_norm2.weight": torch.ones(dim, dtype=dtype),
                    f"{prefix}.cross_attn_gate": torch.ones(1, dtype=dtype),
                })
            return tensors

        state_dict = {
            "x_embedder.weight": torch.randn((dim, 128), dtype=dtype),
            "x_embedder.bias": torch.randn((dim,), dtype=dtype),
            "t_embedder.mlp.0.weight": torch.randn((1024, 256), dtype=dtype),
            "t_embedder.mlp.0.bias": torch.randn((1024,), dtype=dtype),
            "t_embedder.mlp.2.weight": torch.randn((256, 1024), dtype=dtype),
            "t_embedder.mlp.2.bias": torch.randn((256,), dtype=dtype),
            "perceiver.latent_queries": torch.randn((512, dim), dtype=dtype),
            "perceiver.norm_out.weight": torch.ones(dim, dtype=dtype),
            "final_layer.adaLN_modulation.1.weight": torch.ones(dim, dtype=dtype),
            "final_layer.adaLN_modulation.1.bias": torch.zeros(dim, dtype=dtype),
            "final_layer.linear.weight": torch.randn((128, dim), dtype=dtype),
        }
        state_dict.update(block_tensors("noise_refiner.0"))
        state_dict.update({
            "perceiver.layers.0.cross_attn_norm.weight": torch.ones(dim, dtype=dtype),
            "perceiver.layers.0.cross_attention.to_q.weight": torch.randn((dim, dim), dtype=dtype),
            "perceiver.layers.0.cross_attention.to_kv.weight": torch.randn((2 * dim, context_dim), dtype=dtype),
            "perceiver.layers.0.cross_attention.to_out.weight": torch.randn((dim, dim), dtype=dtype),
            "perceiver.layers.0.cross_attention.norm_q.weight": torch.ones(128, dtype=dtype),
            "perceiver.layers.0.cross_attention.norm_k.weight": torch.ones(128, dtype=dtype),
            "perceiver.layers.0.self_attn_norm.weight": torch.ones(dim, dtype=dtype),
            "perceiver.layers.0.self_attention.to_q.weight": torch.randn((dim, dim), dtype=dtype),
            "perceiver.layers.0.self_attention.to_kv.weight": torch.randn((2 * dim, dim), dtype=dtype),
            "perceiver.layers.0.self_attention.to_out.weight": torch.randn((dim, dim), dtype=dtype),
            "perceiver.layers.0.self_attention.norm_q.weight": torch.ones(128, dtype=dtype),
            "perceiver.layers.0.self_attention.norm_k.weight": torch.ones(128, dtype=dtype),
            "perceiver.layers.0.ffn_norm.weight": torch.ones(dim, dtype=dtype),
            "perceiver.layers.0.ffn.w13.weight": torch.randn((2 * ffn_dim, dim), dtype=dtype),
            "perceiver.layers.0.ffn.w2.weight": torch.randn((dim, ffn_dim), dtype=dtype),
        })
        state_dict.update(block_tensors("layers.0", with_cross_attention=True))
        return state_dict

    def test_detects_fibo2_from_perceiver_transformer_and_output_keys(self):
        model_arch = detect_arch(self._source_state_dict())

        self.assertIsInstance(model_arch, ModelFibo2)
        self.assertEqual(model_arch.arch, "fibo2")

    def test_converts_and_loads_fibo2_gguf_with_precision_rules(self):
        state_dict = self._source_state_dict()
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "fibo2-Q4_0.gguf"
            converted_path, model_arch = convert_state_dict(
                state_dict, str(output_path), quant_type_name="Q4_0"
            )

            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            architecture = reader.get_field("general.architecture").contents()
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

            loader = load_gguf_loader()
            for dynamic in (False, True):
                loaded, extra = loader.gguf_sd_loader(
                    converted_path,
                    handle_prefix=None,
                    dynamic=dynamic,
                )
                self.assertEqual(extra["arch_str"], "fibo2")
                del loaded
                gc.collect()

        self.assertIsInstance(model_arch, ModelFibo2)
        self.assertEqual(architecture, "fibo2")
        self.assertEqual(
            tensor_types["layers.0.attention.qkv.weight"],
            gguf.GGMLQuantizationType.Q4_0,
        )
        self.assertEqual(
            tensor_types["x_embedder.weight"],
            gguf.GGMLQuantizationType.Q4_0,
        )
        self.assertEqual(
            tensor_types["perceiver.layers.0.cross_attention.to_kv.weight"],
            gguf.GGMLQuantizationType.Q4_0,
        )
        self.assertEqual(
            tensor_types["perceiver.latent_queries"],
            gguf.GGMLQuantizationType.BF16,
        )
        self.assertEqual(
            tensor_types["layers.0.adaLN_modulation.0.weight"],
            gguf.GGMLQuantizationType.BF16,
        )
        self.assertEqual(
            tensor_types["final_layer.adaLN_modulation.1.weight"],
            gguf.GGMLQuantizationType.F32,
        )
        self.assertEqual(
            tensor_types["noise_refiner.0.feed_forward.w13.weight"],
            gguf.GGMLQuantizationType.Q4_0,
        )

    def test_k_quants_preserve_128_wide_fibo2_input_projection(self):
        for quant_type_name in ("Q6_K", "Q4_K_M"):
            with self.subTest(quant_type_name=quant_type_name):
                state_dict = self._source_state_dict()
                state_dict["x_embedder.weight"] = torch.randn(
                    (3584, 128), dtype=torch.bfloat16
                )
                with TemporaryDirectory() as temp_dir:
                    output_path = Path(temp_dir) / f"fibo2-{quant_type_name}.gguf"
                    converted_path, model_arch = convert_state_dict(
                        state_dict,
                        str(output_path),
                        quant_type_name=quant_type_name,
                    )

                    reader = gguf.GGUFReader(converted_path)
                    tensor_types = {
                        tensor.name: tensor.tensor_type for tensor in reader.tensors
                    }
                    reader.tensors.clear()
                    reader.fields.clear()
                    reader.data._mmap.close()
                    del reader

                self.assertIsInstance(model_arch, ModelFibo2)
                self.assertEqual(
                    tensor_types["x_embedder.weight"],
                    gguf.GGMLQuantizationType.BF16,
                )
                if quant_type_name == "Q6_K":
                    self.assertEqual(
                        tensor_types["layers.0.attention.qkv.weight"],
                        gguf.GGMLQuantizationType.Q6_K,
                    )
                else:
                    self.assertIn(
                        tensor_types["layers.0.attention.qkv.weight"],
                        {
                            gguf.GGMLQuantizationType.Q4_K,
                            gguf.GGMLQuantizationType.Q5_K,
                            gguf.GGMLQuantizationType.Q6_K,
                        },
                    )


class QwenImage21InferenceFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = load_gguf_loader()
        cls.nodes = nodes_factory()
        cls.ops = ops_module_factory()

    def _write_qwen21_q4_gguf(self, output_path):
        writer = gguf.GGUFWriter(path=None, arch="qwen_image21")
        for name in (
            "txt_in.text_norm.weight",
            "transformer_blocks.0.attn.norm_q.weight",
        ):
            writer.add_tensor(
                name,
                np.full((32,), 0x3F80, dtype=np.uint16),
                raw_dtype=gguf.GGMLQuantizationType.BF16,
            )
        for name in ("modulation.1.weight", "img_in.weight", "proj_out.weight"):
            writer.add_tensor(
                name,
                np.full((32, 32), 0x3F80, dtype=np.uint16),
                raw_dtype=gguf.GGMLQuantizationType.BF16,
            )
        q4 = gguf.quants.quantize(
            np.ones((32, 32), dtype=np.float32), gguf.GGMLQuantizationType.Q4_0
        )
        writer.add_tensor(
            "transformer_blocks.0.attn.to_q.weight",
            q4,
            raw_dtype=gguf.GGMLQuantizationType.Q4_0,
        )
        writer.write_header_to_file(path=str(output_path))
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def _patch_node_loader(self, stack, patcher, extra):
        progress = mock.Mock()
        progress.callback_for.return_value = None
        stack.enter_context(
            mock.patch.object(self.nodes, "GGUFLoadProgress", return_value=progress)
        )
        stack.enter_context(
            mock.patch.object(self.nodes, "gguf_sd_loader", return_value=({}, extra))
        )
        stack.enter_context(
            mock.patch.object(self.nodes, "device_supports_bf16", return_value=False)
        )
        stack.enter_context(
            mock.patch.object(
                self.nodes.comfy.model_management,
                "get_torch_device",
                return_value=torch.device("cuda:0"),
            )
        )
        stack.enter_context(
            mock.patch.object(
                self.nodes.comfy.model_management,
                "should_use_fp16",
                return_value=True,
            )
        )
        stack.enter_context(
            mock.patch.object(
                self.nodes.comfy.model_management,
                "args",
                SimpleNamespace(
                    fp32_unet=False,
                    fp64_unet=False,
                    bf16_unet=False,
                    fp8_e4m3fn_unet=False,
                    fp8_e5m2_unet=False,
                    fp8_e8m0fnu_unet=False,
                ),
            )
        )
        stack.enter_context(
            mock.patch.object(
                self.nodes.inspect,
                "signature",
                return_value=SimpleNamespace(
                    parameters={"metadata": None, "manual_cast": None}
                ),
            )
        )
        return stack.enter_context(
            mock.patch.object(
                self.nodes.comfy.sd,
                "load_diffusion_model_state_dict",
                return_value=patcher,
            )
        )

    def test_bf16_storage_weights_fall_back_to_fp16_in_static_and_dynamic_loading(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "qwen-image-2.1-Q4_0.gguf"
            self._write_qwen21_q4_gguf(path)
            with mock.patch.object(
                self.loader, "device_supports_bf16", return_value=False
            ), mock.patch.object(
                self.loader, "dynamic_gguf_file_slice", return_value=None
            ):
                static, static_extra = self.loader.gguf_sd_loader(
                    str(path), handle_prefix=None
                )
                dynamic, dynamic_extra = self.loader.gguf_sd_loader(
                    str(path), handle_prefix=None, dynamic=True
                )

                self.assertEqual(static_extra["arch_str"], "qwen_image21")
                self.assertEqual(dynamic_extra["arch_str"], "qwen_image21")
                self.assertEqual(static["img_in.weight"].dtype, torch.float16)
                self.assertEqual(dynamic["img_in.weight"].dtype, torch.float16)
                self.assertEqual(static["transformer_blocks.0.attn.norm_q.weight"].dtype, torch.float32)
                self.assertEqual(dynamic["transformer_blocks.0.attn.norm_q.weight"].dtype, torch.float32)
                self.assertEqual(static["transformer_blocks.0.attn.to_q.weight"].dtype, torch.float16)
                self.assertEqual(dynamic["transformer_blocks.0.attn.to_q.weight"].dtype, torch.float16)
                del static
                del dynamic
                gc.collect()

                with mock.patch.object(
                    self.loader, "device_supports_bf16", return_value=True
                ):
                    bf16_static, _ = self.loader.gguf_sd_loader(
                        str(path), handle_prefix=None
                    )
                    bf16_dynamic, _ = self.loader.gguf_sd_loader(
                        str(path), handle_prefix=None, dynamic=True
                    )

                self.assertEqual(bf16_static["img_in.weight"].dtype, torch.bfloat16)
                self.assertEqual(bf16_dynamic["img_in.weight"].dtype, torch.bfloat16)
                del bf16_static
                del bf16_dynamic
                gc.collect()

    def test_fp16_fallback_is_limited_to_qwen21_and_fp16_capable_devices(self):
        args = SimpleNamespace()
        with ExitStack() as stack:
            bf16_support = stack.enter_context(
                mock.patch.object(self.nodes, "device_supports_bf16", return_value=False)
            )
            fp16_support = stack.enter_context(
                mock.patch.object(
                    self.nodes.comfy.model_management,
                    "should_use_fp16",
                    return_value=True,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    self.nodes.comfy.model_management,
                    "get_torch_device",
                    return_value=torch.device("cuda:0"),
                )
            )
            stack.enter_context(
                mock.patch.object(self.nodes.comfy.model_management, "args", args)
            )

            qwen21 = {"arch_str": "qwen_image21"}
            self.assertEqual(
                self.nodes._qwen_image21_fp16_fallback_dtype(qwen21),
                torch.float16,
            )
            self.assertIsNone(
                self.nodes._qwen_image21_fp16_fallback_dtype({"arch_str": "flux"})
            )

            bf16_support.return_value = True
            self.assertIsNone(self.nodes._qwen_image21_fp16_fallback_dtype(qwen21))
            bf16_support.return_value = False
            fp16_support.return_value = False
            self.assertIsNone(self.nodes._qwen_image21_fp16_fallback_dtype(qwen21))

            fp16_support.return_value = True
            args.fp32_unet = True
            self.assertIsNone(self.nodes._qwen_image21_fp16_fallback_dtype(qwen21))
            args.fp32_unet = False
            with mock.patch.object(
                self.nodes.comfy.model_management, "FORCE_FP32", True
            ):
                self.assertIsNone(
                    self.nodes._qwen_image21_fp16_fallback_dtype(qwen21)
                )

    def test_static_loader_sets_fp16_model_and_manual_cast_dtypes(self):
        base_model = SimpleNamespace(
            manual_cast_dtype=None,
            model_config=SimpleNamespace(manual_cast_dtype=None),
        )
        patcher = SimpleNamespace(model=base_model)
        extra = {"arch_str": "qwen_image21", "metadata": {}}
        custom_ops = SimpleNamespace(
            Linear=SimpleNamespace(dequant_dtype=None, patch_dtype=None)
        )

        with ExitStack() as stack:
            load_model = self._patch_node_loader(stack, patcher, extra)
            stack.enter_context(
                mock.patch.object(
                    self.nodes, "_legacy_gguf_ops", return_value=custom_ops
                )
            )
            stack.enter_context(
                mock.patch.object(
                    self.nodes.folder_paths,
                    "get_full_path",
                    return_value="qwen-image-2.1-Q4_0.gguf",
                )
            )
            stack.enter_context(
                mock.patch.object(
                    self.nodes.GGUFModelPatcher, "clone", return_value=patcher
                )
            )

            loaded, = self.nodes.UnetLoaderGGUF().load_unet("qwen-image-2.1-Q4_0.gguf")

        model_options = load_model.call_args.kwargs["model_options"]
        self.assertIs(loaded, patcher)
        self.assertEqual(model_options["dtype"], torch.float16)
        self.assertIs(model_options["custom_operations"], custom_ops)
        self.assertEqual(base_model.manual_cast_dtype, torch.float16)
        self.assertEqual(base_model.model_config.manual_cast_dtype, torch.float16)

    def test_dynamic_loader_sets_fp16_model_and_manual_cast_dtypes(self):
        base_model = SimpleNamespace(
            manual_cast_dtype=None,
            model_config=SimpleNamespace(manual_cast_dtype=None),
        )
        patcher = SimpleNamespace(model=base_model)
        extra = {"arch_str": "qwen_image21", "metadata": {}}
        custom_ops = SimpleNamespace(
            Linear=SimpleNamespace(dequant_dtype=None, patch_dtype=None)
        )

        with ExitStack() as stack:
            load_model = self._patch_node_loader(stack, patcher, extra)
            stack.enter_context(
                mock.patch.object(
                    self.nodes, "_legacy_gguf_ops", return_value=custom_ops
                )
            )
            stack.enter_context(
                mock.patch.object(self.nodes, "_require_dynamic_vram")
            )
            stack.enter_context(
                mock.patch.object(
                    self.nodes, "_clone_as_dynamic_gguf_patcher", return_value=patcher
                )
            )

            loaded = self.nodes._load_dynamic_gguf_unet("qwen-image-2.1-Q4_0.gguf")

        model_options = load_model.call_args.kwargs["model_options"]
        self.assertIs(loaded, patcher)
        self.assertEqual(model_options["dtype"], torch.float16)
        self.assertIs(model_options["custom_operations"], custom_ops)
        self.assertEqual(base_model.manual_cast_dtype, torch.float16)
        self.assertEqual(base_model.model_config.manual_cast_dtype, torch.float16)

    def test_ggml_linear_dequantizes_q4_0_in_the_selected_fp16_compute_dtype(self):
        packed = gguf.quants.quantize(
            np.ones((4, 32), dtype=np.float32), gguf.GGMLQuantizationType.Q4_0
        ).copy()
        qweight = self.ops.GGMLTensor(
            torch.from_numpy(packed),
            tensor_type=gguf.GGMLQuantizationType.Q4_0,
            tensor_shape=(4, 32),
        )
        layer = self.ops.GGMLOps().Linear(
            32, 4, bias=False, device="cpu", dtype=torch.float16
        )
        layer.load_state_dict({"weight": qweight}, strict=False)

        dequantized = layer.get_weight(layer.weight, layer.weight.dtype)

        self.assertEqual(layer.weight.dtype, torch.float16)
        self.assertEqual(dequantized.dtype, torch.float16)

    def test_custom_quantized_ops_honor_selected_compute_dtype(self):
        q8_ops = self.nodes._legacy_gguf_ops(
            {"gguf_quant_mode": "int8_convrot"}, compute_dtype=torch.float16
        )
        q8_layer = q8_ops.Linear(64, 6, bias=True, device="cpu", dtype=torch.float16)
        self.assertEqual(q8_layer.factory_kwargs["dtype"], torch.float16)

        q4_ops = self.ops.get_gguf_q4_w4a4_ops(compute_dtype=torch.float16)
        q4_layer = q4_ops.Linear(64, 64, bias=False, device="cpu", dtype=torch.float16)
        self.assertEqual(q4_layer._compute_dtype, torch.float16)


class MiniMaxH3VAEConversionTests(unittest.TestCase):
    def test_detects_video_vae_from_distinctive_decoder_and_encoder_keys(self):
        state_dict = {
            "decoder.transformer_blocks.0.scale1": torch.ones(32),
            "decoder.x_embedder.weight": torch.ones((64, 32)),
            "encoder.down.5.block.0.conv1.weight": torch.ones((2, 2, 3, 3, 3)),
        }

        self.assertIsInstance(detect_arch(state_dict), ModelMinimaxH3VAE)

    def test_q8_cr_keeps_conv3d_fp16_and_restores_its_shape(self):
        state_dict = {
            "decoder.transformer_blocks.0.scale1": torch.ones(32, dtype=torch.float16),
            "decoder.x_embedder.weight": torch.ones((64, 32), dtype=torch.float16),
            "encoder.down.5.block.0.conv1.weight": torch.ones(
                (2, 2, 3, 3, 3), dtype=torch.float16
            ),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3_vae.safetensors"
            output_path = Path(temp_dir) / "minimax_h3_vae-Q8_CR.gguf"
            save_file(state_dict, str(source_path))

            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q8_CR",
                quantization_device="cpu",
            )
            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

            loader = load_gguf_loader()
            loaded, extra = loader.gguf_sd_loader(converted_path, handle_prefix=None)
            conv3d = loaded["encoder.down.5.block.0.conv1.weight"]
            del conv3d.tensor_type
            materialized = dequantize_tensor(conv3d, dtype=torch.Tensor(conv3d).dtype)
            conv3d_shape = tuple(materialized.shape)
            del materialized
            del conv3d
            del loaded
            gc.collect()

        self.assertEqual(extra["arch_str"], "minimax_h3_vae")
        self.assertEqual(
            tensor_types["decoder.x_embedder.weight"], gguf.GGMLQuantizationType.I8
        )
        self.assertEqual(
            tensor_types["encoder.down.5.block.0.conv1.weight"],
            gguf.GGMLQuantizationType.F16,
        )
        self.assertEqual(
            conv3d_shape,
            (2, 2, 3, 3, 3),
        )


class MiniMaxMusic3ConversionTests(unittest.TestCase):
    def test_detects_dit_from_music_specific_conditioning_and_attention_keys(self):
        state_dict = {
            "cond_layer_logits": torch.ones(8),
            "latent_conditioners.0.weight": torch.ones((2, 2, 1)),
            "diffusion_transformer.preprocess_conv.weight": torch.ones((2, 2, 1)),
            "diffusion_transformer.postprocess_conv.weight": torch.ones((2, 2, 1)),
            "diffusion_transformer.timestep_features.weight": torch.ones((128, 1)),
            "diffusion_transformer.transformer.rotary_pos_emb.inv_freq": torch.ones(16),
            "diffusion_transformer.transformer.project_in.weight": torch.ones((64, 64)),
            "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight": torch.ones((192, 64)),
        }

        self.assertIsInstance(detect_arch(state_dict), ModelMiniMaxMusic3DiT)

    def test_detects_pruned_text_encoder_from_audio_and_qwen_keys(self):
        state_dict = {
            "model.embed_tokens_prefill.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.embed_tokens_audio.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.audio_extra_embedding.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.audio_decoder.pos_embedding.weight": torch.ones((16, 32), dtype=torch.bfloat16),
            "model.lm_head_pruned.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.audio_decoder.audio_heads.0.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.layers.0.self_attn.qkv_proj.weight": torch.ones((96, 32), dtype=torch.bfloat16),
        }

        self.assertIsInstance(detect_arch(state_dict), ModelMiniMaxMusic3TextEncoder)

    def test_q8_cr_preserves_dit_convolutions(self):
        state_dict = {
            "cond_layer_logits": torch.ones(8),
            "latent_conditioners.0.weight": torch.ones((2, 2, 1)),
            "diffusion_transformer.preprocess_conv.weight": torch.ones((2, 2, 1)),
            "diffusion_transformer.postprocess_conv.weight": torch.ones((2, 2, 1)),
            "diffusion_transformer.timestep_features.weight": torch.ones((128, 1)),
            "diffusion_transformer.transformer.rotary_pos_emb.inv_freq": torch.ones(16),
            "diffusion_transformer.transformer.project_in.weight": torch.ones((64, 64)),
            "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight": torch.ones((192, 64)),
        }
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "minimax_music3_dit-Q8_CR.gguf"
            converted_path, model_arch = convert_state_dict(
                state_dict,
                str(output_path),
                quant_type_name="Q8_CR",
                quantization_device="cpu",
            )
            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            loaded, extra = load_gguf_loader().gguf_sd_loader(converted_path, handle_prefix=None)
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader
            del loaded

        self.assertIsInstance(model_arch, ModelMiniMaxMusic3DiT)
        self.assertEqual(extra["arch_str"], "minimax_music3")
        self.assertEqual(
            tensor_types["diffusion_transformer.transformer.project_in.weight"],
            gguf.GGMLQuantizationType.I8,
        )
        self.assertEqual(
            tensor_types["latent_conditioners.0.weight"],
            gguf.GGMLQuantizationType.F32,
        )

    def test_q8_cr_preserves_text_embeddings(self):
        state_dict = {
            "model.embed_tokens_prefill.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.embed_tokens_audio.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.audio_extra_embedding.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.audio_decoder.pos_embedding.weight": torch.ones((16, 32), dtype=torch.bfloat16),
            "model.lm_head_pruned.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.audio_decoder.audio_heads.0.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.layers.0.self_attn.qkv_proj.weight": torch.ones((96, 32), dtype=torch.bfloat16),
        }
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_music3_text_encoder.safetensors"
            output_path = Path(temp_dir) / "minimax_music3_text_encoder-Q8_CR.gguf"
            save_file(state_dict, str(source_path))
            converted_path, model_arch = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q8_CR",
                quantization_device="cpu",
                streamed=True,
            )
            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            loaded, extra = load_gguf_loader().gguf_sd_loader(
                converted_path,
                handle_prefix=None,
                is_text_model=True,
            )
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader
            del loaded
            gc.collect()

        self.assertIsInstance(model_arch, ModelMiniMaxMusic3TextEncoder)
        self.assertEqual(extra["arch_str"], "minimax_music3")
        self.assertEqual(
            tensor_types["model.embed_tokens_prefill.weight"],
            gguf.GGMLQuantizationType.BF16,
        )
        self.assertEqual(
            tensor_types["model.layers.0.self_attn.qkv_proj.weight"],
            gguf.GGMLQuantizationType.I8,
        )

    def test_tokenizer_json_roundtrips_as_raw_bytes_for_all_quantizers(self):
        tokenizer_json = b'{"text":"caf\xc3\xb6"}'
        state_dict = {
            "model.embed_tokens_prefill.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.embed_tokens_audio.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.audio_extra_embedding.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.audio_decoder.pos_embedding.weight": torch.ones((16, 32), dtype=torch.bfloat16),
            "model.lm_head_pruned.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.audio_decoder.audio_heads.0.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.layers.0.self_attn.qkv_proj.weight": torch.ones((96, 32), dtype=torch.bfloat16),
            "tokenizer_json": torch.tensor(list(tokenizer_json), dtype=torch.uint8),
        }
        model_arch = detect_arch(state_dict)
        plan, _, _ = plan_target_size_quantization(
            state_dict,
            model_arch,
            max_size_mb=1,
        )
        self.assertEqual(plan["tokenizer_json"], gguf.GGMLQuantizationType.I8)

        loader = load_gguf_loader()
        for quant_type in ("Q4_0", "Q8_CR"):
            with self.subTest(quant_type=quant_type), TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / f"minimax_music3_text_encoder-{quant_type}.gguf"
                converted_path, _ = convert_state_dict(
                    state_dict,
                    str(output_path),
                    quant_type_name=quant_type,
                    quantization_device="cpu",
                )
                reader = gguf.GGUFReader(converted_path)
                tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
                self.assertEqual(
                    tensor_types["tokenizer_json"],
                    gguf.GGMLQuantizationType.I8,
                )
                loaded = loader.gguf_clip_loader(converted_path)
                self.assertEqual(loaded["tokenizer_json"].dtype, torch.uint8)
                self.assertEqual(loaded["tokenizer_json"].tolist(), list(tokenizer_json))
                reader.tensors.clear()
                reader.fields.clear()
                reader.data._mmap.close()
                del reader
                del loaded
                gc.collect()

    def test_clip_loader_restores_legacy_float_tokenizer_bytes(self):
        loader = load_gguf_loader()
        tokenizer_json = b'{"text":"caf\xc3\xb6"}'
        legacy = torch.tensor(list(tokenizer_json), dtype=torch.float32)
        with mock.patch.object(
            loader,
            "gguf_sd_loader",
            return_value=(
                {"tokenizer_json": loader.GGMLTensor(
                    legacy,
                    tensor_type=gguf.GGMLQuantizationType.F32,
                    tensor_shape=legacy.shape,
                )},
                {"arch_str": "minimax_music3"},
            ),
        ):
            loaded = loader.gguf_clip_loader("legacy-minimax-music3.gguf")

        self.assertEqual(loaded["tokenizer_json"].dtype, torch.uint8)
        self.assertEqual(loaded["tokenizer_json"].tolist(), list(tokenizer_json))


class LTX25ConversionTests(unittest.TestCase):
    def test_detects_ltx25_audio_video_transformer(self):
        state_dict = {
            "adaln_single.emb.timestep_embedder.linear_2.weight": torch.ones((32, 16)),
            "audio_adaln_single.linear.weight": torch.ones((32, 16)),
            "transformer_blocks.27.scale_shift_table": torch.ones((6, 32)),
        }

        self.assertIsInstance(detect_arch(state_dict), ModelLTXV)

    def test_converts_temporal_upscaler_without_quantizing_conv3d(self):
        state_dict = {
            "initial_conv.weight": torch.ones((32, 16, 3, 3, 3), dtype=torch.bfloat16),
            "post_upsample_res_blocks.0.conv2.bias": torch.ones(32, dtype=torch.bfloat16),
            "upsampler.0.weight": torch.ones((64, 32, 3, 3, 3), dtype=torch.bfloat16),
            "final_conv.weight": torch.ones((16, 32, 3, 3, 3), dtype=torch.bfloat16),
        }
        metadata = {
            "config": json.dumps(
                {
                    "_class_name": "LatentUpsampler",
                    "in_channels": 16,
                    "mid_channels": 32,
                    "num_blocks_per_stage": 1,
                    "dims": 3,
                    "spatial_upsample": False,
                    "temporal_upsample": True,
                }
            )
        }

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "ltx25-temporal-upscaler-Q8_CR.gguf"
            converted_path, model_arch = convert_state_dict(
                state_dict,
                str(output_path),
                source_metadata=metadata,
                quant_type_name="Q8_CR",
                quantization_device="cpu",
            )
            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            loader = load_gguf_loader()
            loaded, extra = loader.gguf_sd_loader(converted_path, handle_prefix=None)
            restored_shape = tuple(
                loaded["upsampler.0.weight"].shape
            )
            loaded_arch = extra["arch_str"]
            config_dims = json.loads(extra["metadata"]["config"])["dims"]
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader
            del loaded
            del extra
            gc.collect()

        self.assertIsInstance(model_arch, ModelLTXVUpsampler)
        self.assertEqual(loaded_arch, "ltxv_upscaler")
        self.assertEqual(
            tensor_types["upsampler.0.weight"], gguf.GGMLQuantizationType.BF16
        )
        self.assertEqual(restored_shape, (64, 32, 3, 3, 3))
        self.assertEqual(config_dims, 3)


class GGUFLoraTests(unittest.TestCase):
    def _write_lora(self, path, down, up, alpha=4.0):
        writer = gguf.GGUFWriter(path=None, arch="minimax_h3")
        writer.add_string("general.type", "adapter")
        writer.add_string("adapter.type", "lora")
        writer.add_float32("adapter.lora.alpha", alpha)
        writer.add_tensor(
            "blocks.0.attn.qkv_proj.weight.lora_a",
            down.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
        writer.add_tensor(
            "blocks.0.attn.qkv_proj.weight.lora_b",
            up.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
        writer.write_header_to_file(path=str(path))
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_imports_standard_gguf_factor_pair(self):
        down = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        up = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adapter.gguf"
            self._write_lora(path, down, up)
            lora, targets, metadata = load_gguf_lora(path)

        self.assertEqual(metadata["alpha"], 4.0)
        self.assertIn("blocks.0.attn.qkv_proj.lora_A.weight", lora)
        self.assertIn("blocks.0.attn.qkv_proj.lora_B.weight", lora)
        self.assertTrue(torch.equal(targets["blocks.0.attn.qkv_proj"]["down"], down))
        self.assertTrue(torch.equal(targets["blocks.0.attn.qkv_proj"]["up"], up))

    def test_fuses_lora_delta_in_selected_precision(self):
        state_dict = {
            "blocks.0.attn.qkv_proj.weight": torch.zeros((3, 2), dtype=torch.float16)
        }
        targets = {
            "blocks.0.attn.qkv_proj": {
                "base_name": "blocks.0.attn.qkv_proj.weight",
                "down": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "up": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                "alpha": 4.0,
            }
        }

        count = fuse_targets_into_state_dict(
            state_dict, targets, strength=0.5, device=torch.device("cpu")
        )

        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [4.0, 6.0]], dtype=torch.float16)
        self.assertEqual(count, 1)
        self.assertTrue(torch.equal(state_dict["blocks.0.attn.qkv_proj.weight"], expected))

    def test_restores_per_row_scaled_int8_source_weights(self):
        state_dict = {
            "blocks.0.attn.qkv_proj.weight": torch.tensor(
                [[10, -20], [30, -40]], dtype=torch.int8
            ),
            "blocks.0.attn.qkv_proj.weight_scale": torch.tensor([0.1, 0.01]),
        }

        restored_count = materialize_int8_source_weights(state_dict)

        self.assertEqual(restored_count, 1)
        self.assertNotIn("blocks.0.attn.qkv_proj.weight_scale", state_dict)
        self.assertEqual(state_dict["blocks.0.attn.qkv_proj.weight"].dtype, torch.float16)
        self.assertTrue(
            torch.equal(
                state_dict["blocks.0.attn.qkv_proj.weight"],
                torch.tensor([[1.0, -2.0], [0.3, -0.4]], dtype=torch.float16),
            )
        )

    def test_restores_convrot_int8_source_weights(self):
        source = torch.tensor(
            [[2.0, 2.0, 2.0, 2.0], [2.0, -2.0, 2.0, -2.0]],
            dtype=torch.float32,
        )
        qdata, scale, quant_conf, _ = quantize_int8_convrot(
            source, convrot_groupsize=4, device=torch.device("cpu")
        )
        state_dict = {
            "blocks.0.proj.weight": qdata,
            "blocks.0.proj.weight_scale": scale,
        }

        materialize_int8_source_weights(
            state_dict,
            {
                "_quantization_metadata": json.dumps({
                    "format": "int8_convrot",
                    "layers": {
                        "model.diffusion_model.blocks.0.proj": quant_conf,
                    },
                })
            },
        )

        self.assertNotIn("blocks.0.proj.weight_scale", state_dict)
        self.assertTrue(
            torch.equal(
                state_dict["blocks.0.proj.weight"],
                source.to(dtype=torch.float16),
            )
        )

    def test_rejects_int8_source_weight_without_scale(self):
        state_dict = {
            "blocks.0.attn.qkv_proj.weight": torch.ones((2, 2), dtype=torch.int8),
        }

        with self.assertRaisesRegex(ValueError, "missing its scale tensor"):
            materialize_int8_source_weights(state_dict)

    def test_warns_and_skips_unmatched_lora_targets(self):
        targets = {
            "missing.layer": {
                "base_name": "missing.layer.weight",
                "down": torch.ones((1, 2)),
                "up": torch.ones((2, 1)),
                "alpha": 1.0,
            }
        }

        with self.assertLogs(level="WARNING") as logs:
            resolved = resolve_fusion_targets({}, targets)

        self.assertEqual(resolved, {})
        self.assertIn("Skipping 1 LoRA target", logs.output[0])


class OfflineLoraFusionTests(unittest.TestCase):
    def test_imports_and_fuses_safetensors_factor_pair(self):
        down = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        up = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adapter.safetensors"
            save_file(
                {
                    "blocks.0.attn.qkv_proj.lora_A.weight": down,
                    "blocks.0.attn.qkv_proj.lora_B.weight": up,
                    "blocks.0.attn.qkv_proj.alpha": torch.tensor(4.0),
                },
                str(path),
            )
            _, targets, _ = load_lora(path)

        state_dict = {
            "blocks.0.attn.qkv_proj.weight": torch.zeros((3, 2), dtype=torch.float32)
        }
        fuse_targets_into_state_dict(state_dict, targets, strength=0.5, device=torch.device("cpu"))
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [4.0, 6.0]])
        self.assertTrue(torch.equal(state_dict["blocks.0.attn.qkv_proj.weight"], expected))

    @mock.patch(
        "lora._comfy_model_lora_target_map",
        return_value={
            "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj": (
                "diffusion_model.single_blocks.0.linear1.weight"
            )
        },
    )
    def test_fuses_comfy_mapped_flux2_single_block_target(self, _):
        state_dict = {
            "double_blocks.0.img_attn.proj.weight": torch.zeros((2, 2)),
            "single_blocks.0.linear1.weight": torch.zeros((2, 2)),
        }
        targets = {
            "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj": {
                "base_name": (
                    "transformer.single_transformer_blocks.0.attn."
                    "to_qkv_mlp_proj.weight"
                ),
                "down": torch.eye(2),
                "up": torch.eye(2),
                "alpha": 2.0,
            }
        }

        count = fuse_targets_into_state_dict(
            state_dict, targets, strength=1.0, device=torch.device("cpu")
        )

        self.assertEqual(count, 1)
        self.assertTrue(torch.equal(state_dict["single_blocks.0.linear1.weight"], torch.eye(2)))

    @mock.patch(
        "lora._comfy_model_lora_target_map",
        return_value={
            "transformer.transformer_blocks.0.attn.to_q": (
                "diffusion_model.double_blocks.0.img_attn.qkv.weight",
                (0, 0, 2),
            ),
            "transformer.transformer_blocks.0.attn.to_v": (
                "diffusion_model.double_blocks.0.img_attn.qkv.weight",
                (0, 4, 2),
            ),
        },
    )
    def test_fuses_comfy_mapped_qkv_slices(self, _):
        state_dict = {
            "double_blocks.0.img_attn.proj.weight": torch.zeros((2, 2)),
            "single_blocks.0.linear1.weight": torch.zeros((2, 2)),
            "double_blocks.0.img_attn.qkv.weight": torch.zeros((6, 2)),
        }
        targets = {
            "transformer.transformer_blocks.0.attn.to_q": {
                "base_name": "transformer.transformer_blocks.0.attn.to_q.weight",
                "down": torch.eye(2),
                "up": torch.eye(2),
                "alpha": 2.0,
            },
            "transformer.transformer_blocks.0.attn.to_v": {
                "base_name": "transformer.transformer_blocks.0.attn.to_v.weight",
                "down": torch.eye(2),
                "up": torch.eye(2),
                "alpha": 2.0,
            },
        }

        count = fuse_targets_into_state_dict(
            state_dict, targets, strength=1.0, device=torch.device("cpu")
        )

        self.assertEqual(count, 2)
        self.assertTrue(
            torch.equal(
                state_dict["double_blocks.0.img_attn.qkv.weight"],
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
                ),
            )
        )

    @mock.patch(
        "lora._comfy_model_lora_target_map",
        return_value={
            "transformer.transformer_blocks.0.attn.to_q": (
                "diffusion_model.blocks.0.attn.wq.weight"
            )
        },
    )
    def test_fuses_comfy_mapped_krea2_target(self, _):
        state_dict = {
            "blocks.0.attn.wq.weight": torch.zeros((2, 2)),
        }
        targets = {
            "transformer.transformer_blocks.0.attn.to_q": {
                "base_name": "transformer.transformer_blocks.0.attn.to_q.weight",
                "down": torch.eye(2),
                "up": torch.eye(2),
                "alpha": 2.0,
            }
        }

        count = fuse_targets_into_state_dict(
            state_dict, targets, strength=1.0, device=torch.device("cpu")
        )

        self.assertEqual(count, 1)
        self.assertTrue(torch.equal(state_dict["blocks.0.attn.wq.weight"], torch.eye(2)))

    def test_imports_and_fuses_direct_lokr_adapter(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adapter.safetensors"
            save_file(
                {
                    "blocks.0.proj.lokr_w1": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    "blocks.0.proj.lokr_w2": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                    "blocks.0.proj.alpha": torch.tensor(100.0),
                },
                str(path),
            )
            _, targets, metadata = load_lora(path)

        state_dict = {"blocks.0.proj.weight": torch.zeros((4, 4))}
        count = fuse_targets_into_state_dict(
            state_dict, targets, strength=0.5, device=torch.device("cpu")
        )

        self.assertEqual(metadata["target_count"], 1)
        self.assertEqual(count, 1)
        self.assertTrue(
            torch.equal(
                state_dict["blocks.0.proj.weight"],
                0.5 * torch.kron(
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    torch.eye(2),
                ),
            )
        )

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "PyTorch does not support FP8")
    def test_fuses_scaled_fp8_target_as_fp16(self):
        state_dict = {
            "blocks.0.attn.qkv_proj.weight": torch.ones(
                (2, 2), dtype=torch.float8_e4m3fn
            ),
            "blocks.0.attn.qkv_proj.weight_scale": torch.tensor(0.5),
        }
        targets = {
            "blocks.0.attn.qkv_proj": {
                "base_name": "blocks.0.attn.qkv_proj.weight",
                "down": torch.eye(2),
                "up": torch.eye(2),
                "alpha": 2.0,
            }
        }

        fuse_targets_into_state_dict(state_dict, targets, strength=1.0, device=torch.device("cpu"))

        self.assertEqual(state_dict["blocks.0.attn.qkv_proj.weight"].dtype, torch.float16)
        self.assertTrue(
            torch.equal(
                state_dict["blocks.0.attn.qkv_proj.weight"],
                torch.tensor([[1.5, 0.5], [0.5, 1.5]], dtype=torch.float16),
            )
        )

    def test_converter_merges_safetensors_lora_before_gguf_export(self):
        source = {
            "video_patch_proj.weight": torch.zeros((32, 32), dtype=torch.float16),
            "audio_patch_proj.weight": torch.zeros((32, 32), dtype=torch.float16),
            "blocks.0.attn.qkv_proj.weight": torch.zeros((96, 32), dtype=torch.float16),
            "final_layer.video_out.weight": torch.zeros((96, 32), dtype=torch.float16),
        }
        down = torch.zeros((2, 32), dtype=torch.float16)
        down[:, 0] = 1
        up = torch.zeros((96, 2), dtype=torch.float16)
        up[0] = 1

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            lora_path = Path(temp_dir) / "adapter.safetensors"
            output_path = Path(temp_dir) / "merged.gguf"
            save_file(source, str(source_path))
            save_file(
                {
                    "blocks.0.attn.qkv_proj.lora_A.weight": down,
                    "blocks.0.attn.qkv_proj.lora_B.weight": up,
                    "blocks.0.attn.qkv_proj.alpha": torch.tensor(2.0),
                },
                str(lora_path),
            )
            convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="F16",
                lora_paths=[str(lora_path)],
            )
            reader = gguf.GGUFReader(str(output_path))
            tensor = next(
                tensor for tensor in reader.tensors
                if tensor.name == "blocks.0.attn.qkv_proj.weight"
            )
            merged = torch.from_numpy(tensor.data.copy()).view(torch.float16).reshape(
                tuple(reversed(tensor.shape))
            )
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()

        self.assertTrue(torch.equal(merged[0, :2], torch.tensor([2.0, 0.0], dtype=torch.float16)))

    def test_streamed_converter_merges_lora_without_loading_state_dict(self):
        source = {
            "video_patch_proj.weight": torch.zeros((32, 32), dtype=torch.float16),
            "audio_patch_proj.weight": torch.zeros((32, 32), dtype=torch.float16),
            "blocks.0.attn.qkv_proj.weight": torch.zeros((96, 32), dtype=torch.float16),
            "final_layer.video_out.weight": torch.zeros((96, 32), dtype=torch.float16),
        }
        down = torch.zeros((2, 32), dtype=torch.float16)
        down[:, 0] = 1
        up = torch.zeros((96, 2), dtype=torch.float16)
        up[0] = 1

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            lora_path = Path(temp_dir) / "adapter.safetensors"
            output_path = Path(temp_dir) / "merged.gguf"
            save_file(source, str(source_path))
            save_file(
                {
                    "blocks.0.attn.qkv_proj.lora_A.weight": down,
                    "blocks.0.attn.qkv_proj.lora_B.weight": up,
                    "blocks.0.attn.qkv_proj.alpha": torch.tensor(2.0),
                },
                str(lora_path),
            )
            with mock.patch("tools.convert.load_state_dict") as load_state_dict:
                convert_file(
                    str(source_path),
                    str(output_path),
                    interact=False,
                    quant_type_name="F16",
                    lora_paths=[str(lora_path)],
                    streamed=True,
                )
            load_state_dict.assert_not_called()
            reader = gguf.GGUFReader(str(output_path))
            tensor = next(
                tensor for tensor in reader.tensors
                if tensor.name == "blocks.0.attn.qkv_proj.weight"
            )
            merged = torch.from_numpy(tensor.data.copy()).view(torch.float16).reshape(
                tuple(reversed(tensor.shape))
            )
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()

        self.assertTrue(torch.equal(merged[0, :2], torch.tensor([2.0, 0.0], dtype=torch.float16)))


def synthetic_qwen3vl_layout():
    language_shapes = {
        f"{QWEN3VL_SOURCE_PREFIX}embed_tokens.weight": (32, 256),
        f"{QWEN3VL_SOURCE_PREFIX}norm.weight": (256,),
        f"{QWEN3VL_SOURCE_PREFIX}layers.0.input_layernorm.weight": (256,),
        f"{QWEN3VL_SOURCE_PREFIX}layers.0.post_attention_layernorm.weight": (256,),
        f"{QWEN3VL_SOURCE_PREFIX}layers.0.self_attn.k_norm.weight": (128,),
        f"{QWEN3VL_SOURCE_PREFIX}layers.0.self_attn.q_norm.weight": (128,),
        f"{QWEN3VL_SOURCE_PREFIX}layers.0.self_attn.q_proj.weight": (8, 256),
        f"{QWEN3VL_SOURCE_PREFIX}layers.35.self_attn.q_proj.weight": (8, 256),
    }
    visual_shapes = {
        f"{QWEN3VL_VISUAL_PREFIX}patch_embed.proj.weight": (4, 3, 2, 4, 4),
        f"{QWEN3VL_VISUAL_PREFIX}pos_embed.weight": (16, 4),
        f"{QWEN3VL_VISUAL_PREFIX}blocks.0.attn.qkv.weight": (12, 4),
        f"{QWEN3VL_VISUAL_PREFIX}merger.linear_fc2.weight": (8, 16),
        f"{QWEN3VL_VISUAL_PREFIX}deepstack_merger_list.0.norm.weight": (16,),
        f"{QWEN3VL_VISUAL_PREFIX}deepstack_merger_list.2.linear_fc2.weight": (8, 16),
    }
    state_dict = OrderedDict(
        (key, torch.ones(shape, dtype=torch.bfloat16))
        for key, shape in (*language_shapes.items(), *visual_shapes.items())
    )
    return language_shapes, visual_shapes, state_dict


class Qwen3VLConverterTests(unittest.TestCase):
    def layout_patches(self, language_shapes, visual_shapes):
        return mock.patch.multiple(
            converter,
            QWEN3VL_LANGUAGE_SHAPES=language_shapes,
            QWEN3VL_VISUAL_SIGNATURE_SHAPES=visual_shapes,
            QWEN3VL_SOURCE_LANGUAGE_TENSOR_COUNT=len(language_shapes),
            QWEN3VL_SOURCE_VISUAL_TENSOR_COUNT=len(visual_shapes),
            QWEN3VL_SOURCE_TENSOR_COUNT=len(language_shapes) + len(visual_shapes),
        )

    def test_recognizes_and_converts_bf16_source_to_text_only_q8_cr(self):
        language_shapes, visual_shapes, source = synthetic_qwen3vl_layout()
        with self.layout_patches(language_shapes, visual_shapes):
            self.assertIsInstance(detect_arch(source), ModelQwen3VL)
            normalized = normalize_qwen3vl_state_dict(source)
            self.assertIsInstance(normalized, SourceStateDict)
            self.assertEqual(normalized.architecture_hint, "qwen3vl")
            model_arch = detect_arch(normalized)
            self.assertIsInstance(model_arch, ModelQwen3VL)
            self.assertEqual(model_arch.arch, "qwen3vl")

            with TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "qwen3vl-4b-Q8_CR.gguf"
                convert_state_dict(
                    source,
                    str(output_path),
                    quant_type_name="Q8_CR",
                    quantization_device="cpu",
                )
                reader = gguf.GGUFReader(str(output_path))
                tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
                metadata = load_gguf_loader().get_gguf_metadata(reader)
                text_only = metadata[QWEN3VL_TEXT_ONLY_METADATA]
                variant = metadata[QWEN3VL_VARIANT_METADATA]
                reader.tensors.clear()
                reader.fields.clear()
                reader.data._mmap.close()
                del reader

        self.assertEqual(tensor_types["token_embd.weight"], gguf.GGMLQuantizationType.BF16)
        self.assertEqual(tensor_types["output_norm.weight"], gguf.GGMLQuantizationType.F32)
        self.assertEqual(tensor_types["blk.0.attn_q.weight"], gguf.GGMLQuantizationType.I8)
        self.assertEqual(tensor_types["blk.35.attn_q.weight"], gguf.GGMLQuantizationType.I8)
        self.assertFalse(any(name.startswith("model.visual.") for name in tensor_types))
        self.assertTrue(text_only)
        self.assertEqual(variant, "4b")

    def test_streamed_safetensors_conversion_emits_qwen3vl_metadata(self):
        language_shapes, visual_shapes, source = synthetic_qwen3vl_layout()
        with self.layout_patches(language_shapes, visual_shapes), TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "qwen3vl-4b.safetensors"
            output_path = Path(temp_dir) / "qwen3vl-4b-Q8_CR.gguf"
            save_file(source, str(source_path))

            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q8_CR",
                quantization_device="cpu",
                streamed=True,
            )
            reader = gguf.GGUFReader(converted_path)
            metadata = load_gguf_loader().get_gguf_metadata(reader)
            tensor_names = {tensor.name for tensor in reader.tensors}
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()

        self.assertTrue(metadata[QWEN3VL_TEXT_ONLY_METADATA])
        self.assertEqual(metadata[QWEN3VL_VARIANT_METADATA], "4b")
        self.assertIn("blk.35.attn_q.weight", tensor_names)
        self.assertFalse(any(name.startswith("model.visual.") for name in tensor_names))

    def test_qwen3_source_is_not_misdetected_as_qwen3vl(self):
        source = OrderedDict(
            (
                ("layers.0.self_attn.q_proj.weight", torch.ones((8, 256))),
                ("layers.0.self_attn.k_norm.weight", torch.ones((128,))),
                ("layers.0.mlp.gate_proj.weight", torch.ones((8, 256))),
            )
        )

        normalized = normalize_source_state_dict(source)

        self.assertIsInstance(detect_arch(normalized), ModelQwen3)

    def test_rejects_non_bf16_and_incompatible_or_text_only_sources(self):
        language_shapes, visual_shapes, source = synthetic_qwen3vl_layout()
        with self.layout_patches(language_shapes, visual_shapes):
            mixed_dtype = source.copy()
            key = f"{QWEN3VL_SOURCE_PREFIX}layers.35.self_attn.q_proj.weight"
            mixed_dtype[key] = mixed_dtype[key].to(torch.float16)
            with self.assertRaisesRegex(ValueError, "BF16 tensors only"):
                normalize_qwen3vl_state_dict(mixed_dtype)

            text_only = OrderedDict(
                (key, value) for key, value in source.items()
                if not key.startswith(QWEN3VL_VISUAL_PREFIX)
            )
            with self.assertRaisesRegex(ValueError, "both language and visual tensors"):
                normalize_qwen3vl_state_dict(text_only)

            incompatible = OrderedDict(source)
            incompatible["model.language_model.unexpected.weight"] = torch.ones(
                (1,), dtype=torch.bfloat16
            )
            with self.assertRaisesRegex(ValueError, "official Qwen3-VL 4B or 8B BF16 checkpoint"):
                normalize_qwen3vl_state_dict(incompatible)

    def test_rejects_other_quantizers_for_qwen3vl(self):
        language_shapes, visual_shapes, source = synthetic_qwen3vl_layout()
        with self.layout_patches(language_shapes, visual_shapes):
            with self.assertRaisesRegex(ValueError, "only as a text-only Q8_CR GGUF"):
                convert_state_dict(
                    source,
                    "unused.gguf",
                    quant_type_name="Q4_0",
                    quantization_device="cpu",
                )

    def synthetic_8b_layout(self):
        """Sparse stand-in for an official Qwen3-VL 8B checkpoint (same key layout)."""
        language_shapes = {
            f"{QWEN3VL_SOURCE_PREFIX}embed_tokens.weight": (32, 512),
            f"{QWEN3VL_SOURCE_PREFIX}norm.weight": (512,),
            f"{QWEN3VL_SOURCE_PREFIX}layers.0.input_layernorm.weight": (512,),
            f"{QWEN3VL_SOURCE_PREFIX}layers.0.post_attention_layernorm.weight": (512,),
            f"{QWEN3VL_SOURCE_PREFIX}layers.0.self_attn.k_norm.weight": (128,),
            f"{QWEN3VL_SOURCE_PREFIX}layers.0.self_attn.q_norm.weight": (128,),
            f"{QWEN3VL_SOURCE_PREFIX}layers.0.self_attn.q_proj.weight": (16, 512),
            f"{QWEN3VL_SOURCE_PREFIX}layers.0.self_attn.k_proj.weight": (4, 512),
            f"{QWEN3VL_SOURCE_PREFIX}layers.0.self_attn.v_proj.weight": (4, 512),
            f"{QWEN3VL_SOURCE_PREFIX}layers.0.self_attn.o_proj.weight": (512, 16),
            f"{QWEN3VL_SOURCE_PREFIX}layers.0.mlp.gate_proj.weight": (1024, 512),
            f"{QWEN3VL_SOURCE_PREFIX}layers.0.mlp.up_proj.weight": (1024, 512),
            f"{QWEN3VL_SOURCE_PREFIX}layers.0.mlp.down_proj.weight": (512, 1024),
            f"{QWEN3VL_SOURCE_PREFIX}layers.35.self_attn.q_proj.weight": (16, 512),
        }
        visual_shapes = {
            f"{QWEN3VL_VISUAL_PREFIX}patch_embed.proj.weight": (6, 3, 2, 4, 4),
            f"{QWEN3VL_VISUAL_PREFIX}pos_embed.weight": (16, 6),
            f"{QWEN3VL_VISUAL_PREFIX}blocks.0.attn.qkv.weight": (18, 6),
            f"{QWEN3VL_VISUAL_PREFIX}blocks.26.attn.qkv.weight": (18, 6),
            f"{QWEN3VL_VISUAL_PREFIX}merger.linear_fc2.weight": (16, 24),
            f"{QWEN3VL_VISUAL_PREFIX}deepstack_merger_list.0.norm.weight": (24,),
            f"{QWEN3VL_VISUAL_PREFIX}deepstack_merger_list.2.linear_fc2.weight": (16, 24),
        }
        state_dict = OrderedDict(
            (key, torch.ones(shape, dtype=torch.bfloat16))
            for key, shape in (*language_shapes.items(), *visual_shapes.items())
        )
        return language_shapes, visual_shapes, state_dict

    def layout_patches_8b(self, language_shapes, visual_shapes):
        return mock.patch.multiple(
            converter,
            QWEN3VL_8B_LANGUAGE_SHAPES=language_shapes,
            QWEN3VL_8B_VISUAL_SIGNATURE_SHAPES=visual_shapes,
            QWEN3VL_8B_SOURCE_LANGUAGE_TENSOR_COUNT=len(language_shapes),
            QWEN3VL_8B_SOURCE_VISUAL_TENSOR_COUNT=len(visual_shapes),
            QWEN3VL_8B_SOURCE_TENSOR_COUNT=len(language_shapes) + len(visual_shapes),
        )

    def test_recognizes_and_converts_8b_source_to_text_only_q8_cr(self):
        language_shapes, visual_shapes, source = self.synthetic_8b_layout()
        with self.layout_patches_8b(language_shapes, visual_shapes):
            self.assertIsInstance(detect_arch(source), ModelQwen3VL)
            normalized = normalize_qwen3vl_state_dict(source)
            self.assertIsInstance(normalized, SourceStateDict)
            self.assertEqual(normalized.qwen3vl_variant, "8b")
            model_arch = detect_arch(normalized)
            self.assertIsInstance(model_arch, ModelQwen3VL)
            self.assertEqual(model_arch.qwen3vl_variant, "8b")

            with TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "qwen3vl-8b-Q8_CR.gguf"
                convert_state_dict(
                    source,
                    str(output_path),
                    quant_type_name="Q8_CR",
                    quantization_device="cpu",
                )
                reader = gguf.GGUFReader(str(output_path))
                tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
                metadata = load_gguf_loader().get_gguf_metadata(reader)
                reader.tensors.clear()
                reader.fields.clear()
                reader.data._mmap.close()
                del reader

        self.assertTrue(metadata[QWEN3VL_TEXT_ONLY_METADATA])
        self.assertEqual(metadata[QWEN3VL_VARIANT_METADATA], "8b")
        self.assertEqual(
            tensor_types["blk.0.attn_q.weight"], gguf.GGMLQuantizationType.I8
        )
        self.assertFalse(any(name.startswith("model.visual.") for name in tensor_types))

    def test_rejects_qwen3vl_hint_without_a_detected_variant(self):
        # A hand-built SourceStateDict may carry the qwen3vl hint without ever
        # going through normalize_qwen3vl_state_dict, so no size was detected.
        # Guessing a variant would stamp the GGUF with the wrong size.
        state_dict = SourceStateDict(
            [("blk.0.attn_q.weight", torch.ones((4, 4), dtype=torch.bfloat16))],
            architecture_hint="qwen3vl",
        )
        with self.assertRaisesRegex(ValueError, "did not report its size"):
            detect_arch(state_dict)

    def test_rejects_a_layout_mixing_the_two_sizes(self):
        language_shapes, visual_shapes, source = synthetic_qwen3vl_layout()
        mixed = OrderedDict(source)
        mixed[f"{QWEN3VL_VISUAL_PREFIX}blocks.26.attn.qkv.weight"] = torch.ones(
            (12, 4), dtype=torch.bfloat16
        )
        with self.layout_patches(language_shapes, visual_shapes):
            with self.assertRaisesRegex(
                ValueError, "tensor counts differ from the official Qwen3-VL layouts"
            ):
                normalize_qwen3vl_state_dict(mixed)

    def test_8b_tables_match_the_measured_official_checkpoint_layout(self):
        language = converter.QWEN3VL_8B_LANGUAGE_SHAPES
        visual = converter.QWEN3VL_8B_VISUAL_SIGNATURE_SHAPES
        self.assertEqual(
            len(language), converter.QWEN3VL_8B_SOURCE_LANGUAGE_TENSOR_COUNT
        )
        self.assertEqual(
            converter.QWEN3VL_8B_SOURCE_TENSOR_COUNT
            - converter.QWEN3VL_8B_SOURCE_LANGUAGE_TENSOR_COUNT,
            converter.QWEN3VL_8B_SOURCE_VISUAL_TENSOR_COUNT,
        )
        # 4B and 8B share one language key set; only the dimensions differ.
        self.assertEqual(
            set(language), set(converter.QWEN3VL_LANGUAGE_SHAPES)
        )
        self.assertEqual(
            language[f"{QWEN3VL_SOURCE_PREFIX}embed_tokens.weight"], (151936, 4096)
        )
        self.assertEqual(
            language[f"{QWEN3VL_SOURCE_PREFIX}layers.35.self_attn.k_proj.weight"],
            (1024, 4096),
        )
        self.assertEqual(
            language[f"{QWEN3VL_SOURCE_PREFIX}layers.35.mlp.down_proj.weight"],
            (4096, 12288),
        )
        self.assertEqual(
            visual[f"{QWEN3VL_VISUAL_PREFIX}patch_embed.proj.weight"],
            (1152, 3, 2, 16, 16),
        )
        self.assertEqual(
            visual[f"{QWEN3VL_VISUAL_PREFIX}blocks.0.attn.qkv.weight"], (3456, 1152)
        )
        self.assertEqual(
            visual[f"{QWEN3VL_VISUAL_PREFIX}merger.linear_fc2.weight"], (4096, 4608)
        )
        self.assertEqual(
            visual[f"{QWEN3VL_VISUAL_PREFIX}deepstack_merger_list.0.norm.weight"],
            (4608,),
        )


class Qwen3VLNativeOpsTests(unittest.TestCase):
    def test_native_kernel_error_retries_portable_dequantized_linear(self):
        ops_module = ops_module_factory()
        source = torch.randn((6, 64), dtype=torch.bfloat16)
        qdata, scale, quant_conf, shape = quantize_int8_convrot(
            source,
            convrot_groupsize=16,
            device=torch.device("cpu"),
        )
        params = comfy.quant_ops.TensorWiseINT8Layout.Params(
            scale=scale,
            orig_dtype=torch.bfloat16,
            orig_shape=shape,
            convrot=quant_conf["convrot"],
            convrot_groupsize=quant_conf["convrot_groupsize"],
        )
        qweight = comfy.quant_ops.QuantizedTensor(
            qdata,
            "TensorWiseINT8Layout",
            params,
        )
        operations = ops_module.get_gguf_q8_ops(compute_dtype=torch.bfloat16)
        layer = operations.Linear(64, 6, bias=True, device="cpu", dtype=torch.bfloat16)
        layer.weight = torch.nn.Parameter(qweight, requires_grad=False)
        layer.bias = torch.nn.Parameter(torch.randn(6, dtype=torch.bfloat16), requires_grad=False)
        layer._gguf_q8_runtime_fallback = True
        inputs = torch.randn((2, 3, 64), dtype=torch.bfloat16)
        dense_weight = ops_module._dequantize_q8_weight_portable(qweight, torch.bfloat16)
        expected = torch.nn.functional.linear(inputs, dense_weight, layer.bias)
        parent_linear = type(layer).__mro__[1]
        ops_module.q8_execution_stats(reset=True)

        with mock.patch.object(
            parent_linear,
            "forward",
            side_effect=RuntimeError("simulated TensorWiseINT8 kernel failure"),
        ):
            actual = layer(inputs)
            repeated = layer(inputs)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(repeated, expected, atol=1e-3, rtol=1e-3))
        self.assertEqual(
            ops_module.q8_execution_stats(),
            {"native_calls": 0, "fallback_calls": 2},
        )


class Qwen3VLLoaderRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = nodes_factory()

    def test_legacy_static_loader_rejects_q8_cr_with_dynamic_loader_guidance(self):
        progress = mock.Mock()
        with mock.patch.object(
            self.nodes,
            "GGUFLoadProgress",
            return_value=progress,
        ), mock.patch.object(
            self.nodes,
            "gguf_clip_loader",
            return_value=(
                {},
                {
                    "arch_str": "qwen3vl",
                    "qwen3vl_text_only": True,
                    "gguf_quant_formats": frozenset({"int8_tensorwise"}),
                },
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Q8_CR text encoders require CLIPLoaderGGUFDynamicVRAM",
            ):
                self.nodes.CLIPLoaderGGUF().load_data(["qwen3vl-q8.gguf"])

    def test_prunes_vision_tower_and_rejects_image_embeds(self):
        class Qwen3VLFake(torch.nn.Module):
            model_type = "qwen3vl_4b"

            def __init__(self):
                super().__init__()
                self.visual = torch.nn.Linear(2, 2)

        wrapper = torch.nn.Module()
        wrapper.qwen3vl = Qwen3VLFake()
        clip = type("Clip", (), {"cond_stage_model": wrapper})()

        self.nodes._prune_qwen3vl_vision(clip)

        self.assertIsNone(wrapper.qwen3vl.visual)
        with self.assertRaisesRegex(ValueError, "text-only and does not support image inputs"):
            wrapper.qwen3vl.preprocess_embed({"type": "image"}, device=torch.device("cpu"))

    def test_prunes_vision_tower_of_the_8b_text_encoder(self):
        class Qwen3VL8BFake(torch.nn.Module):
            model_type = "qwen3vl_8b"

            def __init__(self):
                super().__init__()
                self.visual = torch.nn.Linear(2, 2)

        wrapper = torch.nn.Module()
        wrapper.qwen3vl = Qwen3VL8BFake()
        clip = type("Clip", (), {"cond_stage_model": wrapper})()

        self.nodes._prune_qwen3vl_vision(clip)

        self.assertIsNone(wrapper.qwen3vl.visual)
        with self.assertRaisesRegex(ValueError, "text-only and does not support image inputs"):
            wrapper.qwen3vl.preprocess_embed({"type": "image"}, device=torch.device("cpu"))

    def test_ampere_feature_probe_selects_tensorwise_int8(self):
        with mock.patch.object(
            self.nodes.comfy.model_management,
            "get_torch_device",
            return_value=torch.device("cuda:0"),
        ), mock.patch.object(self.nodes.torch.cuda, "is_available", return_value=True), mock.patch.object(
            self.nodes.torch.cuda,
            "get_device_capability",
            return_value=(8, 6),
        ), mock.patch.object(
            self.nodes.comfy.quant_ops.TensorWiseINT8Layout,
            "supports_fast_matmul",
            return_value=True,
        ):
            selected, reason = self.nodes._qwen3vl_native_q8_device()

        self.assertEqual(selected, (torch.device("cuda:0"), (8, 6)))
        self.assertIsNone(reason)


class Qwen3VLDetectionMarkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = load_gguf_loader()

    def test_uses_minimax_32b_detection_marker_for_5120_hidden_size(self):
        state_dict = {
            "model.layers.0.input_layernorm.weight": torch.zeros(5120),
            "model.layers.49.self_attn.q_proj.weight": torch.zeros(1),
        }

        self.loader.inject_qwen3vl_detection_markers(state_dict)

        self.assertEqual(
            comfy.sd.detect_te_model(state_dict),
            comfy.sd.TEModel.QWEN3VL_32B,
        )
        self.assertIn("visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertNotIn("model.visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertNotIn("model.visual.merger.linear_fc2.weight", state_dict)
        self.assertEqual(
            state_dict["visual.deepstack_merger_list.0.norm.weight"].shape,
            (4608,),
        )

    def test_uses_model_prefixed_detection_markers_for_8b(self):
        state_dict = {
            "model.layers.0.input_layernorm.weight": torch.zeros(4096),
        }

        self.loader.inject_qwen3vl_detection_markers(state_dict)

        self.assertIn("model.visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertEqual(
            state_dict["model.visual.merger.linear_fc2.weight"].shape,
            (4096, 4608),
        )

    def test_text_only_q8_metadata_skips_mmproj_and_is_returned(self):
        with mock.patch.object(
            self.loader,
            "gguf_sd_loader",
            return_value=(
                {
                    "model.layers.0.input_layernorm.weight": torch.zeros(2560),
                    "blk.0.attn_q.weight": torch.zeros((8, 2560), dtype=torch.int8),
                },
                {
                    "arch_str": "qwen3vl",
                    "metadata": {
                        "comfy.gguf.qwen3vl.text_only": True,
                        "comfy.gguf.qwen3vl.variant": "4b",
                    },
                    "gguf_quant_formats": frozenset({"int8_tensorwise"}),
                },
            ),
        ), mock.patch.object(
            self.loader,
            "gguf_mmproj_loader",
            return_value={},
        ) as mmproj_loader:
            state_dict, extra = self.loader.gguf_clip_loader(
                "qwen3vl-4b-text-only.gguf",
                return_extra=True,
            )

        self.assertTrue(extra["qwen3vl_text_only"])
        self.assertIn("model.visual.deepstack_merger_list.0.norm.weight", state_dict)
        mmproj_loader.assert_not_called()

    def test_text_only_metadata_is_accepted_for_the_8b_variant(self):
        with mock.patch.object(
            self.loader,
            "gguf_sd_loader",
            return_value=(
                {
                    "model.layers.0.input_layernorm.weight": torch.zeros(4096),
                    "blk.0.attn_q.weight": torch.zeros((8, 4096), dtype=torch.int8),
                },
                {
                    "arch_str": "qwen3vl",
                    "metadata": {
                        "comfy.gguf.qwen3vl.text_only": True,
                        "comfy.gguf.qwen3vl.variant": "8b",
                    },
                    "gguf_quant_formats": frozenset({"int8_tensorwise"}),
                },
            ),
        ), mock.patch.object(
            self.loader,
            "gguf_mmproj_loader",
            return_value={},
        ) as mmproj_loader:
            state_dict, extra = self.loader.gguf_clip_loader(
                "qwen3vl-8b-text-only.gguf",
                return_extra=True,
            )

        self.assertTrue(extra["qwen3vl_text_only"])
        self.assertEqual(
            state_dict["model.visual.merger.linear_fc2.weight"].shape, (4096, 4608)
        )
        mmproj_loader.assert_not_called()

    def test_text_only_metadata_is_rejected_for_an_unsupported_variant(self):
        with mock.patch.object(
            self.loader,
            "gguf_sd_loader",
            return_value=(
                {
                    "model.layers.0.input_layernorm.weight": torch.zeros(5120),
                    "blk.0.attn_q.weight": torch.zeros((8, 5120), dtype=torch.int8),
                },
                {
                    "arch_str": "qwen3vl",
                    "metadata": {
                        "comfy.gguf.qwen3vl.text_only": True,
                        "comfy.gguf.qwen3vl.variant": "32b",
                    },
                    "gguf_quant_formats": frozenset({"int8_tensorwise"}),
                },
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "text-only metadata is valid only on a Qwen3-VL 4b or 8b GGUF",
            ):
                self.loader.gguf_clip_loader("qwen3vl-32b-text-only.gguf")

    def test_pruned_32b_clip_loader_injects_marker(self):
        with mock.patch.object(
            self.loader,
            "gguf_sd_loader",
            return_value=(
                {
                    "model.layers.0.input_layernorm.weight": torch.zeros(5120),
                    "model.layers.49.self_attn.q_proj.weight": torch.zeros(1),
                },
                {"arch_str": "qwen3vl"},
            ),
        ), mock.patch.object(
            self.loader,
            "gguf_mmproj_loader",
            return_value={},
        ):
            state_dict = self.loader.gguf_clip_loader("qwen3-vl-pruned.gguf")

        self.assertEqual(
            state_dict["visual.deepstack_merger_list.0.norm.weight"].shape,
            (4608,),
        )

    def test_pruned_32b_clip_loader_maps_matching_mmproj_to_visual_tower(self):
        with mock.patch.object(
            self.loader,
            "gguf_sd_loader",
            return_value=(
                {
                    "model.layers.0.input_layernorm.weight": torch.zeros(5120),
                    "model.layers.49.self_attn.q_proj.weight": torch.zeros(1),
                },
                {"arch_str": "qwen3vl"},
            ),
        ), mock.patch.object(
            self.loader,
            "gguf_mmproj_loader",
            return_value={
                "model.visual.deepstack_merger_list.0.norm.weight": torch.ones(4608),
            },
        ):
            state_dict = self.loader.gguf_clip_loader("qwen3-vl-pruned.gguf")

        self.assertIn("visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertNotIn("model.visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertTrue(
            torch.equal(
                state_dict["visual.deepstack_merger_list.0.norm.weight"],
                torch.ones(4608),
            )
        )

    def test_maps_qwen3vl_mmproj_deepstack_tensors(self):
        mapped = self.loader.sd_map_replace(
            {
                "v.deepstack.0.norm.weight": torch.ones(4608),
                "v.deepstast.1.fc2.weight": torch.ones(5120, 4608),
                "v.blk.0.attn_qkv.weight": torch.ones(3, 3),
            },
            self.loader.CLIP_VISION_QWEN3_MAP,
        )

        self.assertIn(
            "model.visual.deepstack_merger_list.0.norm.weight",
            mapped,
        )
        self.assertIn(
            "model.visual.deepstack_merger_list.1.linear_fc2.weight",
            mapped,
        )
        self.assertIn("model.visual.blocks.0.attn.qkv.weight", mapped)

    def test_qwen3vl_mmproj_stacks_temporal_patch_embeddings(self):
        with TemporaryDirectory() as temp_dir:
            text_encoder = Path(temp_dir) / "qwen3-vl-IQ3_XXS.gguf"
            mmproj = Path(temp_dir) / "qwen3-vl-mmproj-BF16.gguf"
            text_encoder.touch()
            mmproj.touch()
            patch_a = torch.ones((2, 3, 2, 2))
            patch_b = torch.full((2, 3, 2, 2), 2.0)

            with mock.patch.object(
                self.loader,
                "gguf_sd_loader",
                return_value=(
                    {
                        "v.patch_embd.weight": patch_a,
                        "v.patch_embd.weight.1": patch_b,
                        "v.deepstast.0.norm.weight": torch.ones(8),
                    },
                    {},
                ),
            ):
                mapped = self.loader.gguf_mmproj_loader(str(text_encoder))

        weight = mapped["model.visual.patch_embed.proj.weight"]
        self.assertEqual(weight.shape, (2, 3, 2, 2, 2))
        self.assertTrue(torch.equal(weight[:, :, 0], patch_a))
        self.assertTrue(torch.equal(weight[:, :, 1], patch_b))

    def test_qwen3vl_mmproj_lookup_matches_hyphenated_encoder_name(self):
        with TemporaryDirectory() as temp_dir:
            text_encoder = Path(temp_dir) / "Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
            matching_mmproj = Path(temp_dir) / "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
            other_variant = Path(temp_dir) / "mmproj-Qwen3VL-4B-Instruct-F16.gguf"
            text_encoder.touch()
            matching_mmproj.touch()
            other_variant.touch()

            with mock.patch.object(
                self.loader,
                "gguf_sd_loader",
                return_value=({"v.deepstack.0.norm.weight": torch.ones(8)}, {}),
            ) as sd_loader:
                mapped = self.loader.gguf_mmproj_loader(str(text_encoder))

        sd_loader.assert_called_once_with(str(matching_mmproj), is_text_model=True)
        self.assertIn("model.visual.deepstack_merger_list.0.norm.weight", mapped)

    def test_qwen3vl_mmproj_lookup_rejects_ambiguous_alias_matches(self):
        with TemporaryDirectory() as temp_dir:
            text_encoder = Path(temp_dir) / "Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
            text_encoder.touch()
            for suffix in ("F16", "BF16"):
                (Path(temp_dir) / f"mmproj-Qwen3VL-8B-Instruct-{suffix}.gguf").touch()

            with mock.patch.object(self.loader, "gguf_sd_loader") as sd_loader:
                with self.assertRaisesRegex(ValueError, "Ambiguous Qwen3-VL mmproj"):
                    self.loader.gguf_mmproj_loader(str(text_encoder))

        sd_loader.assert_not_called()


class Qwen35GGUFLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = load_gguf_loader()

    def test_qwen35_arch_is_whitelisted(self):
        self.assertIn("qwen35", self.loader.TXT_ARCH_LIST)

    def test_maps_qwen35_tensor_layout_to_comfyui(self):
        state_dict = {
            "token_embd.weight": torch.zeros(1),
            "output_norm.weight": torch.zeros(1),
            "output.weight": torch.zeros(1),
            "blk.0.attn_norm.weight": torch.zeros(1),
            "blk.0.post_attention_norm.weight": torch.zeros(1),
            "blk.0.attn_qkv.weight": torch.zeros(1),
            "blk.0.attn_gate.weight": torch.zeros(1),
            "blk.0.ssm_a": torch.zeros(1),
            "blk.0.ssm_dt.bias": torch.zeros(1),
            "blk.0.ssm_alpha.weight": torch.zeros(1),
            "blk.0.ssm_beta.weight": torch.zeros(1),
            "blk.0.ssm_conv1d.weight": torch.zeros(1),
            "blk.0.ssm_norm.weight": torch.zeros(1),
            "blk.0.ssm_out.weight": torch.zeros(1),
            "blk.3.attn_q.weight": torch.zeros(1),
            "blk.3.attn_k.weight": torch.zeros(1),
            "blk.3.attn_v.weight": torch.zeros(1),
            "blk.3.attn_output.weight": torch.zeros(1),
            "blk.3.attn_q_norm.weight": torch.zeros(1),
            "blk.3.attn_k_norm.weight": torch.zeros(1),
            "blk.3.ffn_up.weight": torch.zeros(1),
            "blk.3.ffn_gate.weight": torch.zeros(1),
            "blk.3.ffn_down.weight": torch.zeros(1),
        }

        mapped = self.loader.sd_map_replace(state_dict, self.loader.QWEN35_SD_MAP)

        self.assertIn("model.language_model.embed_tokens.weight", mapped)
        self.assertIn("model.language_model.norm.weight", mapped)
        self.assertIn("lm_head.weight", mapped)
        self.assertIn("model.language_model.layers.0.input_layernorm.weight", mapped)
        self.assertIn(
            "model.language_model.layers.0.post_attention_layernorm.weight",
            mapped,
        )
        self.assertIn("model.language_model.layers.0.linear_attn.A_log", mapped)
        self.assertIn(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", mapped
        )
        self.assertIn(
            "model.language_model.layers.0.linear_attn.in_proj_z.weight", mapped
        )
        self.assertIn(
            "model.language_model.layers.0.linear_attn.in_proj_a.weight", mapped
        )
        self.assertIn(
            "model.language_model.layers.0.linear_attn.in_proj_b.weight", mapped
        )
        self.assertIn(
            "model.language_model.layers.0.linear_attn.dt_bias", mapped
        )
        self.assertIn(
            "model.language_model.layers.0.linear_attn.conv1d.weight", mapped
        )
        self.assertIn(
            "model.language_model.layers.0.linear_attn.norm.weight", mapped
        )
        self.assertIn(
            "model.language_model.layers.0.linear_attn.out_proj.weight", mapped
        )
        self.assertIn("model.language_model.layers.3.self_attn.q_proj.weight", mapped)
        self.assertIn("model.language_model.layers.3.self_attn.k_proj.weight", mapped)
        self.assertIn("model.language_model.layers.3.self_attn.v_proj.weight", mapped)
        self.assertIn("model.language_model.layers.3.self_attn.o_proj.weight", mapped)
        self.assertIn("model.language_model.layers.3.self_attn.q_norm.weight", mapped)
        self.assertIn("model.language_model.layers.3.self_attn.k_norm.weight", mapped)
        self.assertIn("model.language_model.layers.3.mlp.up_proj.weight", mapped)
        self.assertIn("model.language_model.layers.3.mlp.gate_proj.weight", mapped)
        self.assertIn("model.language_model.layers.3.mlp.down_proj.weight", mapped)

    def test_qwen35_layout_is_detected_by_installed_comfyui(self):
        for hidden_size, expected in (
            (1024, comfy.sd.TEModel.QWEN35_08B),
            (2048, comfy.sd.TEModel.QWEN35_2B),
            (2560, comfy.sd.TEModel.QWEN35_4B),
            (4096, comfy.sd.TEModel.QWEN35_9B),
            (5120, comfy.sd.TEModel.QWEN35_27B),
        ):
            with self.subTest(hidden_size=hidden_size):
                state_dict = {
                    "model.language_model.layers.0.linear_attn.A_log": torch.zeros(1),
                    "model.language_model.layers.0.input_layernorm.weight": torch.zeros(hidden_size),
                }
                self.assertEqual(
                    comfy.sd.detect_te_model(state_dict),
                    expected,
                )

    def test_clip_loader_corrects_norms_and_alog(self):
        norm_stored = torch.full((2560,), 2.0)  # llama.cpp stores w + 1
        alog_stored = torch.full((32,), -2.0)  # llama.cpp stores -exp(A_log)

        with mock.patch.object(
            self.loader,
            "gguf_sd_loader",
            return_value=(
                {
                    "token_embd.weight": torch.zeros((248320, 2560)),
                    "output_norm.weight": norm_stored.clone(),
                    "blk.0.attn_norm.weight": norm_stored.clone(),
                    "blk.0.post_attention_norm.weight": norm_stored.clone(),
                    "blk.0.ssm_norm.weight": torch.ones(128),
                    "blk.0.ssm_a": alog_stored.clone(),
                    "blk.0.attn_qkv.weight": torch.zeros((8192, 2560)),
                    "blk.3.attn_q_norm.weight": norm_stored.clone(),
                    "blk.3.attn_k_norm.weight": norm_stored.clone(),
                },
                {"arch_str": "qwen35"},
            ),
        ), mock.patch.object(
            self.loader,
            "gguf_mmproj_loader",
            return_value={},
        ):
            state_dict = self.loader.gguf_clip_loader("Qwen3.5-4B-BF16.gguf")

        expected_log = torch.log(torch.tensor(2.0))
        self.assertTrue(
            torch.equal(
                state_dict["model.language_model.norm.weight"],
                torch.ones(2560),
            )
        )
        self.assertTrue(
            torch.equal(
                state_dict["model.language_model.layers.0.input_layernorm.weight"],
                torch.ones(2560),
            )
        )
        self.assertTrue(
            torch.equal(
                state_dict["model.language_model.layers.0.post_attention_layernorm.weight"],
                torch.ones(2560),
            )
        )
        self.assertTrue(
            torch.equal(
                state_dict["model.language_model.layers.3.self_attn.q_norm.weight"],
                torch.ones(2560),
            )
        )
        self.assertTrue(
            torch.equal(
                state_dict["model.language_model.layers.3.self_attn.k_norm.weight"],
                torch.ones(2560),
            )
        )
        # linear_attn.norm is RMSNormGated and must NOT be shifted.
        self.assertTrue(
            torch.equal(
                state_dict["model.language_model.layers.0.linear_attn.norm.weight"],
                torch.ones(128),
            )
        )
        # A_log is inverted back from -exp(A_log).
        self.assertTrue(
            torch.allclose(
                state_dict["model.language_model.layers.0.linear_attn.A_log"],
                torch.full((32,), expected_log),
            )
        )

    def test_conv1d_kernel_is_unsqueezed_to_depthwise_shape(self):
        # V channels stored tiled so the corrected kernel equals arange()
        tiled = [i * 2 for i in range(16)] + [i * 2 + 1 for i in range(16)]
        with mock.patch.object(
            self.loader,
            "gguf_sd_loader",
            return_value=(
                {
                    "blk.0.attn_qkv.weight": torch.zeros((8192, 2560)),
                    "blk.0.attn_gate.weight": torch.zeros((4096, 2560)),
                    "blk.0.ssm_a": torch.full((32,), -1.0),
                    "blk.0.ssm_conv1d.weight": torch.cat(
                        [
                            torch.arange(4096).unsqueeze(1).repeat(1, 4),
                            (torch.arange(4096) + 4096).reshape(32, 128)[tiled]
                            .reshape(-1)
                            .unsqueeze(1)
                            .repeat(1, 4),
                        ],
                        dim=0,
                    ).to(torch.float32),
                },
                {"arch_str": "qwen35"},
            ),
        ), mock.patch.object(
            self.loader,
            "gguf_mmproj_loader",
            return_value={},
        ):
            state_dict = self.loader.gguf_clip_loader("Qwen3.5-4B-BF16.gguf")

        conv = state_dict["model.language_model.layers.0.linear_attn.conv1d.weight"]
        self.assertEqual(conv.shape, (8192, 1, 4))
        self.assertTrue(
            torch.equal(
                conv[:, 0, :],
                torch.arange(8192).unsqueeze(1).repeat(1, 4),
            )
        )

    def test_reorders_tiled_v_heads_back_to_grouped_order(self):
        # llama.cpp stores V heads tiled for k=2, v=4: [K0_v0, K1_v0, K0_v1, K1_v1]
        # i.e. head order [h0, h2, h1, h3]; ComfyUI expects grouped [h0, h1, h2, h3].
        num_k_heads, num_v_heads, head_dim = 2, 4, 1
        value_dim = num_v_heads * head_dim
        key_dim = num_k_heads * head_dim
        conv_dim = 2 * key_dim + value_dim
        tiled = [0, 2, 1, 3]

        def head_marked(rows, cols=1):
            return torch.arange(rows, dtype=torch.float32).unsqueeze(1).repeat(1, cols)

        # stored rows/cols encode their V-head index in TILED order
        v_stored = head_marked(value_dim, 3)[tiled]
        prefix = "model.language_model.layers.0.linear_attn."
        qkv = torch.cat(
            [head_marked(key_dim, 3), head_marked(key_dim, 3), v_stored], dim=0
        )
        conv = torch.cat(
            [
                torch.arange(2 * key_dim, dtype=torch.float32).unsqueeze(1).repeat(1, 2),
                (torch.arange(value_dim, dtype=torch.float32) + 2 * key_dim)
                [tiled].unsqueeze(1).repeat(1, 2),
            ],
            dim=0,
        )
        sd = {
            prefix + "in_proj_qkv.weight": qkv,
            prefix + "in_proj_z.weight": v_stored,
            prefix + "in_proj_a.weight": head_marked(num_v_heads)[tiled],
            prefix + "in_proj_b.weight": head_marked(num_v_heads)[tiled],
            prefix + "A_log": torch.full((num_v_heads,), -1.0),
            prefix + "dt_bias": torch.arange(num_v_heads, dtype=torch.float32)[tiled],
            prefix + "conv1d.weight": conv,
            prefix + "out_proj.weight": head_marked(3, value_dim)[:, tiled],
            "model.language_model.layers.0.input_layernorm.weight": torch.full((4,), 2.0),
        }

        corrected = self.loader.qwen35_corrections(sd)

        qkv = corrected[prefix + "in_proj_qkv.weight"]
        self.assertTrue(torch.equal(qkv[: 2 * key_dim], head_marked(key_dim, 3).repeat(2, 1)))
        self.assertTrue(torch.equal(qkv[2 * key_dim:], head_marked(value_dim, 3)))
        self.assertTrue(torch.equal(corrected[prefix + "in_proj_z.weight"], head_marked(value_dim, 3)))
        self.assertTrue(torch.equal(corrected[prefix + "in_proj_a.weight"], head_marked(num_v_heads)))
        self.assertTrue(torch.equal(corrected[prefix + "in_proj_b.weight"], head_marked(num_v_heads)))
        self.assertTrue(torch.equal(corrected[prefix + "A_log"], torch.zeros(num_v_heads)))
        self.assertTrue(
            torch.equal(
                corrected[prefix + "dt_bias"],
                torch.arange(num_v_heads, dtype=torch.float32),
            )
        )
        conv = corrected[prefix + "conv1d.weight"]
        self.assertEqual(conv.shape, (conv_dim, 1, 2))
        self.assertTrue(
            torch.equal(
                conv[:, 0, :],
                torch.arange(conv_dim, dtype=torch.float32).unsqueeze(1).repeat(1, 2),
            )
        )
        self.assertTrue(
            torch.equal(corrected[prefix + "out_proj.weight"], head_marked(3, value_dim))
        )
        # unshifted norm
        self.assertTrue(
            torch.equal(
                corrected["model.language_model.layers.0.input_layernorm.weight"],
                torch.ones(4),
            )
        )

    def test_reorders_quantized_tiled_v_heads(self):
        num_k_heads, num_v_heads, head_dim = 2, 4, 1
        value_dim = num_v_heads * head_dim
        tiled = [0, 2, 1, 3]
        v_stored = torch.arange(value_dim, dtype=torch.float32).unsqueeze(1)[tiled]
        q_tensor = self.loader.GGMLTensor(
            v_stored.to(torch.bfloat16),
            tensor_type=gguf.GGMLQuantizationType.BF16,
            tensor_shape=v_stored.shape,
        )
        reordered = self.loader._qwen35_v_reorder(
            q_tensor, num_v_heads, num_k_heads, head_dim
        )
        self.assertFalse(self.loader.is_quantized(reordered))
        self.assertTrue(
            torch.equal(
                reordered.float(),
                torch.arange(value_dim, dtype=torch.float32).unsqueeze(1),
            )
        )

    def test_v_head_reorder_is_identity_for_balanced_heads(self):
        num_k_heads, num_v_heads = 16, 16
        head_dim = 2
        value_dim = num_v_heads * head_dim
        key_dim = num_k_heads * head_dim
        conv_dim = 2 * key_dim + value_dim
        prefix = "model.language_model.layers.0.linear_attn."
        sd = {
            prefix + "in_proj_qkv.weight": torch.arange(conv_dim * 3, dtype=torch.float32).reshape(conv_dim, 3),
            prefix + "in_proj_z.weight": torch.arange(value_dim * 3, dtype=torch.float32).reshape(value_dim, 3),
            prefix + "A_log": torch.full((num_v_heads,), -1.0),
            prefix + "dt_bias": torch.arange(num_v_heads, dtype=torch.float32),
            prefix + "conv1d.weight": torch.arange(conv_dim * 4, dtype=torch.float32).reshape(conv_dim, 4),
            prefix + "out_proj.weight": torch.arange(3 * value_dim, dtype=torch.float32).reshape(3, value_dim),
        }

        corrected = self.loader.qwen35_corrections(sd)

        conv = corrected[prefix + "conv1d.weight"]
        self.assertEqual(conv.shape, (conv_dim, 1, 4))
        self.assertTrue(
            torch.equal(conv[:, 0, :], torch.arange(conv_dim * 4).reshape(conv_dim, 4))
        )
        self.assertTrue(
            torch.equal(
                corrected[prefix + "in_proj_qkv.weight"],
                sd[prefix + "in_proj_qkv.weight"],
            )
        )
        self.assertTrue(
            torch.equal(corrected[prefix + "out_proj.weight"], sd[prefix + "out_proj.weight"])
        )

    def test_clip_loader_dequantizes_quantized_lm_head(self):
        # BaseGenerate.logits() feeds lm_head straight to F.linear without
        # dequantizing GGML tensors, so a Q8_0 head (raw bytes with scales
        # interleaved) must arrive dequantized with its logical shape.
        with mock.patch.object(
            self.loader,
            "gguf_sd_loader",
            return_value=(
                {
                    "token_embd.weight": torch.zeros((248320, 4096)),
                    "output.weight": self.loader.GGMLTensor(
                        torch.zeros((248320, 4352), dtype=torch.uint8),
                        tensor_type=gguf.GGMLQuantizationType.Q8_0,
                        tensor_shape=(248320, 4096),
                    ),
                    "blk.0.attn_qkv.weight": torch.zeros((8192, 4096)),
                    "blk.0.attn_gate.weight": torch.zeros((4096, 4096)),
                    "blk.0.ssm_a": torch.full((32,), -1.0),
                },
                {"arch_str": "qwen35"},
            ),
        ), mock.patch.object(
            self.loader,
            "gguf_mmproj_loader",
            return_value={},
        ):
            state_dict = self.loader.gguf_clip_loader("Qwen3.5-9B-Q8_0.gguf")

        head = state_dict["lm_head.weight"]
        self.assertFalse(self.loader.is_quantized(head))
        self.assertEqual(head.shape, (248320, 4096))

    def test_clip_loader_keeps_bf16_lm_head_quantized(self):
        # BF16 storage keeps its logical shape, so Dynamic VRAM can keep it
        # quantized and offloaded; only block-quantized heads are unsafe.
        bf16_head = self.loader.GGMLTensor(
            torch.zeros((248320, 4096), dtype=torch.bfloat16),
            tensor_type=gguf.GGMLQuantizationType.BF16,
            tensor_shape=(248320, 4096),
        )
        with mock.patch.object(
            self.loader,
            "gguf_sd_loader",
            return_value=(
                {
                    "token_embd.weight": torch.zeros((248320, 4096)),
                    "output.weight": bf16_head,
                    "blk.0.attn_qkv.weight": torch.zeros((8192, 4096)),
                    "blk.0.attn_gate.weight": torch.zeros((4096, 4096)),
                    "blk.0.ssm_a": torch.full((32,), -1.0),
                },
                {"arch_str": "qwen35"},
            ),
        ), mock.patch.object(
            self.loader,
            "gguf_mmproj_loader",
            return_value={},
        ):
            state_dict = self.loader.gguf_clip_loader("Qwen3.5-9B-BF16.gguf")

        head = state_dict["lm_head.weight"]
        self.assertTrue(self.loader.is_quantized(head))
        self.assertEqual(head.shape, (248320, 4096))

    def test_mmproj_routes_fused_qkv_through_qwen3_vision_map(self):
        with TemporaryDirectory() as temp_dir:
            text_encoder = Path(temp_dir) / "Qwen3.5-4B-Q8_0.gguf"
            mmproj = Path(temp_dir) / "mmproj-Qwen3.5-4B-BF16.gguf"
            text_encoder.touch()
            mmproj.touch()

            with mock.patch.object(
                self.loader,
                "gguf_sd_loader",
                return_value=(
                    {
                        "v.blk.0.attn_qkv.weight": torch.ones((1024, 3072)),
                        "v.blk.0.attn_out.weight": torch.ones((1024, 1024)),
                        "v.blk.0.ln1.weight": torch.ones(1024),
                        "v.blk.0.ln2.weight": torch.ones(1024),
                        "v.blk.0.ffn_up.weight": torch.ones((1024, 4096)),
                        "v.blk.0.ffn_down.weight": torch.ones((4096, 1024)),
                        "v.patch_embd.weight": torch.ones((2, 3, 2, 2)),
                        "v.patch_embd.weight.1": torch.full((2, 3, 2, 2), 2.0),
                        "v.patch_embd.bias": torch.ones(1024),
                        "v.position_embd.weight": torch.ones((1024, 2304)),
                        "mm.0.weight": torch.ones(1),
                        "mm.2.weight": torch.ones(1),
                        "v.post_ln.weight": torch.ones(1024),
                    },
                    {},
                ),
            ):
                mapped = self.loader.gguf_mmproj_loader(str(text_encoder))

        self.assertIn("model.visual.blocks.0.attn.qkv.weight", mapped)
        self.assertIn("model.visual.blocks.0.attn.proj.weight", mapped)
        self.assertIn("model.visual.blocks.0.norm1.weight", mapped)
        self.assertIn("model.visual.blocks.0.norm2.weight", mapped)
        self.assertIn("model.visual.blocks.0.mlp.linear_fc1.weight", mapped)
        self.assertIn("model.visual.blocks.0.mlp.linear_fc2.weight", mapped)
        self.assertIn("model.visual.patch_embed.proj.weight", mapped)
        self.assertIn("model.visual.patch_embed.proj.bias", mapped)
        self.assertIn("visual.pos_embed.weight", mapped)
        self.assertIn("model.visual.merger.linear_fc1.weight", mapped)
        self.assertIn("model.visual.merger.linear_fc2.weight", mapped)
        self.assertIn("model.visual.merger.norm.weight", mapped)
        self.assertEqual(
            mapped["model.visual.patch_embed.proj.weight"].shape,
            (2, 3, 2, 2, 2),
        )


class Qwen3VLQuantizationTests(unittest.TestCase):
    def test_supports_quant_types_used_by_pruned_32b_gguf(self):
        for quant_name in ("IQ3_S", "IQ3_XXS", "IQ2_S", "IQ2_XS"):
            quant_type = getattr(gguf.GGMLQuantizationType, quant_name)
            _, type_size = gguf.GGML_QUANT_SIZES[quant_type]
            output = dequantize(
                torch.zeros((1, type_size), dtype=torch.uint8),
                quant_type,
                (256,),
                dtype=torch.float32,
            )

            self.assertEqual(output.shape, (256,))
            self.assertEqual(output.dtype, torch.float32)
            self.assertIn(quant_type, dequantize_functions)


class Gemma4GGUFLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = load_gguf_loader()

    def test_maps_e4b_specific_tensor_layout_to_comfyui(self):
        state_dict = {
            "token_embd.weight": torch.zeros(1),
            "per_layer_token_embd.weight": torch.zeros(1),
            "per_layer_model_proj.weight": torch.zeros(1),
            "per_layer_proj_norm.weight": torch.zeros(1),
            "blk.0.inp_gate.weight": torch.zeros(1),
            "blk.0.proj.weight": torch.zeros(1),
            "blk.0.layer_output_scale.weight": torch.zeros(1),
            "blk.0.post_norm.weight": torch.zeros(1),
            "blk.0.post_ffw_norm.weight": torch.zeros(1),
        }

        mapped = self.loader.sd_map_replace(state_dict, self.loader.GEMMA4_SD_MAP)

        self.assertEqual(
            set(mapped),
            {
                "model.embed_tokens.weight",
                "model.embed_tokens_per_layer.weight",
                "model.per_layer_model_projection.weight",
                "model.per_layer_projection_norm.weight",
                "model.layers.0.per_layer_input_gate.weight",
                "model.layers.0.per_layer_projection.weight",
                "model.layers.0.layer_scalar",
                "model.layers.0.post_per_layer_input_norm.weight",
                "model.layers.0.post_feedforward_layernorm.weight",
            },
        )

    def test_recreates_gemma4_bpe_tokenizer_json(self):
        tokenizer_json = self.loader.gemma4_tokenizer_json(
            ["<pad>", "<eos>", "<bos>", "<unk>", "hello", "\u2581world"],
            ["h e", "he llo"],
            [3, 3, 3, 3, 1, 1],
        )
        tokenizer = json.loads(bytes(tokenizer_json.tolist()))

        self.assertEqual(tokenizer["model"]["type"], "BPE")
        self.assertEqual(tokenizer["model"]["vocab"]["hello"], 4)
        self.assertEqual(tokenizer["pre_tokenizer"]["type"], "Metaspace")
        self.assertEqual(
            [token["content"] for token in tokenizer["added_tokens"]],
            ["<pad>", "<eos>", "<bos>", "<unk>"],
        )

    def test_e4b_layout_is_detected_by_installed_comfyui(self):
        state_dict = {
            "model.layers.0.post_feedforward_layernorm.weight": torch.zeros(2560),
            "model.layers.41.self_attn.q_norm.weight": torch.zeros(256),
        }

        self.assertEqual(
            comfy.sd.detect_te_model(state_dict),
            comfy.sd.TEModel.GEMMA_4_E4B,
        )


class MinimaxH3DetectionTests(unittest.TestCase):
    def test_detects_native_minimax_h3_checkpoint_layout(self):
        checkpoint_keys = {
            "video_patch_proj.weight",
            "audio_patch_proj.weight",
            "blocks.0.attn.qkv_proj.weight",
            "final_layer.video_out.weight",
        }

        model_arch = detect_arch(checkpoint_keys)

        self.assertIsInstance(model_arch, ModelMinimaxH3)
        self.assertEqual(model_arch.arch, "minimax_h3")

    def test_keeps_adaln_curve_table_in_full_precision(self):
        model_arch = ModelMinimaxH3()

        self.assertIn("adaln_t_table", model_arch.keys_hiprec)

    def test_q4_preserves_sensitive_transformer_components(self):
        state_dict = {
            "video_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "audio_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "blocks.0.attn.qkv_proj.weight": torch.ones((512, 512), dtype=torch.float16),
            "blocks.0.adaln_proj.linear.weight": torch.ones((512, 8), dtype=torch.float16),
            "blocks.0.attn.norm1.weight": torch.ones((512,), dtype=torch.bfloat16),
            "blocks.0.mlp.norm.weight": torch.ones((512,), dtype=torch.bfloat16),
            "blocks.0.modulation.weight": torch.ones((512, 512), dtype=torch.float16),
            "condition_proj.weight": torch.ones((512, 512), dtype=torch.bfloat16),
            "final_layer.video_out.weight": torch.ones((96, 512), dtype=torch.float32),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            output_path = Path(temp_dir) / "minimax_h3-Q4_CR_W4A4.gguf"
            save_file(state_dict, str(source_path))

            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q4_CR_W4A4",
                quantization_device="cpu",
            )

            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

        self.assertEqual(
            tensor_types["blocks.0.attn.qkv_proj.weight"],
            gguf.GGMLQuantizationType.F32,
        )
        self.assertEqual(
            tensor_types["blocks.0.adaln_proj.linear.weight"],
            gguf.GGMLQuantizationType.F32,
        )
        self.assertEqual(
            tensor_types["blocks.0.attn.norm1.weight"],
            gguf.GGMLQuantizationType.F32,
        )
        self.assertEqual(
            tensor_types["blocks.0.modulation.weight"],
            gguf.GGMLQuantizationType.F32,
        )
        self.assertEqual(
            tensor_types["condition_proj.weight"],
            gguf.GGMLQuantizationType.BF16,
        )

    def test_q4_preserves_conditioning_refiner_and_output_paths(self):
        state_dict = {
            "video_patch_proj.weight": torch.ones((512, 96), dtype=torch.float32),
            "audio_patch_proj.weight": torch.ones((512, 32), dtype=torch.float32),
            "blocks.0.attn.qkv_proj.weight": torch.ones((512, 512), dtype=torch.float16),
            "condition_proj.weight": torch.ones((512, 512), dtype=torch.bfloat16),
            "token_refiner.blocks.0.attn.qkv_proj.weight": torch.ones(
                (512, 512), dtype=torch.bfloat16
            ),
            "final_layer.video_out.weight": torch.ones((96, 512), dtype=torch.float32),
            "adaln_t_table": torch.ones((32, 8), dtype=torch.float32),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            output_path = Path(temp_dir) / "minimax_h3-Q4_CR_W4A4.gguf"
            save_file(state_dict, str(source_path))

            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q4_CR_W4A4",
                quantization_device="cpu",
            )

            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

        self.assertEqual(
            tensor_types["blocks.0.attn.qkv_proj.weight"],
            gguf.GGMLQuantizationType.F32,
        )
        self.assertEqual(
            tensor_types["condition_proj.weight"],
            gguf.GGMLQuantizationType.BF16,
        )
        self.assertEqual(
            tensor_types["token_refiner.blocks.0.attn.qkv_proj.weight"],
            gguf.GGMLQuantizationType.BF16,
        )
        self.assertEqual(
            tensor_types["final_layer.video_out.weight"],
            gguf.GGMLQuantizationType.F32,
        )

    def test_converts_to_minimax_h3_gguf_with_full_precision_adaln_table(self):
        state_dict = {
            "video_patch_proj.weight": torch.ones((32, 32), dtype=torch.float16),
            "audio_patch_proj.weight": torch.ones((32, 32), dtype=torch.float16),
            "blocks.0.attn.qkv_proj.weight": torch.ones((96, 32), dtype=torch.float16),
            "final_layer.video_out.weight": torch.ones((96, 32), dtype=torch.float16),
            "adaln_t_table": torch.ones((32, 32), dtype=torch.float32),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            output_path = Path(temp_dir) / "minimax_h3-Q8_0.gguf"
            save_file(state_dict, str(source_path))

            converted_path, model_arch = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q8_0",
            )

            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            architecture = reader.get_field("general.architecture")
            architecture_name = str(architecture.parts[architecture.data[-1]], "utf-8")
            del architecture
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

        self.assertEqual(model_arch.arch, "minimax_h3")
        self.assertEqual(architecture_name, "minimax_h3")
        self.assertEqual(
            tensor_types["blocks.0.attn.qkv_proj.weight"],
            gguf.GGMLQuantizationType.F32,
        )
        self.assertEqual(
            tensor_types["adaln_t_table"],
            gguf.GGMLQuantizationType.F32,
        )


class Q4CRW4A4QuantizationTests(unittest.TestCase):
    def test_int4_cr_w4a4_packs_kitchen_native_layout(self):
        torch.manual_seed(0)
        weight = torch.randn(64, 512, dtype=torch.float32)
        packed, wscales, quant_conf, orig_shape = quantize_int4_cr_w4a4(
            weight, convrot_groupsize=256, quant_group_size=64, device=torch.device("cpu")
        )

        # packed (N, K//2) int8, scales (N,) float32 (per-output-row)
        self.assertEqual(packed.shape, (64, 256))
        self.assertEqual(packed.dtype, torch.int8)
        self.assertEqual(wscales.shape, (64,))
        self.assertEqual(wscales.dtype, torch.float32)
        self.assertEqual(orig_shape, (64, 512))
        self.assertEqual(quant_conf["format"], "int4_cr")
        self.assertEqual(quant_conf["backing"], "w4a4")
        self.assertEqual(quant_conf["convrot_groupsize"], 256)
        self.assertEqual(quant_conf["quant_group_size"], 64)
        self.assertTrue(quant_conf["sym"])

    def test_q4_cr_l_is_exposed_in_converter_node_quantization_choices(self):
        node_module = nodes_factory()
        choices = node_module.TargetedQuantizationGGUF.INPUT_TYPES()["required"]["quantization"][0]
        self.assertIn("Q4_CR_L", choices)
        self.assertIn("Q4_CR_W4A4", choices)
        self.assertIn("Q4_CR", choices)
        self.assertIs(
            node_module.NODE_CLASS_MAPPINGS["TargetedQuantizationGGUF"],
            node_module.TargetedQuantizationGGUF,
        )

    def test_q4_cr_l_is_exposed_and_forwarded_by_conversion_webui(self):
        from tools import conversion_webui

        page = conversion_webui.PAGE.read_text(encoding="utf-8")
        self.assertIn('<option value="Q4_CR_L">', page)

        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "model.safetensors"
            destination = Path(temp_dir) / "model-Q4_CR_L.gguf"
            source.touch()
            with mock.patch.object(conversion_webui.threading, "Thread"):
                manager = conversion_webui.ConversionManager()
                job = manager.create(
                    {
                        "source": str(source),
                        "destination": str(destination),
                        "quant_type": "Q4_CR_L",
                    }
                )

        self.assertEqual(job["quant_type"], "Q4_CR_L")
        self.assertEqual(
            manager.jobs[job["id"]]["command"][-2:],
            ["--quant-type", "Q4_CR_L"],
        )

    def test_int4_cr_w4a8_reuses_packed_layout_and_marks_int8_activations(self):
        torch.manual_seed(1)
        weight = torch.randn(64, 512, dtype=torch.float32)
        packed, wscales, quant_conf, orig_shape = quantize_int4_cr_w4a4(
            weight,
            convrot_groupsize=256,
            quant_group_size=64,
            device=torch.device("cpu"),
            backing="w4a8",
        )

        self.assertEqual(packed.shape, (64, 256))
        self.assertEqual(packed.dtype, torch.int8)
        self.assertEqual(wscales.shape, (64,))
        self.assertEqual(wscales.dtype, torch.float32)
        self.assertEqual(orig_shape, (64, 512))
        self.assertEqual(quant_conf["format"], "int4_cr")
        self.assertEqual(quant_conf["backing"], "w4a8")
        self.assertEqual(quant_conf["linear_dtype"], "int8")

    def test_int4_cr_w4a8_rejects_missing_linear_dtype_metadata(self):
        packed, wscales, quant_conf, _ = quantize_int4_cr_w4a4(
            torch.randn(64, 512), backing="w4a8", device=torch.device("cpu")
        )
        quant_conf.pop("linear_dtype")
        ops = ops_factory(default_linear_dtype="int8")
        lin = ops.Linear(512, 64, bias=False)
        with self.assertRaisesRegex(ValueError, "Missing Q4_CR_L linear_dtype"):
            lin._load_from_state_dict(
                {
                    "weight": packed,
                    "weight_scale": wscales.half(),
                    "comfy_quant": torch.tensor(list(json.dumps(quant_conf).encode("utf-8"))),
                },
                prefix="",
                local_metadata={},
                strict=True,
                missing_keys=[],
                unexpected_keys=[],
                error_msgs=[],
            )

    def test_int4_cr_w4a4_roundtrips_through_dequant(self):
        torch.manual_seed(0)
        weight = torch.randn(64, 1024, dtype=torch.float32)
        packed, wscales, quant_conf, _ = quantize_int4_cr_w4a4(
            weight, convrot_groupsize=256, quant_group_size=64, device=torch.device("cpu")
        )

        # Mirror the ops._dequantized_weight math: signed two's-complement int4
        # per-output-row scale, then un-rotate by the block-diagonal Hadamard.
        n, k = weight.shape
        x32 = packed.to(torch.int32)
        lo = (x32 & 0x0F).to(torch.float32)
        hi = ((x32 >> 4) & 0x0F).to(torch.float32)
        nibbles = torch.stack([lo, hi], dim=-1).reshape(n, k)
        nibbles = torch.where(nibbles >= 8, nibbles - 16, nibbles)
        w_rot = nibbles * wscales.reshape(-1, 1)

        # Un-rotate to the original basis.
        cg = 256
        from tools.convert import _build_regular_hadamard
        h = _build_regular_hadamard(cg, dtype=torch.float32, device="cpu")
        ng = k // cg
        w_rec = (w_rot.reshape(n, ng, cg) @ h).reshape(n, k)

        relative = (w_rec - weight).abs().amax().item() / weight.abs().amax().item()
        self.assertLess(relative, 0.3)

    def test_int4_cr_w4a4_rejects_k_not_divisible_by_convrot(self):
        # K=300 is not divisible by the 256 convrot group -> must reject.
        with self.assertRaisesRegex(ValueError, "must divide input features"):
            quantize_int4_cr_w4a4(
                torch.randn(64, 300), convrot_groupsize=256, quant_group_size=64,
                device=torch.device("cpu")
            )
        with self.assertRaisesRegex(ValueError, "must divide input features"):
            quantize_int4_cr_w4a4(
                torch.randn(64, 224), convrot_groupsize=256, quant_group_size=64,
                device=torch.device("cpu")
            )

    def test_int4_cr_w4a4_serializes_kitchen_metadata(self):
        state_dict = {
            "video_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "audio_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "blocks.0.attn.qkv_proj.weight": torch.ones((96, 512), dtype=torch.float32),
            "blocks.0.mlp.fc1.weight": torch.ones((128, 512), dtype=torch.float32),
            "final_layer.video_out.weight": torch.ones((96, 512), dtype=torch.float32),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            output_path = Path(temp_dir) / "minimax_h3-Q4_CR_W4A4.gguf"
            save_file(state_dict, str(source_path))

            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q4_CR_W4A4",
                quantization_device="cpu",
            )

            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            names = set(tensor_types.keys())
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

        # Weight stays I8 (2 uint4 per byte) and per-row scale stored as F16.
        self.assertEqual(tensor_types["blocks.0.mlp.fc1.weight"], gguf.GGMLQuantizationType.I8)
        self.assertEqual(tensor_types["blocks.0.mlp.fc1.weight_scale"], gguf.GGMLQuantizationType.F16)
        self.assertIn("blocks.0.mlp.fc1.weight_scale", names)

    def test_int4_cr_w4a4_loader_routes_to_w4a4_ops(self):
        state_dict = {
            "video_patch_proj.weight": torch.randn(32, 32, dtype=torch.float32),
            "audio_patch_proj.weight": torch.randn(32, 32, dtype=torch.float32),
            "blocks.0.attn.qkv_proj.weight": torch.randn(96, 512, dtype=torch.float32),
            "blocks.0.mlp.fc1.weight": torch.randn(128, 512, dtype=torch.float32),
            "final_layer.video_out.weight": torch.randn(96, 512, dtype=torch.float32),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            output_path = Path(temp_dir) / "minimax_h3-Q4_CR_W4A4.gguf"
            save_file(state_dict, str(source_path))

            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q4_CR_W4A4",
                quantization_device="cpu",
            )

            loader = load_gguf_loader()
            sd, extra = loader.gguf_sd_loader(converted_path)

            w = sd["blocks.0.mlp.fc1.weight"]
            sd_key = "blocks.0.mlp.fc1"
            qraw = sd[f"{sd_key}.comfy_quant"]
            quant_conf = json.loads(bytes(qraw.tolist()).decode("utf-8"))

            w_detached = w.detach().clone().tolist()
            del w
            del qraw
            del sd
            del extra
            gc.collect()

        self.assertEqual(quant_conf["format"], "int4_cr")
        self.assertEqual(quant_conf["backing"], "w4a4")
        self.assertEqual(quant_conf["orig_shape"], [128, 512])

    def test_minimax_h3_q4_keeps_residual_projections_in_full_precision(self):
        state_dict = {
            "video_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "audio_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "blocks.0.attn.qkv_proj.weight": torch.ones((96, 512), dtype=torch.float32),
            "blocks.0.attn.out_proj.weight": torch.ones((96, 512), dtype=torch.float32),
            "blocks.0.mlp.fc1.weight": torch.ones((128, 512), dtype=torch.float32),
            "blocks.0.mlp.fc2.weight": torch.ones((96, 512), dtype=torch.float32),
            "final_layer.video_out.weight": torch.ones((96, 32), dtype=torch.float32),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            output_path = Path(temp_dir) / "minimax_h3-Q4_CR_W4A4.gguf"
            save_file(state_dict, str(source_path))

            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q4_CR_W4A4",
                quantization_device="cpu",
            )

            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            quant_config_names = {
                field_name[len("comfy.gguf.quant."):]
                for field_name in reader.fields
                if field_name.startswith("comfy.gguf.quant.")
            }
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

        self.assertEqual(
            tensor_types["blocks.0.attn.qkv_proj.weight"],
            gguf.GGMLQuantizationType.F32,
        )
        self.assertEqual(
            tensor_types["blocks.0.attn.out_proj.weight"],
            gguf.GGMLQuantizationType.F32,
        )
        self.assertEqual(
            tensor_types["blocks.0.mlp.fc2.weight"],
            gguf.GGMLQuantizationType.F32,
        )
        self.assertNotIn("blocks.0.attn.qkv_proj.weight", quant_config_names)
        self.assertNotIn("blocks.0.attn.out_proj.weight", quant_config_names)
        self.assertNotIn("blocks.0.mlp.fc2.weight", quant_config_names)

    def test_int4_cr_w4a4_ops_empty_patch_keeps_native_kernel(self):
        # With no weight_function (the normal DynamicVRAM case when no LoRA is active),
        # the native W4A4 kernel runs and the device-resident QuantizedTensor is cached.
        # A real patch (LoRA) must never be probed against the packed int8, because a
        # LoRA adapter swallows its addmm-on-int8 error and silently returns the int8
        # weight unchanged, which would drop the adapter delta and misroute to native.
        ops = ops_factory()
        lin = ops.Linear(64, 64, bias=True)
        weight2 = torch.randn(64, 512, dtype=torch.float32)
        packed, wscales, quant_conf, orig_shape = quantize_int4_cr_w4a4(
            weight2, convrot_groupsize=256, quant_group_size=64, device=torch.device("cpu")
        )
        quant_conf["orig_shape"] = [64, 512]
        lin._load_from_state_dict(
            {
                "weight": packed,
                "weight_scale": wscales.half(),
                "comfy_quant": torch.tensor(list(json.dumps(quant_conf).encode("utf-8"))),
                "bias": torch.randn(64, dtype=torch.float32),
            },
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )
        self.assertTrue(lin._quantized)

        x = torch.randn(8, 512, dtype=torch.float32)

        # No patch: use the cached QuantizedTensor fast path.
        lin.weight_function = []
        lin.bias_function = []
        lin._quantized_weight = None
        lin._quantized_weight_device = None
        out = lin(x)
        self.assertEqual(out.shape, (8, 64))
        self.assertIsNotNone(lin._quantized_weight)
        self.assertEqual(lin._quantized_weight_device, str(x.device))

        # A real (value-changing) patch: cache the patched floating-point weight so
        # the adapter delta is not erased by INT4 re-quantization.
        def patch(t, *a, **kw):
            return t.to(torch.float32) + 1.0
        lin.weight_function = [patch]
        lin._quantized_weight = None
        lin._quantized_weight_device = None
        out = lin(x)
        self.assertEqual(out.shape, (8, 64))
        self.assertIsNone(lin._quantized_weight)
        self.assertIsNotNone(lin._fused_weight)
        self.assertFalse(isinstance(lin._fused_weight, torch.Tensor) and lin._fused_weight.dtype == torch.int8)

    def test_int4_cr_w4a4_ops_value_patch_is_cached_in_compute_dtype(self):
        # A genuine value-changing patch is cached in the compute dtype rather than
        # re-quantized, because INT4 rounding can erase a small adapter delta.
        ops = ops_factory()
        lin = ops.Linear(64, 64, bias=True)
        weight = torch.randn(64, 512, dtype=torch.float32)
        packed, wscales, quant_conf, orig_shape = quantize_int4_cr_w4a4(
            weight, convrot_groupsize=256, quant_group_size=64, device=torch.device("cpu")
        )
        quant_conf["orig_shape"] = [64, 512]
        lin._load_from_state_dict(
            {
                "weight": packed,
                "weight_scale": wscales.half(),
                "comfy_quant": torch.tensor(list(json.dumps(quant_conf).encode("utf-8"))),
                "bias": torch.randn(64, dtype=torch.float32),
            },
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )
        x = torch.randn(8, 512, dtype=torch.float32)

        # A patch that returns fp32 must be fused and retained as a floating weight.
        def patch(t, *a, **kw):
            return t.to(torch.float32)
        lin.weight_function = [patch]
        lin._quantized_weight = None
        lin._quantized_weight_device = None
        out = lin(x)
        self.assertEqual(out.shape, (8, 64))
        self.assertIsNotNone(lin._fused_weight)
        self.assertTrue(isinstance(lin._fused_weight, torch.Tensor))
        self.assertEqual(lin._fused_weight.dtype, lin._compute_dtype)
        self.assertIsNone(lin._quantized_weight)

    def test_int4_cr_w4a4_ops_eager_fusion_and_cache_eviction(self):
        ops = ops_factory()
        lin = ops.Linear(64, 64, bias=True)
        weight = torch.randn(64, 512, dtype=torch.float32)
        packed, wscales, quant_conf, _ = quantize_int4_cr_w4a4(
            weight, convrot_groupsize=256, quant_group_size=64, device=torch.device("cpu")
        )
        quant_conf["orig_shape"] = [64, 512]
        lin._load_from_state_dict(
            {
                "weight": packed,
                "weight_scale": wscales.half(),
                "comfy_quant": torch.tensor(list(json.dumps(quant_conf).encode("utf-8"))),
                "bias": torch.zeros(64, dtype=torch.float32),
            },
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )

        calls = []

        def patch(t, *args, **kwargs):
            calls.append(None)
            return t + 0.25

        lin.weight_function = [patch]
        lin._get_cached_quantized_weight(torch.device("cpu"))
        self.assertIsNotNone(lin._quantized_weight)
        self.assertTrue(lin.prepare_fused_weight(torch.device("cpu")))
        self.assertEqual(len(calls), 1)
        self.assertIsNone(lin._quantized_weight)
        first_fused = id(lin._fused_weight)
        first_patch_id = lin._fused_patch_id
        self.assertEqual(lin._fused_weight.device, torch.device("cpu"))

        # A forward on another device must use a temporary copy, not promote the
        # persistent offload cache and retain it on CUDA for subsequent layers.
        cached = lin._get_cached_fused_weight(torch.device("cuda:0"))
        self.assertEqual(id(cached), first_fused)
        self.assertEqual(cached.device, torch.device("cpu"))

        lin(torch.randn(2, 512))
        self.assertEqual(len(calls), 1)
        self.assertEqual(id(lin._fused_weight), first_fused)

        lin.weight_function = [lambda t, *args, **kwargs: t - 0.25]
        self.assertTrue(lin.prepare_fused_weight(torch.device("cpu")))
        self.assertIsNone(lin._quantized_weight)
        self.assertNotEqual(lin._fused_patch_id, first_patch_id)

        lin.evict_quantized_caches()
        self.assertIsNone(lin._quantized_weight)
        self.assertIsNone(lin._fused_weight)
        self.assertIsNone(lin._fused_bias)

    def test_int4_cr_w4a4_ops_non_native_patch_is_applied(self):
        # A non-native patch must be applied to the dequantized, un-rotated weight
        # and retained in the fallback cache. Standard LoRA uses the native low-rank
        # bypass exercised by the following test instead.
        ops = ops_factory()
        lin = ops.Linear(64, 64, bias=True)
        weight = torch.randn(64, 512, dtype=torch.float32)
        packed, wscales, quant_conf, orig_shape = quantize_int4_cr_w4a4(
            weight, convrot_groupsize=256, quant_group_size=64, device=torch.device("cpu")
        )
        quant_conf["orig_shape"] = [64, 512]
        lin._load_from_state_dict(
            {
                "weight": packed,
                "weight_scale": wscales.half(),
                "comfy_quant": torch.tensor(list(json.dumps(quant_conf).encode("utf-8"))),
                "bias": torch.randn(64, dtype=torch.float32),
            },
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )
        self.assertTrue(lin._quantized)
        x = torch.randn(8, 512, dtype=torch.float32)

        # Low-rank LoRA-style delta encoded as a weight_function.
        r = 8
        a = torch.randn(64, r, dtype=torch.float32)
        b = torch.randn(r, 512, dtype=torch.float32)
        delta = a @ b

        def lora_patch(t, *args, **kwargs):
            # Only valid on a full-precision, un-rotated weight of the base shape.
            self.assertEqual(t.shape, (64, 512))
            self.assertNotEqual(t.dtype, torch.int8)
            return t + delta

        lin.weight_function = [lora_patch]
        lin._quantized_weight = None
        lin._quantized_weight_device = None
        # Fusion keeps the adapter's complete delta in a cached compute-dtype weight;
        # the plain packed cache is not retained.
        out = lin(x)
        self.assertEqual(out.shape, (8, 64))
        self.assertIsNotNone(lin._fused_weight)
        self.assertIsNone(lin._quantized_weight)

        # Reference: apply the same patch to the dequantized weight. The cached fused
        # weight should match it exactly in the compute dtype.
        fused = lin._dequantized_weight(torch.device("cpu"), lin._compute_dtype)
        fused = lora_patch(fused).to(dtype=lin._compute_dtype)
        torch.testing.assert_close(lin._fused_weight, fused)
        ref_out = torch.nn.functional.linear(x, fused.to(dtype=x.dtype))
        ref_bias = lin.bias.to(device=x.device, dtype=x.dtype)
        if ref_bias is not None:
            ref_out = ref_out + ref_bias
        torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=1e-3)
        base_weight = lin._dequantized_weight(torch.device("cpu"), x.dtype)
        base_out = torch.nn.functional.linear(x, base_weight, ref_bias)
        self.assertFalse(torch.allclose(out, base_out, atol=1e-4, rtol=1e-3))

    def test_int4_cr_w4a4_ops_standard_lora_uses_native_base_and_residual(self):
        from comfy.weight_adapter import LoRAAdapter

        ops = ops_factory()
        lin = ops.Linear(64, 64, bias=True)
        weight = torch.randn(64, 512, dtype=torch.float32)
        packed, wscales, quant_conf, orig_shape = quantize_int4_cr_w4a4(
            weight, convrot_groupsize=256, quant_group_size=64, device=torch.device("cpu")
        )
        quant_conf["orig_shape"] = [64, 512]
        lin._load_from_state_dict(
            {
                "weight": packed,
                "weight_scale": wscales.half(),
                "comfy_quant": torch.tensor(list(json.dumps(quant_conf).encode("utf-8"))),
                "bias": torch.randn(64, dtype=torch.float32),
            },
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )

        rank = 8
        up = torch.randn(64, rank, dtype=torch.float32)
        down = torch.randn(rank, 512, dtype=torch.float32)
        alpha = 4.0
        adapter = LoRAAdapter(set(), (up, down, alpha, None, None, None))
        key = "blocks.1.weight"
        patches = {key: [(0.75, adapter, 1.0, None, None)]}

        class FakeLowVramPatch:
            is_lowvram_patch = True

            def __init__(self, key, patches):
                self.key = key
                self.patches = patches
                self.prepared_patches = None

            def __call__(self, value):
                return value

        x = torch.randn(8, 512, dtype=torch.float32)
        base_out = lin(x)
        lin.weight_function = [FakeLowVramPatch(key, patches)]

        # The supported adapter must not trigger full-matrix fusion or retain a
        # second full-precision weight; the native packed base remains cached.
        self.assertIsNotNone(lin._native_lora_patch_entries())
        self.assertFalse(lin.prepare_fused_weight(torch.device("cpu")))
        self.assertIsNotNone(lin._quantized_weight)
        out = lin(x)
        scale = alpha / rank
        expected = base_out + 0.75 * torch.nn.functional.linear(
            torch.nn.functional.linear(x, down), up
        ) * scale
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-3)
        self.assertIsNone(lin._fused_weight)
        self.assertIsNotNone(lin._quantized_weight)

    def test_int4_cr_w4a4_ops_cpu_fallback_applies_native_lora(self):
        from comfy.weight_adapter import LoRAAdapter

        ops = ops_factory()
        lin = ops.Linear(64, 64, bias=True)
        weight = torch.randn(64, 512, dtype=torch.float32)
        packed, wscales, quant_conf, orig_shape = quantize_int4_cr_w4a4(
            weight, convrot_groupsize=256, quant_group_size=64, device=torch.device("cpu")
        )
        quant_conf["orig_shape"] = [64, 512]
        lin._load_from_state_dict(
            {
                "weight": packed,
                "weight_scale": wscales.half(),
                "comfy_quant": torch.tensor(list(json.dumps(quant_conf).encode("utf-8"))),
                "bias": torch.randn(64, dtype=torch.float32),
            },
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )

        rank = 8
        up = torch.randn(64, rank, dtype=torch.float32)
        down = torch.randn(rank, 512, dtype=torch.float32)
        adapter = LoRAAdapter(set(), (up, down, 4.0, None, None, None))
        key = "blocks.1.weight"
        patches = {key: [(0.75, adapter, 1.0, None, None)]}

        class FakeLowVramPatch:
            is_lowvram_patch = True

            def __init__(self, key, patches):
                self.key = key
                self.patches = patches
                self.prepared_patches = None

            def __call__(self, value):
                return value

        x = torch.randn(8, 512, dtype=torch.float32)
        lin.weight_function = [FakeLowVramPatch(key, patches)]
        native_entries = lin._native_lora_patch_entries()

        fallback = lin._cpu_forward_fallback(x, lin.bias, native_entries=native_entries)
        expected = torch.nn.functional.linear(
            x, lin._dequantized_weight(torch.device("cpu"), x.dtype), lin.bias
        ) + 0.75 * torch.nn.functional.linear(
            torch.nn.functional.linear(x, down), up
        ) * (4.0 / rank)
        torch.testing.assert_close(fallback, expected, atol=1e-4, rtol=1e-3)
        self.assertEqual(fallback.device, torch.device("cpu"))

    def test_int4_cr_w4a4_ops_dispatches_to_cpu_on_cuda_oom(self):
        ops = ops_factory()
        lin = ops.Linear(4, 4, bias=False)
        lin.weight = torch.zeros((4, 4), dtype=torch.float32)
        lin._quantized = True
        lin.weight_function = [lambda value: value]
        input_tensor = mock.Mock(device=torch.device("cuda"), dtype=torch.float16)
        sentinel = object()

        with mock.patch.object(
            lin, "_get_cached_quantized_weight", side_effect=torch.OutOfMemoryError("test")
        ), mock.patch.object(lin, "_cpu_forward_fallback", return_value=sentinel) as fallback:
            result = lin.forward_comfy_cast_weights(input_tensor)

        self.assertIs(result, sentinel)
        fallback.assert_called_once_with(
            input_tensor, None, native_entries=None, use_fused_weight=True
        )

    def test_int4_cr_w4a4_ops_lokr_uses_native_base_and_residual(self):
        from comfy.weight_adapter.lokr import LoKrAdapter

        ops = ops_factory()
        lin = ops.Linear(64, 64, bias=True)
        weight = torch.randn(64, 512, dtype=torch.float32)
        packed, wscales, quant_conf, orig_shape = quantize_int4_cr_w4a4(
            weight, convrot_groupsize=256, quant_group_size=64, device=torch.device("cpu")
        )
        quant_conf["orig_shape"] = [64, 512]
        lin._load_from_state_dict(
            {
                "weight": packed,
                "weight_scale": wscales.half(),
                "comfy_quant": torch.tensor(list(json.dumps(quant_conf).encode("utf-8"))),
                "bias": torch.randn(64, dtype=torch.float32),
            },
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )

        # This matches the direct lokr_w1/lokr_w2 layout used by the Krea2 LoKr
        # adapter. Keep the factors on CPU, as they are after normal LoRA loading.
        w1 = torch.randn(8, 8, dtype=torch.float32)
        w2 = torch.randn(8, 64, dtype=torch.float32)
        adapter = LoKrAdapter(
            set(), (w1, w2, None, None, None, None, None, None, None)
        )
        key = "blocks.1.weight"
        patches = {key: [(0.75, adapter, 1.0, None, None)]}

        class FakeLowVramPatch:
            is_lowvram_patch = True

            def __init__(self, key, patches):
                self.key = key
                self.patches = patches
                self.prepared_patches = None

            def __call__(self, value):
                return value

        x = torch.randn(8, 512, dtype=torch.float32)
        base_out = lin(x)
        lin.weight_function = [FakeLowVramPatch(key, patches)]

        self.assertIsNotNone(lin._native_lora_patch_entries())
        self.assertFalse(lin.prepare_fused_weight(torch.device("cpu")))
        with mock.patch.object(
            lin, "_get_cached_fused_weight", side_effect=AssertionError("LoKr fallback")
        ):
            out = lin(x)

        expected = base_out + 0.75 * torch.nn.functional.linear(
            x, torch.kron(w1, w2)
        )
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-3)
        self.assertEqual(w1.device, torch.device("cpu"))
        self.assertEqual(w2.device, torch.device("cpu"))
        self.assertIsNone(lin._fused_weight)
        self.assertIsNotNone(lin._quantized_weight)

    def test_int4_lora_factor_cache_reuses_cuda_factors_with_headroom(self):
        ops = ops_factory()
        lin = ops.Linear(64, 64, bias=False)
        factors = [torch.randn(8, 64), torch.randn(64, 8), torch.randn(4, 4)]
        cuda = torch.device("cuda:0")
        reserve = 1024**3

        with mock.patch.object(
            torch.cuda, "mem_get_info", return_value=(2 * reserve, 8 * reserve)
        ), mock.patch.object(
            comfy.model_management, "extra_reserved_memory", return_value=reserve
        ), mock.patch.object(
            comfy.model_management,
            "cast_to_device",
            side_effect=lambda factor, device, dtype: factor,
        ) as cast:
            for factor in factors:
                self.assertIs(
                    lin._get_cached_lora_factor(factor, cuda, torch.bfloat16), factor
                )
                self.assertIs(
                    lin._get_cached_lora_factor(factor, cuda, torch.bfloat16), factor
                )

        self.assertEqual(cast.call_count, len(factors))
        self.assertEqual(len(lin._lora_factor_cache), len(factors))
        lin.evict_quantized_caches()
        self.assertFalse(lin._lora_factor_cache)

    def test_int4_lora_factor_cache_streams_at_cuda_reserve_boundary(self):
        ops = ops_factory()
        lin = ops.Linear(64, 64, bias=False)
        factor = torch.randn(8, 64)
        cuda = torch.device("cuda:0")
        reserve = 1024**3

        with mock.patch.object(
            torch.cuda, "mem_get_info", return_value=(reserve, 8 * reserve)
        ), mock.patch.object(
            comfy.model_management, "extra_reserved_memory", return_value=reserve
        ), mock.patch.object(
            comfy.model_management,
            "cast_to_device",
            side_effect=lambda factor, device, dtype: factor,
        ) as cast:
            lin._get_cached_lora_factor(factor, cuda, torch.bfloat16)
            lin._get_cached_lora_factor(factor, cuda, torch.bfloat16)

        self.assertEqual(cast.call_count, 2)
        self.assertFalse(lin._lora_factor_cache)

    def test_int4_lora_factor_cache_logs_loaded_models_and_reraises_cuda_oom(self):
        ops = ops_factory()
        lin = ops.Linear(64, 64, bias=False)
        factor = torch.randn(8, 64)
        cuda = torch.device("cuda:0")
        loaded_model = type(
            "FakeLoadedModel",
            (),
            {
                "model": type(
                    "FakePatcher",
                    (),
                    {"model": type("CachedModel", (), {"name": "base"})(), "size": 2 * 1024**2},
                )()
            },
        )()
        missing_size_model = type(
            "FakeLoadedModel",
            (),
            {"model": type("FakePatcher", (), {"model": type("AdapterModel", (), {})()})()},
        )()

        with mock.patch.object(
            torch.cuda, "mem_get_info", return_value=(2 * 1024**3, 8 * 1024**3)
        ), mock.patch.object(
            comfy.model_management,
            "current_loaded_models",
            [loaded_model, missing_size_model],
        ), mock.patch.object(
            comfy.model_management,
            "cast_to_device",
            side_effect=torch.OutOfMemoryError("forced cache OOM"),
        ), self.assertLogs(level="ERROR") as logs:
            with self.assertRaisesRegex(torch.OutOfMemoryError, "forced cache OOM"):
                lin._get_cached_lora_factor(factor, cuda, torch.bfloat16)

        self.assertIn("caching INT4 LoRA/LoKr factors", logs.output[0])
        self.assertIn("CachedModel (base): 2.0 MiB", logs.output[1])
        self.assertIn("AdapterModel: size unavailable", logs.output[2])

    def test_int4_cr_w4a4_ops_fused_cache_survives_lowvram_patch_rebind(self):
        # Dynamic VRAM (GGUFModelPatcherDynamic.load) promotes a freshly created
        # LowVramPatch into weight_function every time the model is moved to device.
        # That patch object is recreated on each reload, so the fused-weight cache must
        # NOT be keyed on id(patch_object) — otherwise the expensive Hadamard rotate +
        # int4 pack runs on every forward (a 12x slowdown measured on large layers).
        # This test verifies the cache survives patch-object rebinding and only re-fuses
        # when the underlying patch *content* (key / patches dict) actually changes.
        ops = ops_factory()
        lin = ops.Linear(64, 64, bias=True)
        weight = torch.randn(64, 512, dtype=torch.float32)
        packed, wscales, quant_conf, orig_shape = quantize_int4_cr_w4a4(
            weight, convrot_groupsize=256, quant_group_size=64, device=torch.device("cpu")
        )
        quant_conf["orig_shape"] = [64, 512]
        lin._load_from_state_dict(
            {
                "weight": packed,
                "weight_scale": wscales.half(),
                "comfy_quant": torch.tensor(list(json.dumps(quant_conf).encode("utf-8"))),
                "bias": torch.randn(64, dtype=torch.float32),
            },
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )
        self.assertTrue(lin._quantized)
        x = torch.randn(8, 512, dtype=torch.float32)

        # Mimic comfy.model_patcher.LowVramPatch: an object with is_lowvram_patch,
        # a stable tensor key, and a pointer to the model-patcher-held patches dict.
        class FakeLowVramPatch:
            is_lowvram_patch = True
            def __init__(self, key, patches):
                self.key = key
                self.patches = patches
                self.prepared_patches = None
            def __call__(self, weight):
                # A genuine value-changing adapter, so it must fuse.
                return weight + 0.1

        # The dict/list live on the model patcher and persist across reloads.
        key = "blocks.1.weight"
        patches = {key: [(1.0, torch.full((64, 512), 0.5), 1.0, None, None)]}

        # First forward builds the fused weight.
        lin.weight_function = [FakeLowVramPatch(key, patches)]
        out = lin(x)
        self.assertEqual(out.shape, (8, 64))
        self.assertIsNotNone(lin._fused_weight)
        first_id = id(lin._fused_weight)
        sig1 = lin._fused_patch_signature()

        # Simulate repeated DynamicVRAM reloads: a NEW patch object each forward, but
        # the same underlying patches dict/list (which persists on the model patcher).
        for _ in range(3):
            lin.weight_function = [FakeLowVramPatch(key, patches)]
            out = lin(x)
        self.assertEqual(out.shape, (8, 64))
        # The cache must have been reused (same fused-QuantizedTensor, same signature).
        self.assertEqual(id(lin._fused_weight), first_id)
        self.assertEqual(lin._fused_patch_signature(), sig1)

        # A real patch change (new patches dict → new list) must invalidate the cache.
        changed_patches = {key: [(1.0, torch.full((64, 512), 0.25), 1.0, None, None)]}
        lin.weight_function = [FakeLowVramPatch(key, changed_patches)]
        out = lin(x)
        self.assertEqual(out.shape, (8, 64))
        self.assertIsNotNone(lin._fused_weight)
        self.assertNotEqual(lin._fused_patch_signature(), sig1)

    def test_int4_lora_fusion_defaults_to_execution_device(self):
        nodes_module = nodes_factory()
        calls = []
        moves = []

        modules = [
            type(
                "FakeQuantizedLayer",
                (),
                {
                    "weight_function": [object()],
                    "bias_function": [],
                    "_fused_patch_signature": lambda self: ("patch",),
                    "prepare_fused_weight": lambda self, device: calls.append(device) or True,
                    "move_fused_caches": lambda self, device: moves.append(device),
                },
            )()
        ]
        patcher = type(
            "FakePatcher",
            (),
            {
                "model": type("FakeModel", (), {"named_modules": lambda self: [("layer", modules[0])]} )(),
                "load_device": torch.device("cuda:0"),
                "offload_device": torch.device("cpu"),
                "_gguf_patch_layout_signature": None,
            },
        )()

        with mock.patch.object(nodes_module.comfy.model_management, "cuda_device_context", return_value=nullcontext()), \
                mock.patch.object(nodes_module.comfy.model_management, "throw_exception_if_processing_interrupted", None):
            prepared = nodes_module.GGUFModelPatcher._prepare_gguf_quantized_weights(patcher)

        self.assertFalse(hasattr(nodes_module, "tqdm"))
        self.assertEqual(prepared, 1)
        self.assertEqual(calls, [torch.device("cuda:0")])
        self.assertEqual(moves, [torch.device("cpu")])

    def test_int4_patch_preparation_evicts_caches_after_interrupt(self):
        nodes_module = nodes_factory()
        calls = []
        evictions = []
        modules = [
            type(
                "FakeQuantizedLayer",
                (),
                {
                    "weight_function": [object()],
                    "bias_function": [],
                    "_fused_patch_signature": lambda self: ("patch",),
                    "prepare_fused_weight": lambda self, device: calls.append(device) or True,
                },
            )(),
            type(
                "FakeQuantizedLayer",
                (),
                {
                    "weight_function": [object()],
                    "bias_function": [],
                    "_fused_patch_signature": lambda self: ("patch",),
                    "prepare_fused_weight": lambda self, device: calls.append(device) or True,
                },
            )(),
        ]
        patcher = type(
            "FakePatcher",
            (),
            {
                "model": type(
                    "FakeModel",
                    (),
                    {
                        "named_modules": lambda self: [
                            ("first", modules[0]),
                            ("second", modules[1]),
                        ]
                    },
                )(),
                "load_device": torch.device("cpu"),
                "offload_device": torch.device("cpu"),
                "_gguf_patch_layout_signature": None,
                "_evict_gguf_quantized_caches": lambda self: evictions.append(True),
            },
        )()
        with mock.patch.object(
            nodes_module.comfy.model_management,
            "throw_exception_if_processing_interrupted",
            side_effect=RuntimeError("interrupted"),
        ), mock.patch.object(
            nodes_module.comfy.model_management,
            "cuda_device_context",
            return_value=nullcontext(),
        ):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                nodes_module.GGUFModelPatcher._prepare_gguf_quantized_weights(patcher)

        self.assertEqual(calls, [])
        self.assertEqual(evictions, [True])

    def test_dynamic_vram_warns_for_unique_offloaded_lora_factors(self):
        nodes_module = nodes_factory()
        factor = torch.ones((2, 2))
        adapter = type(
            "FakeLoRAAdapter",
            (),
            {"name": "lora", "weights": (factor, factor, 1.0, None, None, None)},
        )()
        key = "blocks.1.weight"
        patch_function = type(
            "FakeLowVramPatch",
            (),
            {
                "is_lowvram_patch": True,
                "key": key,
                "patches": {key: [(1.0, adapter, 1.0, None, None)]},
                "prepared_patches": None,
            },
        )()
        module = type(
            "FakeLayer",
            (),
            {"weight_function": [patch_function], "bias_function": []},
        )()
        patcher = type(
            "FakeDynamicPatcher",
            (),
            {
                "model": type(
                    "FakeModel",
                    (),
                    {"named_modules": lambda self: [("layer", module)]},
                )(),
                "load_device": torch.device("cuda:0"),
                "offload_device": torch.device("cpu"),
            },
        )()

        with mock.patch.dict(
            os.environ, {"COMFYUI_GGUF_INT4_LORA_OFFLOAD": "yes"}
        ), mock.patch.object(
            nodes_module, "_DYNAMIC_VRAM_LORA_WARNING_MIN_BYTES", 1
        ), mock.patch.object(torch.cuda, "is_available", return_value=True), mock.patch.object(
            torch.cuda, "mem_get_info", return_value=(2 * 1024**3, 8 * 1024**3)
        ), self.assertLogs(level="WARNING") as logs:
            warned = nodes_module.GGUFModelPatcherDynamic._warn_dynamic_vram_lora_streaming(
                patcher
            )

        self.assertTrue(warned)
        self.assertIn("1 unique offloaded LoRA/LoKr factor tensors", logs.output[0])
        self.assertIn("from cpu to cuda:0", logs.output[0])
        self.assertIn("CUDA free/total 2.0/8.0 GiB", logs.output[0])
        self.assertIn("does not guarantee an OOM", logs.output[0])

    def test_dynamic_vram_preloads_unique_lora_and_lokr_factors(self):
        nodes_module = nodes_factory()
        shared = torch.ones((2, 2))
        lora_only = torch.ones((2, 3))
        lokr_only = torch.ones((3, 2))
        lora = type(
            "FakeLoRAAdapter",
            (),
            {"name": "lora", "weights": (shared, lora_only, 4.0, None, "preserve")},
        )()
        lokr = type(
            "FakeLoKrAdapter",
            (),
            {"name": "lokr", "weights": (lokr_only, shared, None, 2.0)},
        )()
        key = "blocks.1.weight"
        patch_function = type(
            "FakeLowVramPatch",
            (),
            {
                "is_lowvram_patch": True,
                "key": key,
                "patches": {
                    key: [
                        (1.0, lora, 1.0, None, None),
                        (1.0, lokr, 1.0, None, None),
                    ]
                },
                "prepared_patches": None,
            },
        )()
        module = type(
            "FakeLayer",
            (),
            {"weight_function": [patch_function], "bias_function": []},
        )()
        replacements = {}
        moved = []

        def move_factor(factor, device):
            moved.append((factor, device))
            replacement = factor.clone()
            replacements[id(factor)] = replacement
            return replacement

        patcher = type(
            "FakeDynamicPatcher",
            (),
            {
                "model": type(
                    "FakeModel",
                    (),
                    {"named_modules": lambda self: [("layer", module)]},
                )(),
                "load_device": torch.device("cuda:0"),
                "offload_device": torch.device("cpu"),
                "_move_dynamic_vram_lora_factor": staticmethod(move_factor),
            },
        )()
        factor_bytes = sum(
            factor.numel() * factor.element_size()
            for factor in (shared, lora_only, lokr_only)
        )
        reserve = 123456

        with mock.patch.object(torch.cuda, "is_available", return_value=True), mock.patch.object(
            torch.cuda, "mem_get_info", return_value=(reserve + factor_bytes, 8 * reserve)
        ), mock.patch.object(
            nodes_module.comfy.model_management, "extra_reserved_memory", return_value=reserve
        ):
            preloaded = nodes_module.GGUFModelPatcherDynamic._preload_dynamic_vram_lora_factors(
                patcher
            )

        self.assertTrue(preloaded)
        self.assertEqual(len(moved), 3)
        self.assertEqual(
            {id(factor) for factor, _ in moved},
            {id(shared), id(lora_only), id(lokr_only)},
        )
        self.assertTrue(all(device == torch.device("cuda:0") for _, device in moved))
        self.assertIs(lora.weights[0], replacements[id(shared)])
        self.assertIs(lora.weights[1], replacements[id(lora_only)])
        self.assertIs(lokr.weights[0], replacements[id(lokr_only)])
        self.assertIs(lokr.weights[1], replacements[id(shared)])
        self.assertEqual(lora.weights[2:], (4.0, None, "preserve"))
        self.assertEqual(lokr.weights[2:], (None, 2.0))

    def test_dynamic_vram_keeps_lora_factors_on_cuda_without_offload_enabled(self):
        nodes_module = nodes_factory()
        lora_factor = torch.ones((2, 2))
        lokr_factor = torch.ones((2, 3))
        lora = type(
            "FakeLoRAAdapter", (), {"name": "lora", "weights": (lora_factor, 4.0)}
        )()
        lokr = type(
            "FakeLoKrAdapter", (), {"name": "lokr", "weights": (lokr_factor, None)}
        )()
        original_lora_weights = lora.weights
        original_lokr_weights = lokr.weights
        key = "blocks.1.weight"
        patch_function = type(
            "FakeLowVramPatch",
            (),
            {
                "is_lowvram_patch": True,
                "key": key,
                "patches": {
                    key: [
                        (1.0, lora, 1.0, None, None),
                        (1.0, lokr, 1.0, None, None),
                    ]
                },
                "prepared_patches": None,
            },
        )()
        module = type(
            "FakeLayer",
            (),
            {"weight_function": [patch_function], "bias_function": []},
        )()
        moved = []
        patcher = type(
            "FakeDynamicPatcher",
            (),
            {
                "model": type(
                    "FakeModel",
                    (),
                    {"named_modules": lambda self: [("layer", module)]},
                )(),
                "load_device": torch.device("cuda:0"),
                "offload_device": torch.device("cpu"),
                "_move_dynamic_vram_lora_factor": staticmethod(
                    lambda factor, device: moved.append((factor, device)) or factor.clone()
                ),
            },
        )()
        factor_bytes = (
            lora_factor.numel() * lora_factor.element_size()
            + lokr_factor.numel() * lokr_factor.element_size()
        )
        reserve = 123456

        with mock.patch.dict(
            os.environ, {"COMFYUI_GGUF_INT4_LORA_OFFLOAD": "false"}
        ), mock.patch.object(torch.cuda, "is_available", return_value=True), mock.patch.object(
            torch.cuda, "mem_get_info", return_value=(reserve + factor_bytes - 1, 8 * reserve)
        ), mock.patch.object(
            nodes_module.comfy.model_management, "extra_reserved_memory", return_value=reserve
        ):
            preloaded = nodes_module.GGUFModelPatcherDynamic._preload_dynamic_vram_lora_factors(
                patcher
            )

        self.assertTrue(preloaded)
        self.assertEqual(len(moved), 2)
        self.assertIsNot(lora.weights, original_lora_weights)
        self.assertIsNot(lokr.weights, original_lokr_weights)

    def test_dynamic_vram_offloads_lora_factors_only_when_enabled(self):
        nodes_module = nodes_factory()
        factor = torch.ones((2, 2))
        adapter = type("FakeLoRAAdapter", (), {"name": "lora", "weights": (factor, 4.0)})()
        key = "blocks.1.weight"
        patch_function = type(
            "FakeLowVramPatch",
            (),
            {
                "is_lowvram_patch": True,
                "key": key,
                "patches": {key: [(1.0, adapter, 1.0, None, None)]},
                "prepared_patches": None,
            },
        )()
        module = type(
            "FakeLayer", (), {"weight_function": [patch_function], "bias_function": []}
        )()
        moved = []
        patcher = type(
            "FakeDynamicPatcher",
            (),
            {
                "model": type(
                    "FakeModel", (), {"named_modules": lambda self: [("layer", module)]}
                )(),
                "load_device": torch.device("cuda:0"),
                "offload_device": torch.device("cpu"),
                "_move_dynamic_vram_lora_factor": staticmethod(
                    lambda value, device: moved.append((value, device)) or value.clone()
                ),
            },
        )()
        reserve = 123456

        with mock.patch.dict(
            os.environ, {"COMFYUI_GGUF_INT4_LORA_OFFLOAD": "ON"}
        ), mock.patch.object(torch.cuda, "is_available", return_value=True), mock.patch.object(
            torch.cuda, "mem_get_info", return_value=(reserve + factor.numel() * factor.element_size() - 1, 8 * reserve)
        ), mock.patch.object(
            nodes_module.comfy.model_management, "extra_reserved_memory", return_value=reserve
        ):
            preloaded = nodes_module.GGUFModelPatcherDynamic._preload_dynamic_vram_lora_factors(
                patcher
            )

        self.assertFalse(preloaded)
        self.assertFalse(moved)
        self.assertIs(adapter.weights[0], factor)

    def test_int4_lora_offload_environment_values(self):
        nodes_module = nodes_factory()
        for value in ("true", "TRUE", "1", "yes", "On"):
            with mock.patch.dict(os.environ, {"COMFYUI_GGUF_INT4_LORA_OFFLOAD": value}):
                self.assertTrue(nodes_module.int4_lora_offload_enabled())
        for value in ("", "false", "0", "no", "off", "unexpected"):
            with mock.patch.dict(os.environ, {"COMFYUI_GGUF_INT4_LORA_OFFLOAD": value}):
                self.assertFalse(nodes_module.int4_lora_offload_enabled())
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(nodes_module.int4_lora_offload_enabled())

    def test_dynamic_vram_preload_logs_loaded_models_and_reraises_cuda_oom(self):
        nodes_module = nodes_factory()
        factor = torch.ones((2, 2))
        adapter = type("FakeLoRAAdapter", (), {"name": "lora", "weights": (factor, 4.0)})()
        key = "blocks.1.weight"
        patch_function = type(
            "FakeLowVramPatch",
            (),
            {
                "is_lowvram_patch": True,
                "key": key,
                "patches": {key: [(1.0, adapter, 1.0, None, None)]},
                "prepared_patches": None,
            },
        )()
        module = type(
            "FakeLayer", (), {"weight_function": [patch_function], "bias_function": []}
        )()
        patcher = type(
            "FakeDynamicPatcher",
            (),
            {
                "model": type(
                    "FakeModel", (), {"named_modules": lambda self: [("layer", module)]}
                )(),
                "load_device": torch.device("cuda:0"),
                "offload_device": torch.device("cpu"),
                "_move_dynamic_vram_lora_factor": staticmethod(
                    mock.Mock(side_effect=torch.OutOfMemoryError("forced preload OOM"))
                ),
            },
        )()
        loaded_model = type(
            "FakeLoadedModel",
            (),
            {
                "model": type(
                    "FakePatcher",
                    (),
                    {"model": type("DiffusionModel", (), {"name": "gguf"})(), "size": 3 * 1024**2},
                )()
            },
        )()

        with mock.patch.dict(
            os.environ, {"COMFYUI_GGUF_INT4_LORA_OFFLOAD": "false"}
        ), mock.patch.object(torch.cuda, "is_available", return_value=True), mock.patch.object(
            torch.cuda, "mem_get_info", return_value=(2 * 1024**3, 8 * 1024**3)
        ), mock.patch.object(
            nodes_module.comfy.model_management, "extra_reserved_memory", return_value=0
        ), mock.patch.object(
            nodes_module.comfy.model_management, "current_loaded_models", [loaded_model]
        ), self.assertLogs(level="ERROR") as logs:
            with self.assertRaisesRegex(torch.OutOfMemoryError, "forced preload OOM"):
                nodes_module.GGUFModelPatcherDynamic._preload_dynamic_vram_lora_factors(
                    patcher
                )

        self.assertIn("preloading Dynamic VRAM INT4 LoRA/LoKr factors", logs.output[0])
        self.assertIn("DiffusionModel (gguf): 3.0 MiB", logs.output[1])

    def test_dynamic_vram_does_not_warn_for_static_lora_patch(self):
        nodes_module = nodes_factory()
        module = type(
            "FakeLayer",
            (),
            {"weight_function": [object()], "bias_function": []},
        )()
        patcher = type(
            "FakeDynamicPatcher",
            (),
            {
                "model": type(
                    "FakeModel",
                    (),
                    {"named_modules": lambda self: [("layer", module)]},
                )(),
                "load_device": torch.device("cuda:0"),
                "offload_device": torch.device("cpu"),
            },
        )()

        self.assertFalse(
            nodes_module.GGUFModelPatcherDynamic._warn_dynamic_vram_lora_streaming(
                patcher
            )
        )

    def test_performance_log_records_quantized_forward(self):
        import sys

        ops_factory()
        ops_module = sys.modules["comfyui_gguf_test.ops"]
        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "performance.log"
            with mock.patch.dict(os.environ, {ops_module._PERF_LOG_ENV: str(log_path)}):
                logger = ops_module._configure_perf_logger()
                self.assertIsNotNone(logger)
                handler = logger.handlers[-1]
                previous_logger = ops_module._PERF_LOGGER
                ops_module._PERF_LOGGER = logger
                try:
                    class FakeLayer:
                        in_features = 16
                        out_features = 32

                    input_tensor = torch.zeros((2, 16), dtype=torch.float32)
                    result = ops_module._perf_forward(
                        "int4_convrot_w4a4",
                        FakeLayer(),
                        input_tensor,
                        lambda: input_tensor + 1,
                    )
                    self.assertTrue(torch.equal(result, torch.ones_like(input_tensor)))
                    handler.flush()
                    contents = log_path.read_text(encoding="utf-8")
                finally:
                    ops_module._PERF_LOGGER = previous_logger
                    logger.removeHandler(handler)
                    handler.close()

        self.assertIn("forward mode=int4_convrot_w4a4", contents)
        self.assertIn("elapsed_ms=", contents)
        self.assertIn("input_shape=(2, 16)", contents)

    def test_performance_log_is_opt_in(self):
        import sys

        ops_factory()
        ops_module = sys.modules["comfyui_gguf_test.ops"]
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(ops_module._configure_perf_logger())
        with mock.patch.dict(os.environ, {ops_module._PERF_LOG_ENV: ""}):
            self.assertIsNone(ops_module._configure_perf_logger())


class Q1CRXLQuantizationTests(unittest.TestCase):
    def test_int1_sign_packing_uses_little_endian_bits_and_group_scales(self):
        from q1_cr_triton_backend import pack_int1_sign_bits, unpack_int1_sign_bits

        signs = torch.tensor([0, 1] * 32, dtype=torch.float32)
        weight = torch.where(signs > 0, 2.0, -2.0).reshape(1, 64)
        packed, scales = pack_int1_sign_bits(weight)

        self.assertEqual(packed.shape, (1, 8))
        self.assertEqual(packed.dtype, torch.int8)
        self.assertEqual(packed.view(torch.uint8)[0, 0].item(), 0xAA)
        self.assertEqual(scales.shape, (1, 1))
        self.assertEqual(scales.dtype, torch.float16)
        self.assertEqual(scales.item(), 2.0)
        decoded = unpack_int1_sign_bits(packed, scales, dtype=torch.float32)
        torch.testing.assert_close(decoded, weight)

    def test_int1_cr_xl_quantization_and_linear_match_dense_reference(self):
        from q1_cr_triton_backend import (
            dequantize_q1_cr_xl_weight,
            q1_cr_xl_linear,
        )

        torch.manual_seed(17)
        weight = torch.randn(8, 512, dtype=torch.float32)
        packed, scales, quant_conf, orig_shape = quantize_int1_cr_xl(
            weight,
            device=torch.device("cpu"),
        )
        self.assertEqual(packed.shape, (8, 64))
        self.assertEqual(packed.dtype, torch.int8)
        self.assertEqual(scales.shape, (8, 8))
        self.assertEqual(scales.dtype, torch.float16)
        self.assertEqual(orig_shape, (8, 512))
        self.assertEqual(quant_conf["format"], "int1_cr")
        self.assertEqual(quant_conf["backing"], "w1a16")
        self.assertEqual(quant_conf["linear_dtype"], "bf16")
        self.assertEqual(quant_conf["convrot_groupsize"], 256)
        self.assertEqual(quant_conf["quant_group_size"], 64)

        dense = dequantize_q1_cr_xl_weight(
            packed,
            scales,
            orig_shape,
            dtype=torch.float32,
        )
        x = torch.randn(3, 512, dtype=torch.float32)
        bias = torch.randn(8, dtype=torch.float32)
        actual = q1_cr_xl_linear(
            x,
            packed,
            scales,
            bias=bias,
            orig_shape=orig_shape,
        )
        expected = torch.nn.functional.linear(x, dense, bias)
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_int1_cr_xl_cuda_triton_decode_matches_rotated_dense_reference(self):
        import q1_cr_triton_backend as backend

        if backend.triton is None:
            self.skipTest("Triton is unavailable")
        torch.manual_seed(29)
        weight = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
        packed, scales, _, orig_shape = quantize_int1_cr_xl(
            weight,
            device=torch.device("cuda"),
        )
        x = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(8, device="cuda", dtype=torch.bfloat16)
        actual = backend.q1_cr_xl_linear(
            x,
            packed,
            scales,
            bias=bias,
            orig_shape=orig_shape,
        )
        x_rotated = backend._rotate_activation(x, 256)
        weight_rotated = backend.unpack_int1_sign_bits(
            packed,
            scales,
            dtype=x.dtype,
        )
        expected = torch.nn.functional.linear(x_rotated, weight_rotated, bias)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_q1_cr_xl_conversion_loader_webui_and_qwen21_dispatch(self):
        state_dict = {
            "txt_in.text_norm.weight": torch.ones(16, dtype=torch.bfloat16),
            "modulation.1.weight": torch.ones((32, 16), dtype=torch.bfloat16),
            "transformer_blocks.0.attn.norm_q.weight": torch.ones(4, dtype=torch.bfloat16),
            "transformer_blocks.0.attn.to_q.weight": torch.randn((64, 512), dtype=torch.bfloat16),
            "transformer_blocks.0.img_mlp.gate_up.weight": torch.randn((64, 512), dtype=torch.bfloat16),
            "transformer_blocks.0.img_mlp.out.weight": torch.randn((64, 512), dtype=torch.bfloat16),
            "transformer_blocks.0.img_mlp.unaligned.weight": torch.randn((64, 300), dtype=torch.bfloat16),
            "img_in.weight": torch.ones((16, 64), dtype=torch.bfloat16),
            "proj_out.weight": torch.ones((64, 16), dtype=torch.bfloat16),
        }

        self.assertIn("Q1_CR_XL", QUANT_TYPE_MAP)
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "qwen-image-2.1-Q1_CR_XL.gguf"
            converted_path, model_arch = convert_state_dict(
                state_dict,
                str(output_path),
                quant_type_name="Q1_CR_XL",
                quantization_device="cpu",
            )
            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            q1_field = reader.get_field(
                "comfy.gguf.quant.transformer_blocks.0.img_mlp.gate_up.weight"
            )
            q1_conf = json.loads(str(q1_field.parts[q1_field.data[-1]], "utf-8"))
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

            self.assertIsInstance(model_arch, ModelQwenImage21)
            self.assertEqual(
                tensor_types["transformer_blocks.0.attn.to_q.weight"],
                gguf.GGMLQuantizationType.I8,
            )
            self.assertEqual(
                tensor_types["transformer_blocks.0.attn.to_q.weight_scale"],
                gguf.GGMLQuantizationType.F32,
            )
            self.assertEqual(
                tensor_types["transformer_blocks.0.img_mlp.gate_up.weight"],
                gguf.GGMLQuantizationType.I8,
            )
            self.assertEqual(
                tensor_types["transformer_blocks.0.img_mlp.gate_up.weight_scale"],
                gguf.GGMLQuantizationType.F16,
            )
            self.assertEqual(
                tensor_types["transformer_blocks.0.img_mlp.unaligned.weight"],
                gguf.GGMLQuantizationType.F16,
            )
            self.assertEqual(q1_conf["orig_shape"], [64, 512])
            self.assertEqual(q1_conf["quant_group_size"], 64)

            loader = load_gguf_loader()
            loaded, extra = loader.gguf_sd_loader(converted_path)
            self.assertEqual(extra["gguf_quant_mode"], "int1_cr_xl")
            self.assertIn("int1_cr", extra["gguf_quant_formats"])
            dynamic_loaded, dynamic_extra = loader.gguf_sd_loader(
                converted_path,
                dynamic=True,
            )
            self.assertEqual(dynamic_extra["gguf_quant_mode"], "int1_cr_xl")
            del dynamic_loaded, dynamic_extra
            gc.collect()
            ops_type = nodes_factory()._legacy_gguf_ops(extra)
            q1_layer = ops_type.Linear(512, 64, bias=False)
            prefix = "transformer_blocks.0.img_mlp.gate_up."
            q1_layer._load_from_state_dict(
                loaded,
                prefix=prefix,
                local_metadata={},
                strict=True,
                missing_keys=[],
                unexpected_keys=[],
                error_msgs=[],
            )
            self.assertEqual(q1_layer._quantized_format, "int1_cr")
            self.assertEqual(tuple(q1_layer.weight.shape), (64, 64))
            self.assertEqual(tuple(q1_layer.weight_scale.shape), (64, 8))

            x = torch.randn(2, 512, dtype=torch.float32)
            actual = q1_layer(x)
            expected = torch.nn.functional.linear(
                x,
                q1_layer._dequantized_weight(torch.device("cpu"), x.dtype),
            )
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)

            del q1_layer, loaded
            gc.collect()

        node_module = nodes_factory()
        choices = node_module.TargetedQuantizationGGUF.INPUT_TYPES()["required"]["quantization"][0]
        self.assertIn("Q1_CR_XL", choices)
        from tools import conversion_webui
        page = conversion_webui.PAGE.read_text(encoding="utf-8")
        self.assertIn('name="quant_type" value="Q1_CR_XL"', page)
        self.assertIn('<option value="Q1_CR_XL">', page)

    def test_q1_cr_xl_supports_native_and_fused_lora_patches(self):
        from comfy.weight_adapter import LoRAAdapter
        from q1_cr_triton_backend import pack_int1_sign_bits

        ops_type = ops_factory()
        layer = ops_type.Linear(512, 8, bias=False)
        weight = torch.randn(8, 512, dtype=torch.float32)
        from tools.convert import _build_regular_hadamard
        hadamard = _build_regular_hadamard(256, dtype=torch.float32, device="cpu")
        rotated = (
            weight.reshape(8, 2, 256) @ hadamard.T
        ).reshape(8, 512)
        packed, scales = pack_int1_sign_bits(rotated)
        quant_conf = {
            "format": "int1_cr",
            "backing": "w1a16",
            "linear_dtype": "bf16",
            "convrot_groupsize": 256,
            "quant_group_size": 64,
            "orig_shape": [8, 512],
            "weight_rotated": True,
        }
        layer._load_from_state_dict(
            {
                "weight": packed,
                "weight_scale": scales,
                "comfy_quant": torch.tensor(
                    list(json.dumps(quant_conf).encode("utf-8")),
                    dtype=torch.uint8,
                ),
            },
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )

        rank = 4
        up = torch.randn(8, rank, dtype=torch.float32)
        down = torch.randn(rank, 512, dtype=torch.float32)
        adapter = LoRAAdapter(set(), (up, down, float(rank), None, None, None))
        key = "blocks.0.weight"
        patches = {key: [(0.5, adapter, 1.0, None, None)]}

        class FakeLowVramPatch:
            is_lowvram_patch = True

            def __init__(self):
                self.key = key
                self.patches = patches
                self.prepared_patches = None

            def __call__(self, value):
                return value

        x = torch.randn(3, 512, dtype=torch.float32)
        base = layer(x)
        layer.weight_function = [FakeLowVramPatch()]
        native = layer(x)
        expected_native = base + 0.5 * torch.nn.functional.linear(
            torch.nn.functional.linear(x, down),
            up,
        )
        torch.testing.assert_close(native, expected_native, atol=2e-4, rtol=2e-4)
        self.assertIsNone(layer._fused_weight)

        layer.weight_function = [lambda value, *args, **kwargs: value + 0.125]
        fused = layer(x)
        self.assertIsNotNone(layer._fused_weight)
        expected_fused = torch.nn.functional.linear(
            x,
            layer._fused_weight.to(dtype=x.dtype),
        )
        torch.testing.assert_close(fused, expected_fused, atol=2e-4, rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
