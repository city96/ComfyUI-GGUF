import unittest
import json
import gc
import os
from contextlib import nullcontext
from collections import OrderedDict
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import gguf
import torch
import comfy.sd
from safetensors.torch import save_file

from tools.convert import (
    MEBIBYTE,
    ModelLTXV,
    ModelLTXVUpsampler,
    ModelMinimaxH3,
    ModelMinimaxH3VAE,
    ModelMiniMaxMusic3DiT,
    ModelMiniMaxMusic3TextEncoder,
    ModelTemplate,
    _streamed_safetensors_layout,
    convert_file,
    convert_state_dict,
    detect_arch,
    plan_target_size_quantization,
    quantize_int8_convrot,
    quantize_int4_cr_w4a4,
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


def ops_factory():
    """Load the repo's ops.py and return get_gguf_q4_w4a4_ops."""
    import sys
    ops_path = Path(__file__).parents[1] / "ops.py"
    package_name = "comfyui_gguf_test"
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.ops",
        ops_path,
        submodule_search_locations=[str(ops_path.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{package_name}.ops"] = module
    spec.loader.exec_module(module)
    return module.get_gguf_q4_w4a4_ops(torch.bfloat16)


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
            gguf.GGMLQuantizationType.I8,
        )
        self.assertEqual(
            tensor_types["blocks.0.attn.qkv_proj.weight_scale"],
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
            "blocks.0.proj.comfy_quant": torch.tensor(
                list(json.dumps(quant_conf).encode("utf-8")), dtype=torch.uint8
            ),
        }

        materialize_int8_source_weights(state_dict)

        self.assertNotIn("blocks.0.proj.weight_scale", state_dict)
        self.assertNotIn("blocks.0.proj.comfy_quant", state_dict)
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
            gguf.GGMLQuantizationType.BF16,
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
            gguf.GGMLQuantizationType.Q8_0,
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


if __name__ == "__main__":
    unittest.main()
