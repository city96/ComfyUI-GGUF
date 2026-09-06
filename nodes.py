# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import torch
import logging
import inspect
import collections
import json
import os
from contextlib import nullcontext

import nodes
import comfy.sd
import comfy.lora
import comfy.float
import comfy.ops
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import comfy.memory_management
import folder_paths

from .ops import (
    GGMLTensor,
    GGMLOps,
    get_gguf_q8_ops,
    get_gguf_q4_w4a4_ops,
    int4_lora_offload_enabled,
    log_cuda_oom_loaded_models,
    move_patch_to_device,
)
from .loader import gguf_sd_loader, gguf_clip_loader, gguf_tensor_count
from .dequant import dequantize_tensor, is_quantized, is_torch_compatible
from .tools.convert import (
    DEFAULT_TARGET_SIZE_Q8_TYPE,
    QUANT_TYPE_MAP,
    QUANTIZATION_DEVICE_OPTIONS,
    TARGET_SIZE_Q8_TYPES,
    TARGET_SIZE_QUANT_TYPE,
    convert_file,
)
from .lora import load_gguf_lora


_DYNAMIC_VRAM_LORA_WARNING_MIN_BYTES = 64 * 1024 * 1024


def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")

# Add a custom keys for files ending in .gguf
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])
update_folder_names_and_paths("lora_gguf", ["loras"])
update_folder_names_and_paths("vae_gguf", ["vae"])
update_folder_names_and_paths("latent_upscale_models_gguf", ["latent_upscale_models"])


class GGUFLoadProgress:
    """Coordinate one ComfyUI progress bar across one or more model files."""

    def __init__(self, paths):
        self.path_totals = {
            path: gguf_tensor_count(path) if path.endswith(".gguf") else 1
            for path in paths
        }
        self.total = sum(self.path_totals.values())
        self.pbar = comfy.utils.ProgressBar(self.total)
        self.completed = 0

    def callback_for(self, path):
        offset = self.completed
        total = self.path_totals[path]

        def update(current, _loader_total):
            self.pbar.update_absolute(offset + current, self.total)

        return update

    def complete_file(self, path):
        self.completed += self.path_totals[path]
        self.pbar.update_absolute(self.completed, self.total)

class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False

    def _evict_gguf_quantized_caches(self):
        for module in self.model.modules():
            evict = getattr(module, "evict_quantized_caches", None)
            if evict is not None:
                evict()

    def _move_gguf_quantized_caches(self, device):
        for module in self.model.modules():
            move = getattr(module, "move_fused_caches", None)
            if move is not None:
                move(device)

    def _prepare_gguf_quantized_weights(self, device=None):
        """Eagerly prepare fallback caches for active, non-native INT4 patches."""
        prepared = []
        layout = []
        active_modules = []
        for name, module in self.model.named_modules():
            signature = getattr(module, "_fused_patch_signature", None)
            prepare = getattr(module, "prepare_fused_weight", None)
            if signature is None or prepare is None:
                continue
            module_signature = signature()
            layout.append((name, module_signature))
            if getattr(module, "weight_function", ()) or getattr(module, "bias_function", ()):
                active_modules.append((name, module))

        layout = tuple(layout)
        previous_layout = getattr(self, "_gguf_patch_layout_signature", None)
        if previous_layout is not None and previous_layout != layout:
            self._evict_gguf_quantized_caches()
        self._gguf_patch_layout_signature = layout

        if device is None:
            # Fallback cache preparation is compute-heavy. Use the execution
            # device rather than Dynamic VRAM's commonly-CPU offload device.
            device = getattr(self, "load_device", None)
        if device is None:
            device = getattr(self, "offload_device", None)
        device = torch.device(device) if device is not None else torch.device("cpu")
        if not active_modules:
            return 0

        interrupt = getattr(comfy.model_management, "throw_exception_if_processing_interrupted", None)
        cuda_context = getattr(comfy.model_management, "cuda_device_context", None)
        device_context = cuda_context(device) if callable(cuda_context) else nullcontext()
        try:
            with device_context:
                for _, module in active_modules:
                    if interrupt is not None:
                        interrupt()
                    if module.prepare_fused_weight(device):
                        prepared.append(module)
                        # Fallback-patched INT4 layers retain a full-precision matrix
                        # for correctness. Stage it on the offload device so all
                        # layers do not accumulate on the execution GPU at load time.
                        move = getattr(module, "move_fused_caches", None)
                        offload_device = getattr(self, "offload_device", None)
                        if move is not None and offload_device is not None and device != offload_device:
                            move(offload_device)
        except BaseException:
            # Do not leave a mixture of old and newly fused representations after
            # an interrupt or a failed adapter preparation.
            self._evict_gguf_quantized_caches()
            raise
        return len(prepared)

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)

        patches = self.patches[key]
        # Legacy Q4_CR (INT4) path: the base weight stays packed (plain
        # Parameter) and adapters run through the module's weight_function
        # (native low-rank bypass or fused fallback). Fusing directly with
        # calculate_weight here would corrupt the packed INT4 bytes.
        # Standard GGML quants and all other tensors keep their paths below
        # unchanged.
        if key.rsplit('.', 1)[-1] == "weight" and '.' in key:
            module = None
            try:
                module = comfy.utils.get_attr(self.model, key.rsplit('.', 1)[0])
            except Exception:
                module = None
            install_patch = getattr(module, "install_patch_entries", None)
            if install_patch is not None and bool(getattr(module, "_quantized", False)):
                out_weight = weight.to(device_to)
                patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
                install_patch(patches, key)
                logging.warning(f"Q4_CR type GGUF is best used with the f{UnetLoaderGGUFDynamicVRAM.TITLE} loader node, because using it with Legacy VRAM is much slower.")
                if inplace_update:
                    comfy.utils.copy_to_param(self.model, key, out_weight)
                else:
                    comfy.utils.set_attr_param(self.model, key, out_weight)
                return
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
            module = comfy.utils.get_attr(self.model, key.rsplit('.', 1)[0])
            install_patch = getattr(module, "install_patch_entries", None)
            if install_patch is not None:
                install_patch(patches, key)
            # TODO: do we ever have legitimate duplicate patches? (i.e. patch on top of patched weight)
            out_weight.patches = [(patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                    weight.to(device=self.offload_device, copy=inplace_update), inplace_update
                )

            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = comfy.lora.calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            self._evict_gguf_quantized_caches()
            self._gguf_patch_layout_signature = None
            for module in self.model.modules():
                if getattr(module, "_gguf_static_patch", False):
                    module.weight_function = []
                    module._gguf_static_patch = False
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        # TODO: Find another way to not unload after patches
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)


    def pin_weight_to_device(self, key):
        op_key = key.rsplit('.', 1)[0]
        if not self.mmap_released and op_key in self.named_modules_to_munmap:
            # TODO: possible to OOM, find better way to detach
            self.named_modules_to_munmap[op_key].to(self.load_device).to(self.offload_device)
            del self.named_modules_to_munmap[op_key]
        super().pin_weight_to_device(key)

    mmap_released = False
    named_modules_to_munmap = {}

    def partially_unload(self, *args, force_patch_weights=False, **kwargs):
        result = super().partially_unload(*args, force_patch_weights=True, **kwargs)
        if args:
            self._move_gguf_quantized_caches(args[0])
        elif "device_to" in kwargs:
            self._move_gguf_quantized_caches(kwargs["device_to"])
        return result

    def partially_load(self, *args, force_patch_weights=False, **kwargs):
        return super().partially_load(*args, force_patch_weights=True, **kwargs)

    def load(self, *args, force_patch_weights=False, **kwargs):
        if not self.mmap_released:
            self.named_modules_to_munmap = dict(self.model.named_modules())

        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)
        self._prepare_gguf_quantized_weights()

        # make sure nothing stays linked to mmap after first load
        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.named_modules_to_munmap.items():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked and self.load_device != self.offload_device:
                logging.info(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    # TODO: possible to OOM, find better way to detach
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True
            self.named_modules_to_munmap = {}

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        # GGUF specific clone values below
        n.patch_on_device = getattr(self, "patch_on_device", False)
        n.mmap_released = getattr(self, "mmap_released", False)
        n._gguf_patch_layout_signature = getattr(self, "_gguf_patch_layout_signature", None)
        if src_cls != GGUFModelPatcher:
            n.size = 0 # force recalc
        return n

class UnetLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "bootleg"
    TITLE = "Unet Loader (GGUF, Legacy node)"

    def load_unet(self, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None):
        unet_path = folder_paths.get_full_path("unet", unet_name)
        progress = GGUFLoadProgress([unet_path])
        sd, extra = gguf_sd_loader(unet_path, progress_callback=progress.callback_for(unet_path))
        progress.complete_file(unet_path)

        mode = extra.get("gguf_quant_mode")
        if mode == "int8_convrot":
            # Use ComfyUI native INT8 path (weights stay INT8)
            ops = get_gguf_q8_ops(compute_dtype=torch.bfloat16)()
        elif mode == "int4_pytorch":
            raise RuntimeError(
                "Q4_PT is retired because PyTorch's Ampere INT4 kernel is not "
                "performance-competitive. Reconvert the model as Q8_CR."
            )
        elif mode == "int4_cr_w4a4":
            # Q4_CR_W4A4: custom W4A4 INT4 backed by comfy_kitchen's fast
            # ConvRot int4 tensor-core MMA.
            ops = get_gguf_q4_w4a4_ops(compute_dtype=torch.bfloat16)()
        else:
            ops = GGMLOps()

        if dequant_dtype in ("default", None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ["target"]:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

        if patch_dtype in ("default", None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ["target"]:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)

        # init model

        kwargs = {}
        valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
        if "metadata" in valid_params:
            kwargs["metadata"] = extra.get("metadata", {})

        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}, **kwargs,
        )
        if model is None:
            logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
            raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device
        return (model,)


class VAELoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae_gguf"),),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "bootleg"
    TITLE = "VAE Loader (GGUF, Legacy node)"

    def load_vae(self, vae_name):
        vae_path = folder_paths.get_full_path("vae", vae_name)
        progress = GGUFLoadProgress([vae_path])
        sd, extra = gguf_sd_loader(
            vae_path,
            handle_prefix=None,
            progress_callback=progress.callback_for(vae_path),
        )
        progress.complete_file(vae_path)

        if extra["arch_str"] != "minimax_h3_vae":
            raise ValueError(
                "VAE Loader (GGUF) currently supports only MiniMax H3 video VAE GGUF files."
            )
        if extra.get("gguf_quant_mode") != "int8_convrot":
            raise ValueError(
                "MiniMax H3 VAE GGUF must use Q8_CR so decoder Linear weights stay on the native INT8 path."
            )

        for key, value in tuple(sd.items()):
            if isinstance(value, GGMLTensor):
                sd[key] = dequantize_tensor(value, dtype=torch.Tensor(value).dtype)

        operations = get_gguf_q8_ops(compute_dtype=torch.float16)()
        vae_kwargs = {"sd": sd, "metadata": extra.get("metadata", {})}
        vae_init_params = inspect.signature(comfy.sd.VAE.__init__).parameters
        if "operations" in vae_init_params:
            vae_kwargs["operations"] = operations
            if "disable_dynamic" in vae_init_params:
                vae_kwargs["disable_dynamic"] = True
            vae = comfy.sd.VAE(**vae_kwargs)
        else:
            minimax_vae = comfy.ldm.minimax.vae
            if not hasattr(minimax_vae, "ops"):
                raise RuntimeError(
                    "This ComfyUI version cannot inject Q8_CR operations into MiniMax H3 VAE."
                )
            original_operations = minimax_vae.ops
            minimax_vae.ops = operations
            try:
                vae = comfy.sd.VAE(**vae_kwargs)
            finally:
                minimax_vae.ops = original_operations
        vae.throw_exception_if_invalid()
        return (vae,)


class LTXVLatentUpscaleModelLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("latent_upscale_models_gguf"),
                ),
            }
        }

    RETURN_TYPES = ("LATENT_UPSCALE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "bootleg"
    TITLE = "LTXV Latent Upscale Model Loader (GGUF)"

    def load_model(self, model_name):
        model_path = folder_paths.get_full_path(
            "latent_upscale_models_gguf", model_name
        )
        progress = GGUFLoadProgress([model_path])
        state_dict, extra = gguf_sd_loader(
            model_path,
            handle_prefix=None,
            progress_callback=progress.callback_for(model_path),
        )
        progress.complete_file(model_path)

        if extra["arch_str"] != "ltxv_upscaler":
            raise ValueError(
                "LTXV Latent Upscale Model Loader (GGUF) requires an LTX 2.5 "
                "latent upscaler GGUF file."
            )
        config_json = extra["metadata"].get("config")
        if config_json is None:
            raise ValueError("LTX 2.5 latent upscaler GGUF is missing its config metadata.")

        from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler

        config = json.loads(config_json)
        model = LatentUpsampler.from_config(config, operations=GGMLOps)
        model_dtype = comfy.model_management.vae_dtype(
            allowed_dtypes=[torch.bfloat16, torch.float32]
        )
        model = model.to(dtype=model_dtype)
        comfy.model_management.archive_model_dtypes(model)
        model_patcher = comfy.model_patcher.CoreModelPatcher(
            model,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
        )
        model.load_state_dict(state_dict, assign=model_patcher.is_dynamic())
        return (model_patcher,)


class TargetedQuantizationGGUF:
    """Convert a source checkpoint to GGUF from a ComfyUI workflow."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Absolute path to a .safetensors, .ckpt, .pt, .bin, or .pth source model.",
                    },
                ),
                "destination_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Output .gguf path. Leave empty to derive it from the source path.",
                    },
                ),
                "quantization": (
                    [TARGET_SIZE_QUANT_TYPE, *QUANT_TYPE_MAP.keys()],
                    {
                        "default": TARGET_SIZE_QUANT_TYPE,
                        "tooltip": (
                            "TARGET_SIZE starts at the selected Q8 type, reduces central core matrices "
                            "to Q5_0 then Q4_0, then ordinary 1-D tensors to BF16 only when necessary."
                        ),
                    },
                ),
                "max_size_mb": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1000000.0,
                        "step": 1.0,
                        "tooltip": "Maximum output size in MiB. Required only for TARGET_SIZE.",
                    },
                ),
                "target_size_q8_type": (
                    list(TARGET_SIZE_Q8_TYPES),
                    {
                        "default": DEFAULT_TARGET_SIZE_Q8_TYPE,
                        "tooltip": (
                            "TARGET_SIZE baseline: Q8_CR uses native INT8 ConvRot; "
                            "Q8_0 uses standard GGUF Q8 before layers are reduced to Q4_0."
                        ),
                    },
                ),
                "quantization_device": (
                    list(QUANTIZATION_DEVICE_OPTIONS),
                    {
                        "default": "auto",
                        "tooltip": (
                            "Q8_CR conversion device. auto uses CUDA when available and "
                            "falls back to CPU per matrix when VRAM is insufficient."
                        ),
                    },
                ),
                "overwrite": ("BOOLEAN", {"default": False}),
                "streamed": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "For safetensors sources, process one tensor at a time and stage "
                            "GGUF data on disk to reduce RAM and VRAM usage."
                        ),
                    },
                ),
            },
            "optional": {
                "lora_paths": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Absolute .safetensors or .gguf LoRA paths, one per line or comma-separated.",
                    },
                ),
                "lora_strengths": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma-separated merge strengths matching lora_paths; blank uses 1.0.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("gguf_path", "quantization_info")
    FUNCTION = "quantize"
    CATEGORY = "bootleg/quantization"
    TITLE = "Targeted Quantization (GGUF)"

    def quantize(
        self,
        source_path,
        destination_path,
        quantization,
        max_size_mb,
        target_size_q8_type,
        quantization_device,
        overwrite,
        streamed=False,
        lora_paths="",
        lora_strengths="",
    ):
        source_path = os.path.abspath(os.path.expanduser(source_path))
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source model does not exist: {source_path}")

        if quantization == TARGET_SIZE_QUANT_TYPE and max_size_mb <= 0:
            raise ValueError("TARGET_SIZE requires max_size_mb greater than zero.")
        lora_paths = _parse_lora_paths(lora_paths) if lora_paths.strip() else []
        lora_strengths = _parse_strengths(lora_strengths, len(lora_paths)) if lora_paths else []

        progress = {"bar": None, "read_total": None, "total": None}

        def report_progress(stage, current, total):
            if stage == "read":
                if progress["bar"] is None:
                    progress["read_total"] = total
                    progress["total"] = total * 2 + 1
                    progress["bar"] = comfy.utils.ProgressBar(progress["total"])
                progress["bar"].update_absolute(current, progress["total"])
            elif stage == "quantize":
                if progress["bar"] is None:
                    progress["read_total"] = 0
                    progress["total"] = total + 1
                    progress["bar"] = comfy.utils.ProgressBar(progress["total"])
                progress["bar"].update_absolute(progress["read_total"] + current, progress["total"])

        output_path, _ = convert_file(
            source_path,
            dst_path=destination_path or None,
            interact=False,
            overwrite=overwrite,
            quant_type_name=None if quantization == TARGET_SIZE_QUANT_TYPE else quantization,
            max_size_mb=max_size_mb if quantization == TARGET_SIZE_QUANT_TYPE else None,
            target_size_q8_type=target_size_q8_type,
            quantization_device=quantization_device,
            progress_callback=report_progress,
            lora_paths=lora_paths,
            lora_strengths=lora_strengths,
            streamed=streamed,
        )
        if progress["bar"] is not None:
            progress["bar"].update_absolute(progress["total"], progress["total"])

        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        info = f"{quantization}: {output_size_mb:.2f} MiB written to {output_path}"
        return (output_path, info)


def _gguf_lora_path(lora_name):
    path = folder_paths.get_full_path("loras", lora_name)
    if path is None:
        raise FileNotFoundError(f"GGUF LoRA does not exist: {lora_name}")
    if not path.lower().endswith(".gguf"):
        raise ValueError(f"GGUF LoRA Import requires a .gguf file, got {lora_name!r}.")
    return path


def _gguf_lora_key_map(model, clip):
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    return key_map


def _remap_gguf_lora_for_comfy(targets, key_map):
    lora = {}
    missing = []
    for target_name, target in targets.items():
        candidates = (
            target_name,
            f"diffusion_model.{target_name}",
            f"text_encoders.{target_name}",
            target_name.removeprefix("diffusion_model."),
            target_name.removeprefix("text_encoders."),
        )
        mapped_name = next((candidate for candidate in candidates if candidate in key_map), None)
        if mapped_name is None:
            missing.append(target["base_name"])
            continue
        lora[f"{mapped_name}.lora_A.weight"] = target["down"]
        lora[f"{mapped_name}.lora_B.weight"] = target["up"]
        if target["alpha"] is not None:
            lora[f"{mapped_name}.alpha"] = torch.tensor(
                target["alpha"], dtype=torch.float32
            )
    if missing:
        raise ValueError(
            "GGUF LoRA targets do not match the connected MODEL/CLIP: "
            + ", ".join(sorted(missing))
        )
    return lora


class GGUFLoraImport:
    """Load standard GGUF LoRA factors through ComfyUI's normal patch mechanism."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("lora_gguf"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            },
            "optional": {
                "clip": ("CLIP",),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "bootleg/LoRA"
    TITLE = "Load LoRA (GGUF)"

    def load_lora(self, model, lora_name, strength_model, clip=None, strength_clip=1.0):
        path = _gguf_lora_path(lora_name)
        _, targets, metadata = load_gguf_lora(path)
        key_map = _gguf_lora_key_map(model, clip)
        lora = _remap_gguf_lora_for_comfy(targets, key_map)
        return comfy.sd.load_lora_for_models(
            model,
            clip,
            lora,
            strength_model,
            strength_clip,
            lora_metadata={"gguf_lora": metadata},
        )


def _parse_lora_paths(lora_paths):
    paths = [
        os.path.abspath(os.path.expanduser(path.strip()))
        for path in lora_paths.replace(",", "\n").splitlines()
        if path.strip()
    ]
    if not paths:
        raise ValueError("Provide at least one LoRA path.")
    for path in paths:
        if not path.lower().endswith((".gguf", ".safetensors")):
            raise ValueError(f"LoRA fusion accepts only .gguf or .safetensors adapters, got {path!r}.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LoRA does not exist: {path}")
    return paths


def _parse_strengths(strengths, count):
    values = [value.strip() for value in strengths.replace("\n", ",").split(",") if value.strip()]
    if not values:
        return [1.0] * count
    if len(values) != count:
        raise ValueError(
            f"Expected {count} LoRA strength value(s), received {len(values)}."
        )
    try:
        return [float(value) for value in values]
    except ValueError as error:
        raise ValueError("LoRA strengths must be comma-separated numbers.") from error


def _require_dynamic_vram():
    if not comfy.memory_management.aimdo_enabled:
        raise RuntimeError(
            "Dynamic VRAM is not enabled in this ComfyUI installation. "
            "Start ComfyUI without --disable-dynamic-vram and use a build that supports DynamicVRAM."
        )


def _clone_as_dynamic_gguf_patcher(model_patcher):
    source_class = model_patcher.__class__
    model_patcher.__class__ = GGUFModelPatcherDynamic
    cloned = model_patcher.clone()
    model_patcher.__class__ = source_class
    return cloned


def _legacy_gguf_ops(extra):
    mode = extra.get("gguf_quant_mode")
    if mode == "int8_convrot":
        return get_gguf_q8_ops(compute_dtype=torch.bfloat16)()
    if mode == "int4_pytorch":
        raise RuntimeError(
            "Q4_PT is retired because PyTorch's Ampere INT4 kernel is not "
            "performance-competitive. Reconvert the model as Q8_CR."
        )
    if mode == "int4_cr_w4a4":
        return get_gguf_q4_w4a4_ops(compute_dtype=torch.bfloat16)()
    return GGMLOps()


def _load_dynamic_gguf_unet(unet_path, disable_dynamic=False, progress=None):
    if not disable_dynamic:
        _require_dynamic_vram()
    progress = progress or GGUFLoadProgress([unet_path])
    sd, extra = gguf_sd_loader(
        unet_path,
        dynamic=not disable_dynamic,
        progress_callback=progress.callback_for(unet_path),
    )
    progress.complete_file(unet_path)

    kwargs = {}
    valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
    if "metadata" in valid_params:
        kwargs["metadata"] = extra.get("metadata", {})

    # Target-size models combine native Q8_CR layers with standard GGML Q4_0
    # layers. Dynamic VRAM's default mixed-precision Linear cannot serialize a
    # bare GGML Q4_0 weight, so use the GGUF Q8 ops for both paths. Those ops
    # retain Q8_CR metadata and materialize only standard GGML layers as needed.
    model_options = {
        "custom_operations": _legacy_gguf_ops(extra),
    }
    model = comfy.sd.load_diffusion_model_state_dict(
        sd,
        model_options=model_options,
        disable_dynamic=disable_dynamic,
        **kwargs,
    )
    if model is None:
        logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
    model = GGUFModelPatcher.clone(model) if disable_dynamic else _clone_as_dynamic_gguf_patcher(model)
    model.cached_patcher_init = (_load_dynamic_gguf_unet, (unet_path,))
    return model


class GGUFModelPatcherDynamic(comfy.model_patcher.ModelPatcherDynamic):
    _evict_gguf_quantized_caches = GGUFModelPatcher._evict_gguf_quantized_caches
    _move_gguf_quantized_caches = GGUFModelPatcher._move_gguf_quantized_caches
    _prepare_gguf_quantized_weights = GGUFModelPatcher._prepare_gguf_quantized_weights

    def _dynamic_vram_lora_cpu_factors(self):
        """Return CPU LoRA/LoKr factors and adapters from active low-VRAM patches."""
        execution_device = torch.device(getattr(self, "load_device", "cpu"))
        offload_device = torch.device(getattr(self, "offload_device", "cpu"))
        if execution_device.type != "cuda" or offload_device.type != "cpu":
            return execution_device, offload_device, {}, ()

        factors = {}
        adapters = {}
        for _, module in self.model.named_modules():
            functions = (
                *getattr(module, "weight_function", ()),
                *getattr(module, "bias_function", ()),
            )
            for patch_function in functions:
                if not getattr(patch_function, "is_lowvram_patch", False):
                    continue
                patches = getattr(patch_function, "patches", None)
                key = getattr(patch_function, "key", None)
                patch_entries = getattr(patch_function, "prepared_patches", None)
                if patch_entries is None and patches is not None and key is not None:
                    patch_entries = patches.get(key)
                if patch_entries is None:
                    continue
                for patch in patch_entries:
                    if len(patch) < 2:
                        continue
                    adapter = patch[1]
                    if getattr(adapter, "name", None) not in {"lora", "lokr"}:
                        continue
                    has_cpu_factor = False
                    for factor in getattr(adapter, "weights", ()):
                        if isinstance(factor, torch.Tensor) and factor.device.type == "cpu":
                            factors[id(factor)] = factor
                            has_cpu_factor = True
                    if has_cpu_factor:
                        adapters[id(adapter)] = adapter
        return execution_device, offload_device, factors, tuple(adapters.values())

    def _move_dynamic_vram_lora_factor(self, factor, device):
        return factor.to(device)

    def _preload_dynamic_vram_lora_factors(self):
        """Atomically place active low-VRAM LoRA/LoKr factors on the execution GPU."""
        execution_device, _, factors, adapters = (
            GGUFModelPatcherDynamic._dynamic_vram_lora_cpu_factors(self)
        )
        if not factors or not torch.cuda.is_available():
            return False

        try:
            free_bytes, _ = torch.cuda.mem_get_info(execution_device)
        except RuntimeError:
            return False

        factor_bytes = sum(
            factor.numel() * factor.element_size() for factor in factors.values()
        )
        if (
            int4_lora_offload_enabled()
            and free_bytes - factor_bytes < comfy.model_management.extra_reserved_memory()
        ):
            return False

        transferred = {}
        try:
            for factor in factors.values():
                transferred[id(factor)] = self._move_dynamic_vram_lora_factor(
                    factor, execution_device
                )
        except torch.OutOfMemoryError:
            transferred.clear()
            log_cuda_oom_loaded_models("preloading Dynamic VRAM INT4 LoRA/LoKr factors")
            raise

        for adapter in adapters:
            adapter.weights = tuple(
                transferred.get(id(weight), weight)
                if isinstance(weight, torch.Tensor)
                else weight
                for weight in adapter.weights
            )
        logging.info(
            "Dynamic VRAM preloaded %d unique LoRA/LoKr factor tensors (%.1f MiB) on %s.",
            len(factors),
            factor_bytes / 1024**2,
            execution_device,
        )
        return True

    def _warn_dynamic_vram_lora_streaming(self):
        """Warn when active low-VRAM LoRA/LoKr factors must stream to CUDA repeatedly."""
        if not int4_lora_offload_enabled():
            return False
        execution_device, offload_device, factors, _ = (
            GGUFModelPatcherDynamic._dynamic_vram_lora_cpu_factors(self)
        )
        if execution_device.type != "cuda" or offload_device.type != "cpu":
            return False

        factor_bytes = sum(factor.numel() * factor.element_size() for factor in factors.values())
        if factor_bytes < _DYNAMIC_VRAM_LORA_WARNING_MIN_BYTES:
            return False

        signature = (
            tuple(sorted((id(factor), factor.numel(), factor.element_size()) for factor in factors.values())),
            str(execution_device),
            str(offload_device),
        )
        if getattr(self, "_gguf_lora_stream_warning_signature", None) == signature:
            return False

        memory_report = "CUDA memory telemetry unavailable"
        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(execution_device)
            except RuntimeError:
                pass
            else:
                memory_report = (
                    f"CUDA free/total {free_bytes / 1024**3:.1f}/{total_bytes / 1024**3:.1f} GiB"
                )
        logging.warning(
            "Dynamic VRAM runtime LoRA/LoKr warning: %.1f MiB across %d unique offloaded "
            "LoRA/LoKr factor tensors may be repeatedly streamed from %s to %s (%s). "
            "This can cause severe slowdown and increase CUDA OOM risk, but does not "
            "guarantee an OOM.",
            factor_bytes / 1024**2,
            len(factors),
            offload_device,
            execution_device,
            memory_report,
        )
        self._gguf_lora_stream_warning_signature = signature
        return True

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            self._evict_gguf_quantized_caches()
            self._gguf_patch_layout_signature = None
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        result = super().partially_unload(
            device_to, memory_to_free=memory_to_free, force_patch_weights=force_patch_weights
        )
        self._move_gguf_quantized_caches(device_to)
        return result

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)
        # GGML weights cannot be requantized after applying a LoRA patch.
        for _, module in self.model.named_modules():
            for param_key in ("weight", "bias"):
                attr = f"{param_key}_lowvram_function"
                lowvram_function = getattr(module, attr, None)
                if lowvram_function is not None:
                    setattr(module, attr, None)
                    functions = getattr(module, f"{param_key}_function", [])
                    functions.append(lowvram_function)
                    setattr(module, f"{param_key}_function", functions)
        self._preload_dynamic_vram_lora_factors()
        self._warn_dynamic_vram_lora_streaming()
        self._prepare_gguf_quantized_weights()

    def clone(self, disable_dynamic=False, model_override=None):
        if disable_dynamic:
            if model_override is None:
                fallback = self.cached_patcher_init[0](
                    *self.cached_patcher_init[1],
                    disable_dynamic=True,
                )
                model_override = fallback.get_clone_model_override()
            return GGUFModelPatcher.clone(self, model_override=model_override)
        return super().clone(disable_dynamic=disable_dynamic, model_override=model_override)


class UnetLoaderGGUFDynamicVRAM(UnetLoaderGGUF):
    TITLE = "Unet Loader (GGUF, Dynamic VRAM)"

    def load_unet(self, unet_name, **kwargs):
        unet_path = folder_paths.get_full_path("unet", unet_name)
        return (_load_dynamic_gguf_unet(unet_path, progress=GGUFLoadProgress([unet_path])),)

class UnetLoaderGGUFAdvanced(UnetLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
                "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_on_device": ("BOOLEAN", {"default": False}),
            }
        }
    TITLE = "Unet Loader (GGUF/Advanced)"

class CLIPLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.CLIPLoader.INPUT_TYPES()
        return {
            "required": {
                "clip_name": (s.get_filename_list(),),
                "type": base["required"]["type"],
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "bootleg"
    TITLE = "CLIPLoader (GGUF, Legacy node)"

    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list("clip")
        files += folder_paths.get_filename_list("clip_gguf")
        return sorted(files)

    def load_data(self, ckpt_paths):
        clip_data = []
        progress = GGUFLoadProgress(ckpt_paths)
        for p in ckpt_paths:
            if p.endswith(".gguf"):
                sd = gguf_clip_loader(p, progress_callback=progress.callback_for(p))
            else:
                sd = comfy.utils.load_torch_file(p, safe_load=True)
                if "scaled_fp8" in sd: # NOTE: Scaled FP8 would require different custom ops, but only one can be active
                    raise NotImplementedError(f"Mixing scaled FP8 with GGUF is not supported! Use regular CLIP loader or switch model(s)\n({p})")
            clip_data.append(sd)
            progress.complete_file(p)
        return clip_data

    def load_patcher(self, clip_paths, clip_type, clip_data):
        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type = clip_type,
            state_dicts = clip_data,
            model_options = {
                "custom_operations": GGMLOps,
                "initial_device": comfy.model_management.text_encoder_offload_device()
            },
            embedding_directory = folder_paths.get_folder_paths("embeddings"),
        )
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)
        return clip

    def load_clip(self, clip_name, type="stable_diffusion"):
        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher([clip_path], clip_type, self.load_data([clip_path])),)


def _load_dynamic_gguf_clip(clip_paths, clip_type, disable_dynamic=False, progress=None):
    if not disable_dynamic:
        _require_dynamic_vram()
    progress = progress or GGUFLoadProgress(clip_paths)
    clip_data = []
    for path in clip_paths:
        if path.endswith(".gguf"):
            clip_data.append(
                gguf_clip_loader(
                    path,
                    dynamic=not disable_dynamic,
                    progress_callback=progress.callback_for(path),
                )
            )
        else:
            clip_data.append(comfy.utils.load_torch_file(path, safe_load=True))
        progress.complete_file(path)

    model_options = {
        "initial_device": comfy.model_management.text_encoder_offload_device(),
    }
    if disable_dynamic:
        model_options["custom_operations"] = GGMLOps

    clip = comfy.sd.load_text_encoder_state_dicts(
        clip_type=clip_type,
        state_dicts=clip_data,
        model_options=model_options,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        disable_dynamic=disable_dynamic,
    )
    clip.patcher = (
        GGUFModelPatcher.clone(clip.patcher)
        if disable_dynamic
        else _clone_as_dynamic_gguf_patcher(clip.patcher)
    )
    clip.patcher.cached_patcher_init = (_load_dynamic_gguf_clip_patcher, (clip_paths, clip_type))
    return clip


def _load_dynamic_gguf_clip_patcher(clip_paths, clip_type, disable_dynamic=False):
    return _load_dynamic_gguf_clip(
        clip_paths,
        clip_type,
        disable_dynamic=disable_dynamic,
    ).patcher


class CLIPLoaderGGUFDynamicVRAM(CLIPLoaderGGUF):
    TITLE = "CLIPLoader (GGUF, Dynamic VRAM)"

    def load_clip(self, clip_name, type="stable_diffusion"):
        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (_load_dynamic_gguf_clip([clip_path], clip_type, progress=GGUFLoadProgress([clip_path])),)

class DualCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.DualCLIPLoader.INPUT_TYPES()
        file_options = (s.get_filename_list(), )
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "type": base["required"]["type"],
            }
        }

    TITLE = "DualCLIPLoader (GGUF, Legacy node)"

    def load_clip(self, clip_name1, clip_name2, type):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip_paths = (clip_path1, clip_path2)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)


class DualCLIPLoaderGGUFDynamicVRAM(DualCLIPLoaderGGUF):
    TITLE = "DualCLIPLoader (GGUF, Dynamic VRAM)"

    def load_clip(self, clip_name1, clip_name2, type):
        clip_paths = (
            folder_paths.get_full_path("clip", clip_name1),
            folder_paths.get_full_path("clip", clip_name2),
        )
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (_load_dynamic_gguf_clip(clip_paths, clip_type),)

class TripleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "clip_name3": file_options,
            }
        }

    TITLE = "TripleCLIPLoader (GGUF, Legacy node)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, type="sd3"):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip_path3 = folder_paths.get_full_path("clip", clip_name3)
        clip_paths = (clip_path1, clip_path2, clip_path3)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)


class TripleCLIPLoaderGGUFDynamicVRAM(TripleCLIPLoaderGGUF):
    TITLE = "TripleCLIPLoader (GGUF, Dynamic VRAM)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, type="sd3"):
        clip_paths = (
            folder_paths.get_full_path("clip", clip_name1),
            folder_paths.get_full_path("clip", clip_name2),
            folder_paths.get_full_path("clip", clip_name3),
        )
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (_load_dynamic_gguf_clip(clip_paths, clip_type),)

class QuadrupleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        return {
            "required": {
            "clip_name1": file_options,
            "clip_name2": file_options,
            "clip_name3": file_options,
            "clip_name4": file_options,
        }
    }

    TITLE = "QuadrupleCLIPLoader (GGUF, Legacy node)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4, type="stable_diffusion"):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip_path3 = folder_paths.get_full_path("clip", clip_name3)
        clip_path4 = folder_paths.get_full_path("clip", clip_name4)
        clip_paths = (clip_path1, clip_path2, clip_path3, clip_path4)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)


class QuadrupleCLIPLoaderGGUFDynamicVRAM(QuadrupleCLIPLoaderGGUF):
    TITLE = "QuadrupleCLIPLoader (GGUF, Dynamic VRAM)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4, type="stable_diffusion"):
        clip_paths = (
            folder_paths.get_full_path("clip", clip_name1),
            folder_paths.get_full_path("clip", clip_name2),
            folder_paths.get_full_path("clip", clip_name3),
            folder_paths.get_full_path("clip", clip_name4),
        )
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (_load_dynamic_gguf_clip(clip_paths, clip_type),)

NODE_CLASS_MAPPINGS = {
    "TargetedQuantizationGGUF": TargetedQuantizationGGUF,
    "GGUFLoraImport": GGUFLoraImport,
    "UnetLoaderGGUF": UnetLoaderGGUF,
    "VAELoaderGGUF": VAELoaderGGUF,
    "LTXVLatentUpscaleModelLoaderGGUF": LTXVLatentUpscaleModelLoaderGGUF,
    "CLIPLoaderGGUF": CLIPLoaderGGUF,
    "DualCLIPLoaderGGUF": DualCLIPLoaderGGUF,
    "TripleCLIPLoaderGGUF": TripleCLIPLoaderGGUF,
    "QuadrupleCLIPLoaderGGUF": QuadrupleCLIPLoaderGGUF,
    "UnetLoaderGGUFAdvanced": UnetLoaderGGUFAdvanced,
    "UnetLoaderGGUFDynamicVRAM": UnetLoaderGGUFDynamicVRAM,
    "CLIPLoaderGGUFDynamicVRAM": CLIPLoaderGGUFDynamicVRAM,
    "DualCLIPLoaderGGUFDynamicVRAM": DualCLIPLoaderGGUFDynamicVRAM,
    "TripleCLIPLoaderGGUFDynamicVRAM": TripleCLIPLoaderGGUFDynamicVRAM,
    "QuadrupleCLIPLoaderGGUFDynamicVRAM": QuadrupleCLIPLoaderGGUFDynamicVRAM,
}
