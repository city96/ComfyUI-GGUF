# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import torch
import logging
import inspect
import collections

import nodes
import comfy.sd
import comfy.lora
import comfy.float
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import comfy.memory_management
import folder_paths

from .ops import GGMLOps, move_patch_to_device
from .loader import gguf_sd_loader, gguf_clip_loader
from .dequant import is_quantized, is_torch_compatible

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

def _clone_as_gguf_model_patcher(self, *args, model_override=None, **kwargs):
    if model_override is None:
        model_override = self.get_clone_model_override()
    mmap_released = model_override[2] if len(model_override) > 2 else False
    src_cls = self.__class__
    self.__class__ = GGUFModelPatcher
    n = comfy.model_patcher.ModelPatcher.clone(self, *args, model_override=model_override, **kwargs)
    n.__class__ = GGUFModelPatcher
    self.__class__ = src_cls
    n.patch_on_device = getattr(self, "patch_on_device", False)
    n.mmap_released = mmap_released
    return n

class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
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

    def get_clone_model_override(self):
        return (*super().get_clone_model_override(), self.mmap_released)

    def load(self, *args, force_patch_weights=False, **kwargs):
        if not self.mmap_released:
            self.named_modules_to_munmap = dict(self.model.named_modules())

        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

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
        n = _clone_as_gguf_model_patcher(self, *args, **kwargs)
        if self.__class__ != GGUFModelPatcher:
            n.size = 0 # force recalc
        return n

class GGUFModelPatcherDynamic(comfy.model_patcher.ModelPatcherDynamic):
    patch_on_device = False

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)
        # GGML can't requantize after LoRA - demote lowvram_function to weight_function
        for n, m in self.model.named_modules():
            for param_key in ("weight", "bias"):
                attr = param_key + "_lowvram_function"
                fn = getattr(m, attr, None)
                if fn is not None:
                    setattr(m, attr, None)
                    fns = getattr(m, param_key + "_function", [])
                    fns.append(fn)
                    setattr(m, param_key + "_function", fns)
        if self.patch_on_device:
            for key in self.patches:
                self.patches[key] = move_patch_to_device(self.patches[key], self.load_device)

    def clone(self, disable_dynamic=False, model_override=None):
        if disable_dynamic:
            if model_override is None:
                temp = self.cached_patcher_init[0](*self.cached_patcher_init[1], disable_dynamic=True)
                model_override = temp.get_clone_model_override()
            n = _clone_as_gguf_model_patcher(self, model_override=model_override)
            return n
        n = super().clone(disable_dynamic=disable_dynamic, model_override=model_override)
        n.patch_on_device = self.patch_on_device
        return n

def _clone_patcher_to_gguf(model_patcher):
    if model_patcher.is_dynamic():
        src_cls = model_patcher.__class__
        model_patcher.__class__ = GGUFModelPatcherDynamic
        n = model_patcher.clone()
        model_patcher.__class__ = src_cls
        return n
    else:
        return GGUFModelPatcher.clone(model_patcher)

def _load_gguf_unet(unet_path, ops, disable_dynamic=False):
    dynamic = not disable_dynamic and comfy.memory_management.aimdo_enabled
    sd, extra = gguf_sd_loader(unet_path, dynamic=dynamic)

    kwargs = {}
    valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
    if "metadata" in valid_params:
        kwargs["metadata"] = extra.get("metadata", {})

    model = comfy.sd.load_diffusion_model_state_dict(
        sd, model_options={} if dynamic else { "custom_operations" : ops }, disable_dynamic=disable_dynamic, **kwargs,
    )
    if model is None:
        logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
    model = _clone_patcher_to_gguf(model)

    model.cached_patcher_init = (_load_gguf_unet, (unet_path, ops))

    return model

def _load_gguf_clip_patcher(clip_paths, clip_type, disable_dynamic=False):
    return _load_gguf_clip(clip_paths, clip_type, disable_dynamic=disable_dynamic).patcher

def _load_gguf_clip(clip_paths, clip_type, disable_dynamic=False):
    dynamic = not disable_dynamic and comfy.memory_management.aimdo_enabled

    clip_data = []
    for p in clip_paths:
        if p.endswith(".gguf"):
            sd = gguf_clip_loader(p, dynamic=dynamic)
        else:
            sd = comfy.utils.load_torch_file(p, safe_load=True)
            if not dynamic and "scaled_fp8" in sd: # NOTE: Scaled FP8 would require different custom ops, but only one can be active
                raise NotImplementedError(f"Mixing scaled FP8 with GGUF is not supported! Use regular CLIP loader or switch model(s)\n({p})")
        clip_data.append(sd)

    model_options = {"initial_device": comfy.model_management.text_encoder_offload_device()}
    if not dynamic:
        model_options["custom_operations"] = GGMLOps

    clip = comfy.sd.load_text_encoder_state_dicts(
        clip_type = clip_type,
        state_dicts = clip_data,
        model_options = model_options,
        embedding_directory = folder_paths.get_folder_paths("embeddings"),
        disable_dynamic = disable_dynamic,
    )
    clip.patcher = _clone_patcher_to_gguf(clip.patcher)

    clip.patcher.cached_patcher_init = (_load_gguf_clip_patcher, (clip_paths, clip_type))
    return clip

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
    TITLE = "Unet Loader (GGUF)"

    def load_unet(self, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None):
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

        unet_path = folder_paths.get_full_path("unet", unet_name)
        model = _load_gguf_unet(unet_path, ops)
        model.patch_on_device = patch_on_device
        return (model,)

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
    TITLE = "CLIPLoader (GGUF)"

    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list("clip")
        files += folder_paths.get_filename_list("clip_gguf")
        return sorted(files)

    def load_clip(self, clip_name, type="stable_diffusion"):
        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (_load_gguf_clip([clip_path], clip_type),)

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

    TITLE = "DualCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, type):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip_paths = (clip_path1, clip_path2)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (_load_gguf_clip(clip_paths, clip_type),)

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

    TITLE = "TripleCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, type="sd3"):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip_path3 = folder_paths.get_full_path("clip", clip_name3)
        clip_paths = (clip_path1, clip_path2, clip_path3)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (_load_gguf_clip(clip_paths, clip_type),)

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

    TITLE = "QuadrupleCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4, type="stable_diffusion"):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip_path3 = folder_paths.get_full_path("clip", clip_name3)
        clip_path4 = folder_paths.get_full_path("clip", clip_name4)
        clip_paths = (clip_path1, clip_path2, clip_path3, clip_path4)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (_load_gguf_clip(clip_paths, clip_type),)

NODE_CLASS_MAPPINGS = {
    "UnetLoaderGGUF": UnetLoaderGGUF,
    "CLIPLoaderGGUF": CLIPLoaderGGUF,
    "DualCLIPLoaderGGUF": DualCLIPLoaderGGUF,
    "TripleCLIPLoaderGGUF": TripleCLIPLoaderGGUF,
    "QuadrupleCLIPLoaderGGUF": QuadrupleCLIPLoaderGGUF,
    "UnetLoaderGGUFAdvanced": UnetLoaderGGUFAdvanced,
}
