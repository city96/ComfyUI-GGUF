# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import warnings
import logging
import torch
import gguf
import json
import re
import os
import threading
import comfy.memory_management
from .ops import GGMLTensor
from .dequant import is_quantized, dequantize_tensor
from .quant_ops import make_quantized

IMG_ARCH_LIST = {"flux", "sd1", "sdxl", "sd3", "aura", "hidream", "cosmos", "ltxv", "ltxv_upscaler", "hyvid", "wan", "lumina2", "qwen_image", "ideogram", "krea2", "minimax_h3", "minimax_h3_vae", "minimax_music3"}
TXT_ARCH_LIST = {"t5", "t5encoder", "llama", "qwen2vl", "qwen3", "qwen3vl", "qwen35", "gemma3", "gemma4", "minimax_music3"}
VIS_TYPE_LIST = {"clip-vision", "mmproj"}
RAW_BYTE_TENSOR_KEYS = frozenset(("tokenizer_json", "spiece_model", "tekken_model"))

def device_supports_bf16():
    """
    Return True if the active torch device can run bf16 natively. On devices
    without native bf16 support, computation silently falls back to fp32 which
    is very slow, so callers should load tensors as fp16 instead.
    """
    try:
        import comfy.model_management
        return comfy.model_management.should_use_bf16(comfy.model_management.get_torch_device())
    except Exception:
        # If support can't be determined, keep the previous bf16 behavior.
        return True


def dynamic_gguf_file_slice(path):
    """
    Create the file handle metadata used by DynamicVRAM to transfer a GGUF
    tensor directly from its mmap-backed file to GPU memory.
    """
    if not comfy.memory_management.aimdo_enabled:
        return None

    import comfy_aimdo.model_mmap

    model_mmap = comfy_aimdo.model_mmap.ModelMMAP(path)
    return model_mmap, threading.Lock()


def attach_dynamic_file_slice(torch_tensor, model_mmap, file_lock, offset, size):
    storage = torch_tensor.untyped_storage()
    storage._comfy_tensor_file_slice = comfy.memory_management.TensorFileSlice(
        model_mmap.get_file_handle(),
        file_lock,
        offset,
        size,
    )
    # Keep the native mmap alive for the storage lifetime.
    storage._comfy_tensor_mmap_refs = (model_mmap,)

def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))

def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        # extra check here as this is used for checking arch string
        if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF {field_name} key: expected string, got {field.types!r}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]].item())
    else:
        raise TypeError(f"Unknown field type {field_type}")

def get_list_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        return tuple(str(field.parts[part_idx], encoding="utf-8") for part_idx in field.data)
    elif field_type in [int, float, bool]:
        return tuple(field_type(field.parts[part_idx][0]) for part_idx in field.data)
    else:
        raise TypeError(f"Unknown field type {field_type}")

def get_gguf_metadata(reader):
    """Extract all simple metadata fields like safetensors"""
    metadata = {}
    for field_name in reader.fields:
        try:
            field = reader.get_field(field_name)
            if len(field.types) == 1:  # Simple scalar fields only
                if field.types[0] == gguf.GGUFValueType.STRING:
                    metadata[field_name] = str(field.parts[field.data[-1]], "utf-8")
                elif field.types[0] == gguf.GGUFValueType.INT32:
                    metadata[field_name] = int(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.F32:
                    metadata[field_name] = float(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.BOOL:
                    metadata[field_name] = bool(field.parts[field.data[-1]])
        except:
            continue
    return metadata

def gguf_tensor_count(path):
    return len(gguf.GGUFReader(path).tensors)


def normalize_raw_byte_tensor(value):
    """Restore tokenizer payloads to the uint8 contract expected by ComfyUI."""
    if isinstance(value, GGMLTensor):
        qtype = getattr(value, "tensor_type", None)
        if qtype == gguf.GGMLQuantizationType.I8:
            return value.data.view(torch.uint8).reshape(value.tensor_shape).contiguous()
        if is_quantized(value):
            value = dequantize_tensor(value, dtype=torch.float32)
        else:
            value = torch.Tensor(value).reshape(value.tensor_shape)
    elif hasattr(value, "dequantize"):
        value = value.dequantize()

    if not torch.is_tensor(value):
        raise TypeError(f"Expected a tensor for tokenizer payload, got {type(value).__name__}")
    return value.to(torch.uint8).contiguous()


def gguf_sd_loader(path, handle_prefix="model.diffusion_model.", is_text_model=False, dynamic=False, progress_callback=None):
    """
    Read state dict as fake tensors
    """
    reader = gguf.GGUFReader(path)
    dynamic_file_slice = dynamic_gguf_file_slice(path) if dynamic else None

    # filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    # detect and verify architecture
    compat = None
    arch_str = get_field(reader, "general.architecture", str)
    type_str = get_field(reader, "general.type", str)
    if arch_str in [None, "pig", "cow"]:
        if is_text_model:
            raise ValueError(f"This gguf file is incompatible with llama.cpp!\nConsider using safetensors or a compatible gguf file\n({path})")
        compat = "sd.cpp" if arch_str is None else arch_str
        # import here to avoid changes to convert.py breaking regular models
        from .tools.convert import detect_arch
        try:
            arch_str = detect_arch(set(val[0] for val in tensors)).arch
        except Exception as e:
            raise ValueError(f"This model is not currently supported - ({e})")
    elif arch_str not in TXT_ARCH_LIST and is_text_model:
        if type_str not in VIS_TYPE_LIST:
            raise ValueError(f"Unexpected text model architecture type in GGUF file: {arch_str!r}")
    elif arch_str not in IMG_ARCH_LIST and not is_text_model:
        raise ValueError(f"Unexpected architecture type in GGUF file: {arch_str!r}")

    if compat:
        logging.warning(f"Warning: This gguf model file is loaded in compatibility mode '{compat}' [arch:{arch_str}]")

    # Q8_CR weights must use ComfyUI's native TensorWiseINT8Layout rather than
    # the generic GGML layout so DynamicVRAM retains native INT8 ConvRot kernels.
    custom_quant_configs = {}
    for field_name in reader.fields:
        if field_name.startswith("comfy.gguf.quant."):
            key = field_name[len("comfy.gguf.quant."):]
            field = reader.get_field(field_name)
            custom_quant_configs[key] = json.loads(str(field.parts[field.data[-1]], "utf-8"))
    custom_quant_tensor_names = {
        tensor_name
        for key, quant_conf in custom_quant_configs.items()
        if quant_conf.get("format") == "int8_tensorwise"
        for tensor_name in (key, f"{key}_scale")
    } | {
        tensor_name
        for key, quant_conf in custom_quant_configs.items()
        if quant_conf.get("format") == "int4_cr"
        for tensor_name in (key, f"{key}_scale")
    }

    # main loading loop
    # Devices without native bf16 fall back to slow fp32 compute, so load the
    # full-precision BF16 storage tensors as fp16 there instead.
    bf16_storage_dtype = torch.bfloat16 if device_supports_bf16() else torch.float16
    state_dict = {}
    qtype_dict = {}
    for tensor_index, (sd_key, tensor) in enumerate(tensors, start=1):
        tensor_name = tensor.name
        # torch_tensor = torch.from_numpy(tensor.data) # mmap

        # NOTE: line above replaced with this block to avoid persistent numpy warning about mmap
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data) # mmap
        if dynamic_file_slice is not None:
            model_mmap, file_lock = dynamic_file_slice
            attach_dynamic_file_slice(
                torch_tensor,
                model_mmap,
                file_lock,
                tensor.data_offset,
                tensor.n_bytes,
            )

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            # Workaround for stable-diffusion.cpp SDXL detection.
            if compat == "sd.cpp" and arch_str == "sdxl":
                if any([tensor_name.endswith(x) for x in (".proj_in.weight", ".proj_out.weight")]):
                    while len(shape) > 2 and shape[-1] == 1:
                        shape = shape[:-1]

        # add to state dict
        raw_byte_tensor = sd_key in RAW_BYTE_TENSOR_KEYS and len(shape) == 1
        if raw_byte_tensor and tensor.tensor_type == gguf.GGMLQuantizationType.I8:
            # GGUF has no U8 type. I8 is used only as a byte-preserving
            # container for tokenizer payloads.
            state_dict[sd_key] = torch_tensor.view(torch.uint8).reshape(shape)
        elif dynamic and sd_key not in custom_quant_tensor_names:
            if tensor.tensor_type in {
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.F16,
            }:
                state_dict[sd_key] = torch_tensor.view(*shape)
            elif tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
                state_dict[sd_key] = torch_tensor.view(torch.bfloat16).reshape(shape).to(
                    dtype=torch.float32 if len(shape) <= 1 else bf16_storage_dtype,
                )
            else:
                state_dict[sd_key] = make_quantized(torch_tensor, tensor.tensor_type, shape)
        elif tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
            state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)
        else:
            state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)

        # BF16 GGUF tensors are full-precision storage, not compressed quants.
        if not dynamic and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            dtype = torch.float32 if len(shape) <= 1 else bf16_storage_dtype
            state_dict[sd_key] = dequantize_tensor(state_dict[sd_key], dtype=dtype)

        # keep track of loaded tensor types
        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1
        if progress_callback is not None:
            progress_callback(tensor_index, len(tensors))

    # print loaded tensor type counts
    logging.info("gguf qtypes: " + ", ".join(f"{k} ({v})" for k, v in qtype_dict.items()))

    # mark largest tensor for vram estimation
    qsd = {k:v for k,v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    # extra info to return
    extra = {
        "arch_str": arch_str,
        "metadata": get_gguf_metadata(reader)
    }

    # Detect custom ComfyUI native quantization metadata
    warned_unrotated_convrot = False
    for field_name in reader.fields:
        if not field_name.startswith("comfy.gguf.quant."):
            continue
        key = field_name[len("comfy.gguf.quant."):]
        field = reader.get_field(field_name)
        quant_conf = custom_quant_configs[key]
        fmt = quant_conf.get("format")

        if fmt in {"int4_compact_gemm", "int4_pytorch"}:
            raise ValueError(
                "Q4_PT GGUF files are retired because PyTorch's Ampere INT4 "
                "kernel is not performance-competitive. Reconvert as Q8_CR."
            )

        weight_key = key
        scale_key = f"{key}_scale"
        if weight_key not in state_dict or scale_key not in state_dict:
            logging.warning(f"Missing custom quant tensors for {weight_key}")
            continue

        weight_ggml = state_dict[weight_key]
        scale_ggml = state_dict[scale_key]

        if fmt == "int8_tensorwise":
            if quant_conf.get("convrot") and not quant_conf.get("weight_rotated", False):
                if not warned_unrotated_convrot:
                    logging.warning(
                        "Disabling ConvRot because this GGUF does not mark its weights "
                        "as pre-rotated. Reconvert with the current converter to enable ConvRot."
                    )
                    warned_unrotated_convrot = True
                quant_conf["convrot"] = False
                quant_conf.pop("convrot_groupsize", None)
            elif quant_conf.get("convrot"):
                groupsize = quant_conf.get("convrot_groupsize", 256)
                weight_shape = weight_ggml.shape if dynamic else weight_ggml.tensor_shape
                if weight_shape[-1] % groupsize != 0:
                    logging.warning(
                        "Disabling ConvRot for %s because %d input features are not "
                        "divisible by group size %d.",
                        weight_key,
                        weight_shape[-1],
                        groupsize,
                    )
                    quant_conf["convrot"] = False
                    quant_conf.pop("convrot_groupsize", None)

            # Convert to ComfyUI native state-dict layout
            if dynamic:
                weight = weight_ggml.view(torch.int8).reshape(weight_ggml.shape)
                scale = scale_ggml.view(torch.float32).reshape(scale_ggml.shape)
            else:
                weight = weight_ggml.data.view(torch.int8).reshape(weight_ggml.tensor_shape)
                scale = scale_ggml.data.view(torch.float32).reshape(scale_ggml.tensor_shape)

            state_dict[weight_key] = torch.nn.Parameter(weight, requires_grad=False)
            state_dict[scale_key] = torch.nn.Parameter(scale, requires_grad=False)

            layer_prefix = weight_key[:weight_key.rfind("weight")]
            quant_json = json.dumps(quant_conf)
            state_dict[f"{layer_prefix}comfy_quant"] = torch.nn.Parameter(
                torch.tensor(list(quant_json.encode("utf-8")), dtype=torch.uint8),
                requires_grad=False,
            )
            extra["gguf_quant_mode"] = "int8_convrot"

        # Q4_CR_W4A4: custom W4A4 INT4 backed by comfy_kitchen's fast ConvRot
        # int4 tensor-core MMA. On-disk is kitchen-native packed int4 (N, K//2)
        # int8 + a per-output-row fp16 scale; no per-group scales/zeros.
        # The weight is pre-rotated by a block-diagonal Hadamard along K, matching
        # the activation rotation the kernel applies at runtime.
        elif fmt == "int4_cr" and quant_conf.get("backing") == "w4a4":
            orig_shape = torch.Size(tuple(quant_conf["orig_shape"]))
            convrot_groupsize = quant_conf.get("convrot_groupsize", 256)
            quant_group_size = quant_conf.get("quant_group_size", 64)
            packed_shape = (orig_shape[0], orig_shape[1] // 2)

            if dynamic:
                packed = weight_ggml.view(torch.int8).reshape(packed_shape)
                scale = scale_ggml.view(torch.float16).to(torch.bfloat16).reshape(orig_shape[0])
            else:
                packed = weight_ggml.data.view(torch.int8).reshape(packed_shape)
                scale = scale_ggml.data.view(torch.float16).to(torch.bfloat16).reshape(orig_shape[0])

            layer_prefix = weight_key[:weight_key.rfind("weight")]

            state_dict[weight_key] = torch.nn.Parameter(packed, requires_grad=False)
            state_dict[scale_key] = torch.nn.Parameter(scale, requires_grad=False)
            state_dict[f"{layer_prefix}comfy_quant"] = torch.nn.Parameter(
                torch.tensor(list(json.dumps(quant_conf).encode("utf-8")), dtype=torch.uint8),
                requires_grad=False,
            )
            extra["gguf_quant_mode"] = "int4_cr_w4a4"

        elif fmt == "int4_cr":
            # The retired W4A16 (AWQ GEMV) Q4_CR format is no longer supported.
            raise ValueError(
                "This Q4_CR GGUF uses the retired W4A16 backing (no 'backing': 'w4a4'). "
                "Reconvert with --quant-type Q4_CR_W4A4."
            )

    return (state_dict, extra)

# for remapping llama.cpp -> original key names
T5_SD_MAP = {
    "enc.": "encoder.",
    ".blk.": ".block.",
    "token_embd": "shared",
    "output_norm": "final_layer_norm",
    "attn_q": "layer.0.SelfAttention.q",
    "attn_k": "layer.0.SelfAttention.k",
    "attn_v": "layer.0.SelfAttention.v",
    "attn_o": "layer.0.SelfAttention.o",
    "attn_norm": "layer.0.layer_norm",
    "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
    "ffn_up": "layer.1.DenseReluDense.wi_1",
    "ffn_down": "layer.1.DenseReluDense.wo",
    "ffn_gate": "layer.1.DenseReluDense.wi_0",
    "ffn_norm": "layer.1.layer_norm",
}

LLAMA_SD_MAP = {
    "blk.": "model.layers.",
    "attn_norm": "input_layernorm",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_v_norm.": "self_attn.v_norm.",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output.weight": "lm_head.weight",
}

GEMMA3_SD_MAP = LLAMA_SD_MAP.copy()
GEMMA3_SD_MAP.update({
    "ffn_norm": "pre_feedforward_layernorm",
    "post_ffw_norm": "post_feedforward_layernorm",
    "post_attention_norm": "post_attention_layernorm",
})

# These specific keys must precede the generic token_embd mapping below.
GEMMA4_SD_MAP = {
    "per_layer_token_embd": "model.embed_tokens_per_layer",
    "per_layer_model_proj": "model.per_layer_model_projection",
    "per_layer_proj_norm": "model.per_layer_projection_norm",
    "inp_gate": "per_layer_input_gate",
    "layer_output_scale.weight": "layer_scalar",
    "post_norm": "post_per_layer_input_norm",
    ".proj.": ".per_layer_projection.",
    **GEMMA3_SD_MAP,
}

# Qwen3.5 (llama.cpp ``qwen35`` arch). ComfyUI's detect_te_model() identifies
# Qwen3.5 by the unprefixed ``model.language_model.*`` layout before applying
# its own prefix rename, so the LM must map with that prefix. Hybrid layers
# carry linear attention (``attn_qkv``/``attn_gate``/``ssm_*`` tensors); the
# second layernorm is exported as ``post_attention_norm`` (no ``ffn_norm``).
# Entry order matters because sd_map_replace() does substring replacement:
# the fused/SSM variants must precede the generic ``attn_q``/``ssm_a`` keys.
QWEN35_SD_MAP = {
    "blk.": "model.language_model.layers.",
    "attn_norm": "input_layernorm",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_qkv": "linear_attn.in_proj_qkv",
    "attn_gate": "linear_attn.in_proj_z",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ssm_alpha": "linear_attn.in_proj_a",
    "ssm_beta": "linear_attn.in_proj_b",
    "ssm_conv1d": "linear_attn.conv1d",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_norm": "linear_attn.norm",
    "ssm_out": "linear_attn.out_proj",
    "ssm_a": "linear_attn.A_log",
    "post_attention_norm": "post_attention_layernorm",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "token_embd": "model.language_model.embed_tokens",
    "output_norm": "model.language_model.norm",
    "output.weight": "lm_head.weight",
}

CLIP_VISION_SD_MAP = {
    "mm.": "visual.merger.mlp.",
    "v.post_ln.": "visual.merger.ln_q.",
    "v.patch_embd": "visual.patch_embed.proj",
    "v.blk.": "visual.blocks.",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "attn_out.": "attn.proj.",
    "ln1.": "norm1.",
    "ln2.": "norm2.",
}

CLIP_VISION_QWEN3_MAP = {
    "v.blk": "model.visual.blocks",
    ".fc": ".linear_fc",
    "ck.8.": "st.0.",
    "ck.16.": "st.1.",
    "ck.24.": "st.2.",
    "ck.5.": "st.0.",
    "ck.11.": "st.1.",
    "ck.17.": "st.2.",
    "attn_out": "attn.proj",
    "ln1": "norm1",
    "ln2": "norm2",
    "attn_qkv": "attn.qkv",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
    "mm.0": "model.visual.merger.linear_fc1",
    "mm.2": "model.visual.merger.linear_fc2",
    "v.post_ln": "model.visual.merger.norm",
    "v.patch_embd": "model.visual.patch_embed.proj",
    "v.position_embd.weight": "visual.pos_embed.weight",
    "v.deepstack.": "model.visual.deepstack_merger_list.",
    # Older llama.cpp Qwen3-VL exporters misspelled this tensor prefix.
    "v.deepstast.": "model.visual.deepstack_merger_list.",
}

def sd_map_replace(raw_sd, key_map):
    sd = {}
    for k,v in raw_sd.items():
        for s,d in key_map.items():
            k = k.replace(s,d)
        sd[k] = v
    return sd

def llama_permute(raw_sd, n_head, n_head_kv):
    # Reverse version of LlamaModel.permute in llama.cpp convert script
    sd = {}
    permute = lambda x,h: x.reshape(h, x.shape[0] // h // 2, 2, *x.shape[1:]).swapaxes(1, 2).reshape(x.shape)
    for k,v in raw_sd.items():
        if k.endswith(("q_proj.weight", "q_proj.bias")):
            v.data = permute(v.data, n_head)
        if k.endswith(("k_proj.weight", "k_proj.bias")):
            v.data = permute(v.data, n_head_kv)
        sd[k] = v
    return sd

def gemma3_norm_corrections(sd):
    # Reverse change from Gemma3Model modify_tensors in llama.cpp convert script
    norm_patterns = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "model.norm.weight"
    ]
    corrected = 0
    for key in list(sd.keys()):
        if any(p in key for p in norm_patterns):
            if is_quantized(sd[key]):
                sd[key] = dequantize_tensor(sd[key], dtype=torch.float32) - 1.0
            else:
                sd[key] = sd[key].float() - 1.0
            corrected += 1
    #logging.info(f"Gemma3: Applied -1 norm correction to {corrected} tensors")
    return sd

def _qwen35_v_reorder(value, num_v_heads, num_k_heads, head_dim, qk_rows=0, axis=0):
    """Reverse llama.cpp's tiled V-head order for linear attention.

    llama.cpp (``_LinearAttentionVReorderBase``) stores V heads in tiled
    order ``[K0_v0, K1_v0, ..., K0_v1, K1_v1, ...]`` so its kernels can
    broadcast K with ``ggml_repeat``. ComfyUI expects the grouped-by-K-head
    order ``[K0_v0..v{r-1}, K1_v0..v{r-1}, ...]`` that the HF checkpoints
    use. Quantized tensors are dequantized first because the head blocks
    do not align with every GGML quant block size. ``qk_rows`` skips the
    untouched Q/K rows of fused tensors; ``axis=1`` reorders columns.
    """
    if num_k_heads <= 0 or num_v_heads <= num_k_heads:
        return value
    if is_quantized(value):
        dtype = torch.bfloat16 if device_supports_bf16() else torch.float16
        value = dequantize_tensor(value, dtype=dtype)
    r = num_v_heads // num_k_heads
    # grouped head g = (k_idx, v_idx); its tiled position is v_idx * k + k_idx
    src = [(g % r) * num_k_heads + (g // r) for g in range(num_v_heads)]
    idx = torch.tensor(src, dtype=torch.long)
    if value.ndim == 1:
        return value.index_select(0, idx)
    if qk_rows > 0:
        # fused qkv / conv1d: only the V block is reordered
        head_rows = value.shape[0] - qk_rows
        v = value[qk_rows:].view(num_v_heads, -1).index_select(0, idx)
        v = v.reshape(head_rows, value.shape[1])
        return torch.cat([value[:qk_rows], v], dim=0)
    if axis == 1:
        return (
            value.view(-1, num_v_heads, head_dim)
            .index_select(1, idx)
            .reshape(value.shape)
        )
    return value.view(num_v_heads, -1).index_select(0, idx).reshape(value.shape)

def qwen35_corrections(sd):
    # Reverse llama.cpp's Qwen3.5 conversion tweaks (see Qwen3NextModel and
    # _LinearAttentionVReorderBase modify_tensors in llama.cpp): it stores
    # RMS norm weights as (w + 1), stores ``ssm_a`` as ``-exp(A_log)``,
    # squeezes the depthwise conv1d kernels to 2D, and stores linear-attention
    # V heads in tiled order. ComfyUI's Qwen35 TE expects the raw weights and
    # applies ``w + 1`` and ``-exp(A_log)`` itself.
    norm_patterns = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "model.language_model.norm.weight",
    ]
    corrected = 0
    for key in list(sd.keys()):
        if any(p in key for p in norm_patterns):
            if is_quantized(sd[key]):
                sd[key] = dequantize_tensor(sd[key], dtype=torch.float32) - 1.0
            else:
                sd[key] = sd[key].float() - 1.0
            corrected += 1

    # derive the linear-attention head layout from the checkpoint shapes
    qkv_key = next((k for k in sd if k.endswith(".linear_attn.in_proj_qkv.weight")), None)
    z_key = next((k for k in sd if k.endswith(".linear_attn.in_proj_z.weight")), None)
    alog_key = next((k for k in sd if k.endswith(".linear_attn.A_log")), None)
    num_k_heads = num_v_heads = head_dim = 0
    if qkv_key and z_key and alog_key:
        value_dim = sd[z_key].shape[0]
        conv_dim = sd[qkv_key].shape[0]
        key_dim = (conv_dim - value_dim) // 2
        num_v_heads = sd[alog_key].shape[0]
        head_dim = value_dim // num_v_heads
        num_k_heads = key_dim // head_dim

    for key in list(sd.keys()):
        if key.endswith(".linear_attn.A_log"):
            # llama.cpp stores ``ssm_a = -exp(A_log)`` (always negative);
            # ComfyUI computes ``-A_log.exp()``, so invert back to A_log.
            value = sd[key]
            if is_quantized(value):
                value = dequantize_tensor(value, dtype=torch.float32)
            else:
                value = value.float()
            sd[key] = torch.log(-value)
            corrected += 1
        elif key.endswith(".linear_attn.dt_bias"):
            corrected += 1
        elif key.endswith(".linear_attn.conv1d.weight"):
            corrected += 1
        elif key.endswith(".linear_attn.in_proj_qkv.weight"):
            corrected += 1
        elif key.endswith((
            ".linear_attn.in_proj_z.weight",
            ".linear_attn.in_proj_a.weight",
            ".linear_attn.in_proj_b.weight",
        )):
            corrected += 1
        elif key.endswith(".linear_attn.out_proj.weight"):
            corrected += 1
        else:
            continue

        # reverse the tiled V-head order (identity when heads are balanced)
        if key.endswith((".linear_attn.A_log", ".linear_attn.dt_bias")):
            sd[key] = _qwen35_v_reorder(sd[key], num_v_heads, num_k_heads, 1)
        elif key.endswith(".linear_attn.conv1d.weight"):
            value = sd[key]
            if value.ndim == 2:
                conv_dim = value.shape[0]
                qk_channels = conv_dim - num_v_heads * head_dim
                # llama.cpp squeezes the depthwise conv1d kernel to 2D
                # (out_channels, kernel_size); ComfyUI's Conv1d expects
                # (out_channels, 1, kernel_size).
                value = _qwen35_v_reorder(
                    value, num_v_heads, num_k_heads, head_dim, qk_rows=qk_channels
                )
                sd[key] = value.unsqueeze(-2)
        elif key.endswith(".linear_attn.in_proj_qkv.weight"):
            value = sd[key]
            qk_rows = value.shape[0] - num_v_heads * head_dim
            sd[key] = _qwen35_v_reorder(
                value, num_v_heads, num_k_heads, head_dim, qk_rows=qk_rows
            )
        elif key.endswith(".linear_attn.out_proj.weight"):
            sd[key] = _qwen35_v_reorder(
                sd[key], num_v_heads, num_k_heads, head_dim, axis=1
            )
        else:
            sd[key] = _qwen35_v_reorder(sd[key], num_v_heads, num_k_heads, 1)

    logging.info(f"qwen35 GGUF: corrected {corrected} norm/A_log/conv1d/V-head tensors")
    return sd

def strip_quant_suffix(name):
    pattern = r"[-_]?(?:ud-)?i?q[0-9]_[a-z0-9_\-]{1,8}$"
    match = re.search(pattern, name, re.IGNORECASE)
    if match:
        name = name[:match.start()]
    return name

def gguf_mmproj_loader(path):
    # Reverse version of Qwen2VLVisionModel.modify_tensors
    logging.info("Attenpting to find mmproj file for text encoder...")

    # get name to match w/o quant suffix
    tenc_fname = os.path.basename(path)
    tenc = os.path.splitext(tenc_fname)[0].lower()
    tenc = strip_quant_suffix(tenc)

    # try and find matching mmproj
    target = []
    root = os.path.dirname(path)
    for fname in os.listdir(root):
        name, ext = os.path.splitext(fname)
        if ext.lower() != ".gguf":
            continue
        if "mmproj" not in name.lower():
            continue
        if tenc in name.lower():
            target.append(fname)

    if len(target) == 0:
        logging.error(f"Error: Can't find mmproj file for '{tenc_fname}' (matching:'{tenc}')! Qwen-Image-Edit will be broken!")
        return {}
    if len(target) > 1:
        logging.error(f"Ambiguous mmproj for text encoder '{tenc_fname}', will use first match.")

    logging.info(f"Using mmproj '{target[0]}' for text encoder '{tenc_fname}'.")
    target = os.path.join(root, target[0])
    vsd, _ = gguf_sd_loader(target, is_text_model=True)

    # concat 4D to 5D
    if "v.patch_embd.weight.1" in vsd:
        w1 = dequantize_tensor(vsd.pop("v.patch_embd.weight"), dtype=torch.float32)
        w2 = dequantize_tensor(vsd.pop("v.patch_embd.weight.1"), dtype=torch.float32)
        vsd["v.patch_embd.weight"] = torch.stack([w1, w2], dim=2)

    if any("deepstack" in key or "deepstast" in key or "attn_qkv" in key for key in vsd):
        # Qwen3-VL / Qwen3.5 style vision towers use fused ``v.blk.N.attn_qkv``
        # projections and a ``linear_fc`` merger. Qwen3.5 mmprojs omit the
        # deepstack tensors, so detect them by the fused projection instead.
        return sd_map_replace(vsd, CLIP_VISION_QWEN3_MAP)

    # run main replacement
    vsd = sd_map_replace(vsd, CLIP_VISION_SD_MAP)

    # handle split Q/K/V
    if "visual.blocks.0.attn_q.weight" in vsd:
        attns = {}
        # filter out attentions + group
        for k,v in vsd.items():
            if any(x in k for x in ["attn_q", "attn_k", "attn_v"]):
                k_attn, k_name = k.rsplit(".attn_", 1)
                k_attn += ".attn.qkv." + k_name.split(".")[-1]
                if k_attn not in attns:
                    attns[k_attn] = {}
                attns[k_attn][k_name] = dequantize_tensor(
                    v, dtype=(torch.bfloat16 if is_quantized(v) else torch.float16)
                )

        # recombine
        for k,v in attns.items():
            suffix = k.split(".")[-1]
            vsd[k] = torch.cat([
                v[f"q.{suffix}"],
                v[f"k.{suffix}"],
                v[f"v.{suffix}"],
            ], dim=0)
        del attns

    return vsd

def gguf_tokenizer_loader(path, temb_shape):
    # convert gguf tokenizer to spiece
    logging.info("Attempting to recreate sentencepiece tokenizer from GGUF file metadata...")
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError("Please make sure sentencepiece and protobuf are installed.\npip install sentencepiece protobuf")
    spm = model.ModelProto()

    reader = gguf.GGUFReader(path)

    if get_field(reader, "tokenizer.ggml.model", str) == "t5":
        if temb_shape == (256384, 4096): # probably UMT5
            spm.trainer_spec.model_type == 1 # Unigram (do we have a T5 w/ BPE?)
        else:
            raise NotImplementedError("Unknown model, can't set tokenizer!")
    else:
        raise NotImplementedError("Unknown model, can't set tokenizer!")

    spm.normalizer_spec.add_dummy_prefix = get_field(reader, "tokenizer.ggml.add_space_prefix", bool)
    spm.normalizer_spec.remove_extra_whitespaces = get_field(reader, "tokenizer.ggml.remove_extra_whitespaces", bool)

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    scores = get_list_field(reader, "tokenizer.ggml.scores", float)
    toktypes = get_list_field(reader, "tokenizer.ggml.token_type", int)

    for idx, (token, score, toktype) in enumerate(zip(tokens, scores, toktypes)):
        # # These aren't present in the original?
        # if toktype == 5 and idx >= temb_shape[0]%1000):
        #     continue

        piece = spm.SentencePiece()
        piece.piece = token
        piece.score = score
        piece.type = toktype
        spm.pieces.append(piece)

    # unsure if any of these are correct
    spm.trainer_spec.byte_fallback = True
    spm.trainer_spec.vocab_size = len(tokens) # split off unused?
    spm.trainer_spec.max_sentence_length = 4096
    spm.trainer_spec.eos_id = get_field(reader, "tokenizer.ggml.eos_token_id", int)
    spm.trainer_spec.pad_id = get_field(reader, "tokenizer.ggml.padding_token_id", int)

    logging.info(f"Created tokenizer with vocab size of {len(spm.pieces)}")
    del reader
    return torch.ByteTensor(list(spm.SerializeToString()))

def gguf_tekken_tokenizer_loader(path, temb_shape):
    # convert ggml (hf) tokenizer metadata to tekken/comfy data
    logging.info("Attempting to recreate tekken tokenizer from GGUF file metadata...")
    import json
    import base64
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    reader = gguf.GGUFReader(path)

    model_str = get_field(reader, "tokenizer.ggml.model", str)
    if model_str == "gpt2":
        if temb_shape == (131072, 5120): # probably Mistral
            data = {
                "config": {"num_vocab_tokens": 150000, "default_vocab_size": 131072},
                "vocab": [],
                "special_tokens": [],
            }
        else:
            raise NotImplementedError("Unknown model, can't set tokenizer!")
    else:
        raise NotImplementedError("Unknown model, can't set tokenizer!")

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    toktypes = get_list_field(reader, "tokenizer.ggml.token_type", int)

    decoder = {v: k for k, v in bytes_to_unicode().items()}
    for idx, (token, toktype) in enumerate(zip(tokens, toktypes)):
        if toktype == 3:
            data["special_tokens"].append(
                {'rank': idx, 'token_str': token, 'is_control': True}
            )
        else:
            tok = bytes([decoder[char] for char in token])
            data["vocab"].append({
                "rank": len(data["vocab"]),
                "token_bytes": base64.b64encode(tok).decode("ascii"),
                "token_str": tok.decode("utf-8", errors="replace") # ?
            })

    logging.info(f"Created tekken tokenizer with vocab size of {len(data['vocab'])} (+{len(data['special_tokens'])})")
    del reader
    return torch.ByteTensor(list(json.dumps(data).encode('utf-8')))

def gguf_gemma3_tokenizer_loader(path):
    #TODO: merge into gguf_tokenizer_loader
    logging.info("Attempting to recreate sentencepiece tokenizer from GGUF file metadata...")
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError("Please install sentencepiece and protobuf.\npip install sentencepiece protobuf")
    spm = model.ModelProto()
    reader = gguf.GGUFReader(path)

    spm.normalizer_spec.name = "identity"
    spm.normalizer_spec.add_dummy_prefix = False
    spm.trainer_spec.model_type = 2
    spm.trainer_spec.input_format = "tsv"
    spm.trainer_spec.byte_fallback = True
    spm.trainer_spec.max_sentence_length = 4192
    spm.trainer_spec.bos_piece = "<bos>"

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    scores = get_list_field(reader, "tokenizer.ggml.scores", float)
    toktype = get_list_field(reader, "tokenizer.ggml.token_type", int)

    if not tokens or not scores or not toktype:
        raise ValueError("Missing tokenizer metadata")

    for idx in range(len(tokens)):
        piece = spm.SentencePiece()
        piece.piece = tokens[idx]
        if idx == 3:  # UNK position
            piece.type = 2  # UNK Token
            piece.score = 0.0 # UNK Score
        else:
            piece.type = toktype[idx]
            piece.score = scores[idx]
        spm.pieces.append(piece)

    spm.trainer_spec.vocab_size = len(spm.pieces)
    logging.info(f"Created tokenizer with vocab size of {len(spm.pieces)}")

    del reader
    return torch.ByteTensor(list(spm.SerializeToString()))

def gemma4_tokenizer_json(tokens, merges, token_types):
    if not tokens or not merges or not token_types:
        raise ValueError("Missing Gemma 4 tokenizer metadata")
    if len(tokens) != len(token_types):
        raise ValueError("Gemma 4 tokenizer token and token-type counts differ")

    # Gemma 4 stores its BPE vocabulary in standard GGUF token and merge fields.
    data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": index,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for index, (token, token_type) in enumerate(zip(tokens, token_types))
            if token_type == 3
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "\u2581",
            "prepend_scheme": "never",
            "split": True,
        },
        "post_processor": None,
        "decoder": {
            "type": "Metaspace",
            "replacement": "\u2581",
            "prepend_scheme": "never",
            "split": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {token: index for index, token in enumerate(tokens)},
            "merges": list(merges),
        },
    }
    return torch.ByteTensor(list(json.dumps(data, ensure_ascii=False).encode("utf-8")))

def gguf_gemma4_tokenizer_loader(path):
    logging.info("Recreating Gemma 4 BPE tokenizer from GGUF metadata...")
    reader = gguf.GGUFReader(path)
    if get_field(reader, "tokenizer.ggml.model", str) != "gemma4":
        raise ValueError("Expected a Gemma 4 tokenizer in the GGUF metadata")

    tokenizer_json = gemma4_tokenizer_json(
        get_list_field(reader, "tokenizer.ggml.tokens", str),
        get_list_field(reader, "tokenizer.ggml.merges", str),
        get_list_field(reader, "tokenizer.ggml.token_type", int),
    )
    del reader
    return tokenizer_json

def inject_qwen3vl_detection_markers(sd):
    """Add visual sentinels when a llama.cpp Qwen3-VL GGUF excludes its vision tower."""
    ln_key = "model.layers.0.input_layernorm.weight"
    lm_hidden = int(sd[ln_key].shape[0]) if ln_key in sd else 2560
    vis_hidden = 1024 if lm_hidden == 2560 else 1152
    merge_dim = vis_hidden * 4  # spatial_merge_size=2

    if lm_hidden == 5120:
        # MiniMax H3 uses the truncated Qwen3-VL-32B encoder. Its detector
        # deliberately checks this unprefixed visual key plus layer 49.
        marker_key = "visual.deepstack_merger_list.0.norm.weight"
    else:
        marker_key = "model.visual.deepstack_merger_list.0.norm.weight"

    sd[marker_key] = torch.zeros(merge_dim)
    if lm_hidden != 5120:
        sd["model.visual.merger.linear_fc2.weight"] = torch.zeros(lm_hidden, merge_dim)
    logging.info(
        "qwen3vl GGUF: injected visual marker tensor "
        "(lm_hidden=%d, merge_dim=%d)",
        lm_hidden,
        merge_dim,
    )

def gguf_clip_loader(path, dynamic=False, progress_callback=None):
    sd, extra = gguf_sd_loader(
        path,
        is_text_model=True,
        dynamic=dynamic,
        progress_callback=progress_callback,
    )
    arch = extra.get("arch_str", None)
    if arch == "minimax_music3" and "tokenizer_json" in sd:
        sd["tokenizer_json"] = normalize_raw_byte_tensor(sd["tokenizer_json"])
    if arch in {"t5", "t5encoder"}:
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape == (256384, 4096):
            # non-standard Comfy-Org tokenizer
            sd["spiece_model"] = gguf_tokenizer_loader(path, sd[temb_key].shape)
            # TODO: dequantizing token embed here is janky but otherwise we OOM due to tensor being massive.
            logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        sd = sd_map_replace(sd, T5_SD_MAP)
    elif arch in {"llama", "qwen2vl", "qwen3", "qwen3vl", "qwen35", "gemma3", "gemma4"}:
        # TODO: pass model_options["vocab_size"] to loader somehow
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape[0] >= (64 * 1024):
            if arch == "llama" and sd[temb_key].shape == (131072, 5120):
                # non-standard Comfy-Org tokenizer
                sd["tekken_model"] = gguf_tekken_tokenizer_loader(path, sd[temb_key].shape)
            elif arch == "gemma3":
                sd["spiece_model"] = gguf_gemma3_tokenizer_loader(path)
            elif arch == "gemma4":
                sd["tokenizer_json"] = gguf_gemma4_tokenizer_loader(path)
            # See note above for T5.
            logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        if arch == "gemma3":
            sd = sd_map_replace(sd, GEMMA3_SD_MAP)
            sd = gemma3_norm_corrections(sd)
        elif arch == "gemma4":
            # ComfyUI calculates Gemma 4 RoPE frequencies itself.
            sd.pop("rope_freqs.weight", None)
            sd = sd_map_replace(sd, GEMMA4_SD_MAP)
        elif arch == "qwen35":
            sd = sd_map_replace(sd, QWEN35_SD_MAP)
            sd = qwen35_corrections(sd)
            vsd = gguf_mmproj_loader(path)
            sd.update(vsd)
        else:
            sd = sd_map_replace(sd, LLAMA_SD_MAP)
        if arch == "llama":
            sd = llama_permute(sd, 32, 8) # L3 / Mistral
        if arch == "qwen2vl":
            vsd = gguf_mmproj_loader(path)
            sd.update(vsd)
        if arch == "qwen3vl":
            vsd = gguf_mmproj_loader(path)
            if vsd and "model.layers.49.self_attn.q_proj.weight" in sd:
                # MiniMax H3 receives the Qwen3-VL vision tower under
                # ``visual.*``, unlike the standalone Qwen3-VL variants.
                vsd = {
                    key.replace("model.visual.", "visual.", 1)
                    if key.startswith("model.visual.")
                    else key: value
                    for key, value in vsd.items()
                }
            sd.update(vsd)
        if arch == "qwen3vl" and not (
            "model.visual.deepstack_merger_list.0.norm.weight" in sd
            or "visual.deepstack_merger_list.0.norm.weight" in sd
        ):
            # Standard llama.cpp Qwen3-VL GGUFs omit the visual tower. Without it,
            # detect_te_model() mis-classifies the state dict as a Qwen3 LM instead
            # of Qwen3-VL. MiniMax H3 additionally uses Qwen3-VL-32B truncated to
            # 50 layers, whose detector relies on an unprefixed visual marker.
            # Inject zero sentinel tensors with shapes that exactly match the model
            # parameters so that load_state_dict(strict=False) doesn't raise a size
            # mismatch error while still satisfying detect_te_model()'s key checks.
            inject_qwen3vl_detection_markers(sd)
        if "lm_head.weight" in sd and is_quantized(sd["lm_head.weight"]):
            lm_head = sd["lm_head.weight"]
            if getattr(lm_head, "tensor_type", None) != gguf.GGMLQuantizationType.BF16:
                # BaseGenerate.logits() feeds lm_head straight to F.linear,
                # bypassing the GGML ops path that dequantizes on the fly.
                # Block-quantized GGUF data has a byte-expanded shape (e.g.
                # Q8_0 stores scales interleaved), so raw data must never
                # reach the matmul. BF16 storage is already full precision
                # and keeps its logical shape, so it can stay quantized.
                logging.warning(f"Dequantizing lm_head.weight to prevent raw-block matmul.")
                sd["lm_head.weight"] = dequantize_tensor(lm_head, dtype=torch.float16)
    elif arch == "ideogram":
        # Dequantize Ideogram model for inference
        logging.info("Dequantizing Ideogram model for inference...")
        # Use BF16 to save VRAM while maintaining quality, but fall back to FP16
        # on devices that don't support bf16 (avoids slow fp32 compute fallback).
        target_dtype = torch.bfloat16 if device_supports_bf16() else torch.float16
        dequantized_count = 0
        for key in list(sd.keys()):
            if is_quantized(sd[key]):
                sd[key] = dequantize_tensor(sd[key], dtype=target_dtype)
                dequantized_count += 1
        logging.info(f"Dequantized {dequantized_count} tensors for Ideogram model ({target_dtype})")
    else:
        pass
    return sd
