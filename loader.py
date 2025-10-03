# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import warnings
import logging
import torch
import gguf
import re
import os

from .ops import GGMLTensor
from .dequant import is_quantized, dequantize_tensor

IMG_ARCH_LIST = {"flux", "sd1", "sdxl", "sd3", "aura", "hidream", "cosmos", "ltxv", "hyvid", "wan", "lumina2", "qwen_image"}
TXT_ARCH_LIST = {"t5", "t5encoder", "llama", "qwen2vl", "gemma2"}
VIS_TYPE_LIST = {"clip-vision"}

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
        return field_type(field.parts[field.data[-1]])
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

def gguf_sd_loader(path, handle_prefix="model.diffusion_model.", return_arch=False, is_text_model=False):
    """
    Read state dict as fake tensors
    """
    reader = gguf.GGUFReader(path)

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
    if arch_str in [None, "pig"]:
        if is_text_model:
            raise ValueError(f"This text model is incompatible with llama.cpp!\nConsider using the safetensors version\n({path})")
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

    # main loading loop
    state_dict = {}
    qtype_dict = {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        # torch_tensor = torch.from_numpy(tensor.data) # mmap

        # NOTE: line above replaced with this block to avoid persistent numpy warning about mmap
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data) # mmap

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            # Workaround for stable-diffusion.cpp SDXL detection.
            if compat == "sd.cpp" and arch_str == "sdxl":
                if any([tensor_name.endswith(x) for x in (".proj_in.weight", ".proj_out.weight")]):
                    while len(shape) > 2 and shape[-1] == 1:
                        shape = shape[:-1]

        # add to state dict
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
        state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)

        # keep track of loaded tensor types
        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    # print loaded tensor type counts
    logging.info("gguf qtypes: " + ", ".join(f"{k} ({v})" for k, v in qtype_dict.items()))

    # mark largest tensor for vram estimation
    qsd = {k:v for k,v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    if return_arch:
        return (state_dict, arch_str)
    return state_dict

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

GEMMA2_SD_MAP = {
    "blk.": "model.layers.",
    # Attention
    ".attn_q.weight": ".self_attn.q_proj.weight",
    ".attn_k.weight": ".self_attn.k_proj.weight",
    ".attn_v.weight": ".self_attn.v_proj.weight",
    ".attn_output.weight": ".self_attn.o_proj.weight",
    # LayerNorm
    ".attn_norm.weight": ".input_layernorm.weight",
    ".post_attention_norm.weight": ".post_attention_layernorm.weight",
    ".post_ffw_norm.weight": ".post_feedforward_layernorm.weight",
    ".ffn_norm.weight": ".pre_feedforward_layernorm.weight",  # Gemma2 safetensors only has pre_feedforward_layernorm
    # MLP
    ".ffn_up.weight": ".mlp.up_proj.weight",
    ".ffn_down.weight": ".mlp.down_proj.weight",
    ".ffn_gate.weight": ".mlp.gate_proj.weight",
    # emb/out
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
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

def sd_map_replace(raw_sd, key_map):
    sd = {}
    for k,v in raw_sd.items():
        orig_k = k
        for s,d in key_map.items():
            if s in k:
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
    vsd = gguf_sd_loader(target, is_text_model=True)

    # concat 4D to 5D
    if "v.patch_embd.weight.1" in vsd:
        w1 = dequantize_tensor(vsd.pop("v.patch_embd.weight"), dtype=torch.float32)
        w2 = dequantize_tensor(vsd.pop("v.patch_embd.weight.1"), dtype=torch.float32)
        vsd["v.patch_embd.weight"] = torch.stack([w1, w2], dim=2)

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
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError("Please make sure sentencepiece and protobuf are installed.\npip install sentencepiece protobuf")
    
    reader = gguf.GGUFReader(path)
    
    proto_tensor = None
    try:
        for tensor in reader.tensors:
            if tensor.name == "tokenizer.ggml.spiece_model_raw":
                proto_tensor = torch.from_numpy(tensor.data)
                break
    except Exception as e:
        logging.warning(f"Failed to read tokenizer.ggml.spiece_model_raw tensor: {e}")
        proto_tensor = None
    if proto_tensor is not None:
        try:
            proto_bytes = proto_tensor.cpu().numpy().tobytes()
            spm = model.ModelProto()
            spm.ParseFromString(proto_bytes)
            vocab_size = len(spm.pieces)
            logging.info(f"✓ Loaded complete sentencepiece proto from GGUF tensor: {vocab_size} pieces, {len(proto_bytes)} bytes")
            logging.info(f"  unk_id={spm.trainer_spec.unk_id}, bos_id={spm.trainer_spec.bos_id}, "
                        f"eos_id={spm.trainer_spec.eos_id}, pad_id={spm.trainer_spec.pad_id}")
            if temb_shape[0] != vocab_size:
                logging.warning(f"Proto vocab_size ({vocab_size}) != embedding shape[0] ({temb_shape[0]})")
            del reader
            return torch.ByteTensor(list(proto_bytes))
        except Exception as e:
            logging.warning(f"Failed to parse proto from int8 tensor: {e}")
    spiece_tensor = reader.get_tensor("tokenizer.ggml.spiece_model_raw")
    if spiece_tensor is not None:
        del reader
        return spiece_tensor
    raw_proto_field = get_field(reader, "tokenizer.ggml.spiece_model_raw", str)
    if raw_proto_field is not None:
        proto_bytes = raw_proto_field.encode('latin1')
        del reader
        return torch.ByteTensor(list(proto_bytes))
    del reader
    raise NotImplementedError("No sentencepiece proto found in GGUF metadata!")

def gguf_clip_loader(path):
    sd, arch = gguf_sd_loader(path, return_arch=True, is_text_model=True)
    if arch in {"t5", "t5encoder"}:
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape == (256384, 4096):
            # non-standard Comfy-Org tokenizer
            sd["spiece_model"] = gguf_tokenizer_loader(path, sd[temb_key].shape)
            # TODO: dequantizing token embed here is janky but otherwise we OOM due to tensor being massive.
            logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        sd = sd_map_replace(sd, T5_SD_MAP)
    elif arch in {"llama", "qwen2vl"}:
        # TODO: pass model_options["vocab_size"] to loader somehow
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape[0] >= (64 * 1024):
            # See note above for T5.
            logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        sd = sd_map_replace(sd, LLAMA_SD_MAP)
        if arch == "llama":
            sd = llama_permute(sd, 32, 8) # L3
        if arch == "qwen2vl":
            vsd = gguf_mmproj_loader(path)
            sd.update(vsd)
    elif arch == "gemma2":
        temb_key = "token_embd.weight"
        # Load tokenizer from GGUF metadata
        if temb_key in sd:
            try:
                spm_tensor = gguf_tokenizer_loader(path, sd[temb_key].shape)
                if spm_tensor is not None:
                    sd["spiece_model"] = spm_tensor
            except NotImplementedError as e:
                logging.error(f"[Gemma2] Failed to load tokenizer: {e}")
                raise
            if sd[temb_key].shape[0] >= (64 * 1024):
                # Dequantize token embeddings to prevent OOM
                logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
                sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        sd = sd_map_replace(sd, GEMMA2_SD_MAP)
        # Gemma2_2B has 8 attention heads and 4 key-value heads
        sd = llama_permute(sd, 8, 4)
        fix_keys = {}
        for k in list(sd.keys()):
            if k.startswith("model.layers."):
                if (
                    ("layernorm" in k or "mlp." in k or "proj" in k)
                    and not k.endswith(".weight")
                    and not k.endswith(".bias")
                ):
                    fix_keys[k+".weight"] = sd[k]
                    del sd[k]
        sd.update(fix_keys)
    else:
        pass
    return sd
