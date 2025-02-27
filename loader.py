# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import torch
import gguf

from .ops import GGMLTensor
from .dequant import is_quantized

IMG_ARCH_LIST = {"flux", "sd1", "sdxl", "sd3", "aura", "ltxv", "hyvid", "wan"}
TXT_ARCH_LIST = {"t5", "t5encoder", "llama"}

def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))

def gguf_sd_loader(path, handle_prefix="model.diffusion_model.", return_arch=False):
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
    arch_str = None
    arch_field = reader.get_field("general.architecture")
    if arch_field is not None:
        if len(arch_field.types) != 1 or arch_field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF general.architecture key: expected string, got {arch_field.types!r}")
        arch_str = str(arch_field.parts[arch_field.data[-1]], encoding="utf-8")
        if arch_str not in IMG_ARCH_LIST and arch_str not in TXT_ARCH_LIST:
            raise ValueError(f"Unexpected architecture type in GGUF file, expected one of flux, sd1, sdxl, t5encoder but got {arch_str!r}")
    else: # stable-diffusion.cpp
        # import here to avoid changes to convert.py breaking regular models
        from .tools.convert import detect_arch
        arch_str = detect_arch(set(val[0] for val in tensors)).arch
        compat = "sd.cpp"

    # main loading loop
    state_dict = {}
    qtype_dict = {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        tensor_type_str = str(tensor.tensor_type)
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
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    # mark largest tensor for vram estimation
    qsd = {k:v for k,v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    # sanity check debug print
    print("\nggml_sd_loader:")
    for k,v in qtype_dict.items():
        print(f" {k:30}{v:3}")

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

def gguf_clip_loader(path):
    sd, arch = gguf_sd_loader(path, return_arch=True)
    if arch in {"t5", "t5encoder"}:
        sd = sd_map_replace(sd, T5_SD_MAP)
    elif arch in {"llama"}:
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape != (128320, 4096):
            # This still works. Raise error?
            print("Warning! token_embd shape may be incorrect for llama 3 model!")
        sd = sd_map_replace(sd, LLAMA_SD_MAP)
        sd = llama_permute(sd, 32, 8) # L3
    else:
        pass
    return sd
