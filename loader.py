# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import warnings
import logging
import torch
import gguf
import re
import os
import numpy as np

from gguf import GGUFReader, GGUFValueType
from .ops import GGMLTensor
from .dequant import is_quantized, dequantize_tensor
from .quant_ops import make_quantized

IMG_ARCH_LIST = {"flux", "sd1", "sdxl", "sd3", "aura", "hidream", "cosmos", "ltxv", "hyvid", "wan", "lumina2", "qwen_image", "ideogram4", "krea2"}
TXT_ARCH_LIST = {"t5", "t5encoder", "llama", "qwen2vl", "qwen3", "qwen3vl", "gemma3", "gemma4"}
VIS_TYPE_LIST = {"clip-vision", "mmproj"}

class LazyGGUFReader(GGUFReader):
    def _get_field_parts(self, orig_offs: int, raw_type: int):
        gtype = GGUFValueType(raw_type)
        
        if gtype == GGUFValueType.ARRAY:
            raw_itype = self._get(orig_offs, np.uint32)
            offs = orig_offs + int(raw_itype.nbytes)
            alen = self._get(offs, np.uint64)
            array_len = alen[0]

            if array_len > 1000:
                offs += int(alen.nbytes) 
                sub_type = raw_itype[0]
                types = [gtype, GGUFValueType(sub_type)]
                aparts = [raw_itype, alen]
                data_idxs = []
                data_view = self.data
                is_swapped = (self.byte_order == 'S')
                
                if sub_type == 8:
                    for _ in range(array_len):
                        slen_arr = data_view[offs : offs + 8]
                        slen = slen_arr.view(dtype=np.uint64)[0]
                        if is_swapped:
                            slen = slen.newbyteorder('S')
                        
                        str_total_bytes = 8 + int(slen)
                        sdata_arr = data_view[offs + 8 : offs + str_total_bytes]
                        
                        idxs_offs = len(aparts)
                        aparts.append(slen_arr)
                        aparts.append(sdata_arr)
                        data_idxs.append(idxs_offs + 1)                      
                        offs += str_total_bytes

                    return offs - orig_offs, aparts, data_idxs, types
                else:
                    nptype = self.gguf_scalar_to_np.get(GGUFValueType(sub_type))
                    if nptype is not None:
                        item_size = np.dtype(nptype).itemsize
                        total_bytes = array_len * item_size
                        total_data = data_view[offs : offs + total_bytes].view(dtype=nptype)
                        if is_swapped:
                            total_data = total_data.newbyteorder('S')
                        
                        idxs_offs = len(aparts)
                        aparts.extend(total_data[i : i + 1] for i in range(array_len))
                        data_idxs = list(range(idxs_offs, idxs_offs + array_len))                      
                        offs += total_bytes

                        return offs - orig_offs, aparts, data_idxs, types

        return super()._get_field_parts(orig_offs, raw_type)


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

def gguf_sd_loader(path, handle_prefix="model.diffusion_model.", is_text_model=False, dynamic=False):
    """
    Read state dict as fake tensors
    """
    reader = LazyGGUFReader(path)

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
        if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            state_dict[sd_key] = torch_tensor.view(torch.bfloat16).reshape(*shape)
        elif tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            state_dict[sd_key] = torch_tensor.view(*shape)
        elif dynamic:
            state_dict[sd_key] = make_quantized(torch_tensor, tensor.tensor_type, shape)
        else:
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

    # extra info to return
    extra = {
        "arch_str": arch_str,
        "metadata": get_gguf_metadata(reader)
    }
    if is_text_model:
        extra["reader"] = reader
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

GEMMA4_SD_MAP = {
    "per_layer_token_embd": "model.embed_tokens_per_layer",
    "per_layer_model_proj": "model.per_layer_model_projection",
    "proj.weight": "per_layer_projection.weight",
}
GEMMA4_SD_MAP.update(GEMMA3_SD_MAP)
GEMMA4_SD_MAP.update({
    "layer_output_scale.weight": "layer_scalar",
    "inp_gate.weight": "per_layer_input_gate.weight",
    "post_norm.weight": "post_per_layer_input_norm.weight",
    "per_layer_proj_norm": "model.per_layer_projection_norm",
})

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
    "v.deepstast.": "model.visual.deepstack_merger_list.",
}

CLIP_VISION_GEMMA4_MAP = {
    "v.position_embd.weight": "vision_model.patch_embedder.position_embedding_table",
    "mm.input_projection": "multi_modal_projector.embedding_projection",
    "mm.a.input_projection": "audio_projector.embedding_projection",
    "attn_post_norm": "post_attention_layernorm",
    "ffn_post_norm": "post_feedforward_layernorm",
    "attn_k_norm": "self_attn.k_norm",
    "attn_q_norm": "self_attn.q_norm",
    "ln1.weight": "input_layernorm.weight",
    "ln2.weight": "pre_feedforward_layernorm.weight",
    "attn_out.": "self_attn.o_proj.",
    "attn_k.": "self_attn.k_proj.",
    "attn_q.": "self_attn.q_proj.",
    "attn_v.": "self_attn.v_proj.",
    "ffn_down.": "mlp.down_proj.",
    "ffn_gate.": "mlp.gate_proj.",
    "ffn_up.": "mlp.up_proj.",
    "r0.weight": "r0.conv.weight",
    "r1.weight": "r1.conv.weight",
    "v.blk": "vision_model.encoder.layers",
    "_proj.weight": "_proj.linear.weight",
    "v.patch_embd.": "vision_model.patch_embedder.input_proj.",
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

def strip_quant_suffix(name):
    pattern = r"[-_]?(?:ud-)?i?q[0-9]_[a-z0-9_\-]{1,8}$"
    match = re.search(pattern, name, re.IGNORECASE)
    if match:
        name = name[:match.start()]
    return name

def gguf_mmproj_loader(path, dynamic=False):
    # Reverse version of Qwen2VLVisionModel.modify_tensors
    logging.info("Attempting to find mmproj file for text encoder...")

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
        logging.warning(f"Can't find mmproj file for '{tenc_fname}' (matching:'{tenc}'), vision function will not work!")
        return {}
    if len(target) > 1:
        logging.info(f"Ambiguous mmproj for text encoder '{tenc_fname}', will use first match.")

    logging.info(f"Using mmproj '{target[0]}' for text encoder '{tenc_fname}'.")
    target = os.path.join(root, target[0])
    vsd, _ = gguf_sd_loader(target, is_text_model=True, dynamic=dynamic)

    # gemma4
    if "mm.a.input_projection.weight" in vsd:
        vsd["v.patch_embd.weight"] = vsd["v.patch_embd.weight"].permute(0, 2, 3, 1).flatten(start_dim=1)  
        return sd_map_replace(vsd, CLIP_VISION_GEMMA4_MAP)

    # concat 4D to 5D
    if "v.patch_embd.weight.1" in vsd:
        w1 = dequantize_tensor(vsd.pop("v.patch_embd.weight"), dtype=torch.float32)
        w2 = dequantize_tensor(vsd.pop("v.patch_embd.weight.1"), dtype=torch.float32)
        vsd["v.patch_embd.weight"] = torch.stack([w1, w2], dim=2)

    # qwen3vl
    if any("deepstack" in key for key in vsd): 
        return sd_map_replace(vsd, CLIP_VISION_QWEN3_MAP)

    # qwen2vl
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

def gguf_tokenizer_loader(reader, temb_shape):
    # convert gguf tokenizer to spiece
    logging.info("Attempting to recreate sentencepiece tokenizer from GGUF file metadata...")
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError("Please make sure sentencepiece and protobuf are installed.\npip install sentencepiece protobuf")
    spm = model.ModelProto()

    if get_field(reader, "tokenizer.ggml.model", str) == "t5":
        if temb_shape == (256384, 4096): # probably UMT5
            spm.trainer_spec.model_type = 1 # Unigram (do we have a T5 w/ BPE?)
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable")
        return torch.frombuffer(spm.SerializeToString(), dtype=torch.uint8)

def gguf_tekken_tokenizer_loader(reader, temb_shape):
    # convert ggml (hf) tokenizer metadata to tekken/comfy data
    logging.info("Attempting to recreate tekken tokenizer from GGUF file metadata...")
    import json
    import base64
    from transformers.convert_slow_tokenizer import bytes_to_unicode

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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable")
        return torch.frombuffer(json.dumps(data).encode('utf-8'), dtype=torch.uint8)

def gguf_gemma3_tokenizer_loader(reader):
    #TODO: merge into gguf_tokenizer_loader
    logging.info("Attempting to recreate sentencepiece tokenizer from GGUF file metadata...")
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError("Please install sentencepiece and protobuf.\npip install sentencepiece protobuf")
    spm = model.ModelProto()

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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable")
        return torch.frombuffer(spm.SerializeToString(), dtype=torch.uint8)

def gguf_gemma4_tokenizer_loader(reader):
    # convert gguf tokenizer to spiece
    logging.info("Attempting to recreate tokenizer from GGUF file metadata...")
    import json

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    merges = get_list_field(reader, "tokenizer.ggml.merges", str)
    del reader

    if not tokens or not merges:
        raise ValueError("Missing tokenizer metadata")

    vocab = {token: idx for idx, token in enumerate(tokens)}
    target_special_ids = [
        0, 1, 2, 3, 4, 46, 47, 48, 49, 50, 51, 52, 98, 100, 101, 105, 106, 255999, 256000, 258880, 258881, 258882, 258883, 258884
    ]
    
    added_tokens = []
    for sp_id in target_special_ids:
        if sp_id < len(tokens):
            added_tokens.append({
                "id": sp_id,
                "content": tokens[sp_id],
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            })

    tokenizer_dict = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {
            "type": "Replace",
            "pattern": {"String": " "},
            "content": "\u2581"
        },
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"String": " "},
            "behavior": "MergedWithPrevious",
            "invert": False
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}}
            ],
            "special_tokens": {}
        },
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"}
            ]
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges
        }
    }
    
    json_string = json.dumps(tokenizer_dict, ensure_ascii=False)
    
    logging.info(f"Created tokenizer with vocab size of {len(vocab)}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable")
        return torch.frombuffer(json_string.encode('utf-8'), dtype=torch.uint8)

def gguf_json_tokenizer_loader(path):
    tenc_fname = os.path.basename(path)
    tenc = os.path.splitext(tenc_fname)[0].lower()
    tenc = strip_quant_suffix(tenc)

    target = []
    root = os.path.dirname(path)
    for fname in os.listdir(root):
        name, ext = os.path.splitext(fname)
        if ext.lower() != ".json":
            continue
        if "tokenizer" not in name.lower():
            continue
        if tenc in name.lower():
            target.append(fname)

    if len(target) == 0:
        logging.info(f"Can't find tokenizer file for '{tenc_fname}' (matching:'{tenc}')!")
        return None
    if len(target) > 1:
        logging.info(f"Ambiguous tokenizer for text encoder '{tenc_fname}', will use first match.")

    logging.info(f"Using tokenizer '{target[0]}' for text encoder '{tenc_fname}'.")
    target = os.path.join(root, target[0])
    
    with open(target, "rb") as f:
        tokenizer_bytes = f.read()  
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable")
        return torch.frombuffer(tokenizer_bytes, dtype=torch.uint8)

def gguf_clip_loader(path, dynamic=False):
    sd, extra = gguf_sd_loader(path, is_text_model=True, dynamic=dynamic)
    arch = extra.get("arch_str", None)
    if arch in {"t5", "t5encoder"}:
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape == (256384, 4096):
            # non-standard Comfy-Org tokenizer
            sd["spiece_model"] = gguf_tokenizer_loader(extra.pop("reader"), sd[temb_key].shape)
            # TODO: dequantizing token embed here is janky but otherwise we OOM due to tensor being massive.
            logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        sd = sd_map_replace(sd, T5_SD_MAP)
    elif arch in {"llama", "qwen2vl", "qwen3", "qwen3vl", "gemma3", "gemma4"}:
        # TODO: pass model_options["vocab_size"] to loader somehow
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape[0] >= (64 * 1024):
            if arch == "llama" and sd[temb_key].shape == (131072, 5120):
                # non-standard Comfy-Org tokenizer
                sd["tekken_model"] = gguf_tekken_tokenizer_loader(extra.pop("reader"), sd[temb_key].shape)
            elif arch == "gemma3":
                sd["spiece_model"] = gguf_gemma3_tokenizer_loader(extra.pop("reader"))
            if arch == "gemma4":
                sd["tokenizer_json"] = gguf_gemma4_tokenizer_loader(extra.pop("reader"))
            else:
                # See note above for T5.
                logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
                sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        if arch == "gemma3":
            sd = sd_map_replace(sd, GEMMA3_SD_MAP)
            sd = gemma3_norm_corrections(sd)
        elif arch == "gemma4":
            sd = sd_map_replace(sd, GEMMA4_SD_MAP)

            # temporary workaround
            sd["model.embed_tokens.weight"] = dequantize_tensor(sd["model.embed_tokens.weight"], dtype=torch.bfloat16)
            sd["model.embed_tokens_per_layer.weight"] = dequantize_tensor(sd["model.embed_tokens_per_layer.weight"], dtype=torch.bfloat16).as_subclass(torch.Tensor)
            sd["model.norm.weight"] = dequantize_tensor(sd["model.norm.weight"], dtype=torch.bfloat16)
        else:
            sd = sd_map_replace(sd, LLAMA_SD_MAP)
        if arch == "llama":
            sd = llama_permute(sd, 32, 8) # L3 / Mistral
        if arch in {"qwen2vl", "qwen3vl", "gemma4"}:
            vsd = gguf_mmproj_loader(path, dynamic=dynamic)

        if vsd:
        # MiniMax-H3 uses the truncated Qwen3-VL-32B encoder.
        # ComfyUI detects it by:
        #   visual.deepstack_merger_list...
        #   model.layers.49...
        #
        # The generic Qwen3-VL mmproj mapper produces model.visual.*,
        # which makes ComfyUI incorrectly instantiate Qwen3-VL-8B.
            is_minimax_h3 = (
                arch == "qwen3vl"
                and "model.layers.49.self_attn.q_proj.weight" in sd
            )

            if is_minimax_h3:
                vsd = {
                    (
                        key.replace("model.visual.", "visual.", 1)
                        if key.startswith("model.visual.")
                        else key
                    ): value
                    for key, value in vsd.items()
                }

            sd.update(vsd)

        elif arch == "qwen3vl" and "model.norm.weight" in sd:
        # Generic full-model fallback only.
            weight = sd["model.norm.weight"].shape[0]
            sd["model.visual.deepstack_merger_list.0.norm.weight"] = torch.zeros(
                4096 if weight < 4096 else 4608
            )
            sd["model.visual.merger.linear_fc2.weight"] = torch.zeros(weight)
    else:
        pass
    return sd

