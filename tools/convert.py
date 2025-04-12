# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import gguf
import torch
import argparse
from tqdm import tqdm
from safetensors.torch import load_file, save_file

QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127
MAX_TENSOR_DIMS = 4

class ModelTemplate:
    arch = "invalid"  # string describing architecture
    shape_fix = False # whether to reshape tensors
    ndims_fix = False # whether to save fix file for tensors exceeding max dims
    keys_detect = []  # list of lists to match in state dict
    keys_banned = []  # list of keys that should mark model as invalid for conversion
    keys_hiprec = []  # list of keys that need to be kept in fp32 for some reason

class ModelFlux(ModelTemplate):
    arch = "flux"
    keys_detect = [
        ("transformer_blocks.0.attn.norm_added_k.weight",),
        ("double_blocks.0.img_attn.proj.weight",),
    ]
    keys_banned = ["transformer_blocks.0.attn.norm_added_k.weight",]

class ModelSD3(ModelTemplate):
    arch = "sd3"
    keys_detect = [
        ("transformer_blocks.0.attn.add_q_proj.weight",),
        ("joint_blocks.0.x_block.attn.qkv.weight",),
    ]
    keys_banned = ["transformer_blocks.0.attn.add_q_proj.weight",]

class ModelAura(ModelTemplate):
    arch = "aura"
    keys_detect = [
        ("double_layers.3.modX.1.weight",),
        ("joint_transformer_blocks.3.ff_context.out_projection.weight",),
    ]
    keys_banned = ["joint_transformer_blocks.3.ff_context.out_projection.weight",]

class ModelHyVid(ModelTemplate):
    arch = "hyvid"
    ndims_fix = True
    keys_detect = [
        (
            "double_blocks.0.img_attn_proj.weight",
            "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight",
        )
    ]

class ModelWan(ModelTemplate):
    arch = "wan"
    ndims_fix = True
    keys_detect = [
        (
            "blocks.0.self_attn.norm_q.weight",
            "text_embedding.2.weight",
            "head.modulation",
        )
    ]
    keys_hiprec = [
        ".modulation" # nn.parameter, can't load from BF16 ver
    ]

class ModelLTXV(ModelTemplate):
    arch = "ltxv"
    keys_detect = [
        (
            "adaln_single.emb.timestep_embedder.linear_2.weight",
            "transformer_blocks.27.scale_shift_table",
            "caption_projection.linear_2.weight",
        )
    ]
    keys_hiprec = [
        "scale_shift_table" # nn.parameter, can't load from BF16 base quant
    ]

class ModelSDXL(ModelTemplate):
    arch = "sdxl"
    shape_fix = True
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight",
            "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight",
        ), # Non-diffusers
        ("label_emb.0.0.weight",),
    ]

class ModelSD1(ModelTemplate):
    arch = "sd1"
    shape_fix = True
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "input_blocks.9.0.op.weight",
            "output_blocks.2.1.conv.weight", "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight"
        ), # Non-diffusers
    ]

# The architectures are checked in order and the first successful match terminates the search.
arch_list = [ModelFlux, ModelSD3, ModelAura, ModelLTXV, ModelHyVid, ModelWan, ModelSDXL, ModelSD1]

def is_model_arch(model, state_dict):
    # check if model is correct
    matched = False
    invalid = False
    for match_list in model.keys_detect:
        if all(key in state_dict for key in match_list):
            matched = True
            invalid = any(key in state_dict for key in model.keys_banned)
            break
    assert not invalid, "Model architecture not allowed for conversion! (i.e. reference VS diffusers format)"
    return matched

def detect_arch(state_dict):
    model_arch = None
    for arch in arch_list:
        if is_model_arch(arch, state_dict):
            model_arch = arch()
            break
    assert model_arch is not None, "Unknown model architecture!"
    return model_arch

def parse_args():
    parser = argparse.ArgumentParser(description="Generate F16 GGUF files from single UNET")
    parser.add_argument("--src", required=True, help="Source model ckpt file.")
    parser.add_argument("--dst", help="Output unet gguf file.")
    args = parser.parse_args()

    if not os.path.isfile(args.src):
        parser.error("No input provided!")

    return args

def strip_prefix(state_dict):
    # only keep unet with no prefix!
    prefix = None
    for pfx in ["model.diffusion_model.", "model."]:
        if any([x.startswith(pfx) for x in state_dict.keys()]):
            prefix = pfx
            break

    sd = {}
    for k, v in state_dict.items():
        if prefix and prefix not in k:
            continue
        if prefix:
            k = k.replace(prefix, "")
        sd[k] = v

    return sd

def find_main_dtype(state_dict, allow_fp32=False):
    # detect most common dtype in input
    dtypes = [x.dtype for x in state_dict.values()]
    dtypes = {x:dtypes.count(x) for x in set(dtypes)}
    main_dtype = max(dtypes, key=dtypes.get)

    if main_dtype == torch.bfloat16:
        ftype_name = "BF16"
        ftype_gguf = gguf.LlamaFileType.MOSTLY_BF16
    elif main_dtype == torch.float32 and allow_fp32:
        ftype_name = "F32"
        ftype_gguf = gguf.LlamaFileType.ALL_F32
    else:
        ftype_name = "F16"
        ftype_gguf = gguf.LlamaFileType.MOSTLY_F16

    return ftype_name, ftype_gguf

def load_state_dict(path):
    if any(path.endswith(x) for x in [".ckpt", ".pt", ".bin", ".pth"]):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        for subkey in ["model", "module"]:
            if subkey in state_dict:
                state_dict = state_dict[subkey]
                break
        if len(state_dict) < 20:
            raise RuntimeError(f"pt subkey load failed: {state_dict.keys()}")
    else:
        state_dict = load_file(path)

    return strip_prefix(state_dict)

def handle_tensors(writer, state_dict, model_arch, allow_fp32=False):
    name_lengths = tuple(sorted(
        ((key, len(key)) for key in state_dict.keys()),
        key=lambda item: item[1],
        reverse=True,
    ))
    if not name_lengths:
        return
    max_name_len = name_lengths[0][1]

    if max_name_len > MAX_TENSOR_NAME_LENGTH:
        bad_list = ", ".join(f"{key!r} ({namelen})" for key, namelen in name_lengths if namelen > MAX_TENSOR_NAME_LENGTH)
        raise ValueError(f"Can only handle tensor names up to {MAX_TENSOR_NAME_LENGTH} characters. Tensors exceeding the limit: {bad_list}")

    invalid_tensors = {}
    quantized_tensors = {}
    for key, data in tqdm(state_dict.items()):
        old_dtype = data.dtype

        if data.dtype == torch.bfloat16:
            data = data.to(torch.float32).numpy()
        # this is so we don't break torch 2.0.X
        elif data.dtype in [getattr(torch, "float8_e4m3fn", "_invalid"), getattr(torch, "float8_e5m2", "_invalid")]:
            data = data.to(torch.float16).numpy()
        else:
            data = data.numpy()

        n_dims = len(data.shape)
        data_shape = data.shape
        if old_dtype == torch.bfloat16:
            data_qtype = gguf.GGMLQuantizationType.BF16
        elif old_dtype == torch.float32 and allow_fp32:
            data_qtype = gguf.GGMLQuantizationType.F32
        else:
            data_qtype = gguf.GGMLQuantizationType.F16

        # The max no. of dimensions that can be handled by the quantization code is 4
        if len(data.shape) > MAX_TENSOR_DIMS:
            invalid_tensors[key] = data
            continue # needs to be added back later

        # get number of parameters (AKA elements) in this tensor
        n_params = 1
        for dim_size in data_shape:
            n_params *= dim_size

        if old_dtype in (torch.float32, torch.bfloat16):
            if n_dims == 1:
                # one-dimensional tensors should be kept in F32
                # also speeds up inference due to not dequantizing
                data_qtype = gguf.GGMLQuantizationType.F32

            elif n_params <= QUANTIZATION_THRESHOLD:
                # very small tensors
                data_qtype = gguf.GGMLQuantizationType.F32

            elif any(x in key for x in model_arch.keys_hiprec):
                # tensors that require max precision
                data_qtype = gguf.GGMLQuantizationType.F32

        if (model_arch.shape_fix                        # NEVER reshape for models such as flux
            and n_dims > 1                              # Skip one-dimensional tensors
            and n_params >= REARRANGE_THRESHOLD         # Only rearrange tensors meeting the size requirement
            and (n_params / 256).is_integer()           # Rearranging only makes sense if total elements is divisible by 256
            and not (data.shape[-1] / 256).is_integer() # Only need to rearrange if the last dimension is not divisible by 256
        ):
            orig_shape = data.shape
            data = data.reshape(n_params // 256, 256)
            writer.add_array(f"comfy.gguf.orig_shape.{key}", tuple(int(dim) for dim in orig_shape))

        try:
            data = gguf.quants.quantize(data, data_qtype)
            quantized_tensors[key] = data_qtype
        except (AttributeError, gguf.QuantError) as e:
            tqdm.write(f"falling back to F16: {e}")
            data_qtype = gguf.GGMLQuantizationType.F16
            data = gguf.quants.quantize(data, data_qtype)
            quantized_tensors[key] = data_qtype

        shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
        tqdm.write(f"{f'%-{max_name_len + 4}s' % f'{key}'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

        writer.add_tensor(key, data, raw_dtype=data_qtype)
    
    return quantized_tensors, invalid_tensors

def convert_file(path, dst_path=None, interact=True, overwrite=False, allow_fp32=False):
    # load & run model detection logic
    state_dict = load_state_dict(path)
    model_arch = detect_arch(state_dict)
    print(f"* Architecture detected from input: {model_arch.arch}")

    ftype_name, ftype_gguf = find_main_dtype(state_dict, allow_fp32=allow_fp32)

    if dst_path is None:
        dst_path = f"{os.path.splitext(path)[0]}-{ftype_name}.gguf"
    elif "{ftype}" in dst_path: # lcpp logic
        dst_path = dst_path.replace("{ftype}", ftype_name)

    if os.path.isfile(dst_path) and not overwrite:
        if interact:
            input("Output exists enter to continue or ctrl+c to abort!")
        else:
            raise OSError("Output exists and overwriting is disabled!")

    # handle actual file
    writer = gguf.GGUFWriter(path=None, arch=model_arch.arch)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    if ftype_gguf is not None:
        writer.add_file_type(ftype_gguf)

    quantized_tensors, invalid_tensors = handle_tensors(writer, state_dict, model_arch, allow_fp32=allow_fp32)
    if len(invalid_tensors) > 0:
        if not model_arch.ndims_fix: # only applies to 5D fix for now, possibly expand to cover more cases?
            raise ValueError(f"Tensor(s) detected that exceeds dims supported by C++ code! ({invalid_tensors.keys()})")

        fix_path = os.path.join(
            os.path.dirname(dst_path),
            f"fix_5d_tensors_{model_arch.arch}.safetensors"
        )
        if os.path.isfile(fix_path):
            raise RuntimeError(f"Tensor fix file already exists! {path}")

        invalid_tensors = {k:torch.from_numpy(v.copy()) for k,v in invalid_tensors.items()}
        save_file(invalid_tensors, fix_path)
        print(f"\n### Warning! Fix file saved to '{fix_path}'")
        print(f" you most likely need to run 'fix_5d_tensors.py' after quantization.")
    else:
        fix_path = None

    writer.write_header_to_file(path=dst_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    return dst_path, model_arch, fix_path

if __name__ == "__main__":
    args = parse_args()
    convert_file(args.src, args.dst)
