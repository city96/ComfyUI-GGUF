# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import torch
import gguf # This needs to be the llama.cpp one specifically!
import argparse
from tqdm import tqdm

from safetensors.torch import load_file

QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127

# Tuple of arch_name, match_lists.
# Each item in match_lists is a tuple of keys that must match.
# All keys in a match_lists item must exist for the architecture to match.
# The architectures are checked in order and the first successful match terminates the search.
MODEL_DETECTION = (
    ("flux", (
        ("transformer_blocks.0.attn.norm_added_k.weight",),
        ("double_blocks.0.img_attn.proj.weight",),
    )),
    ("sd3", (
        ("transformer_blocks.0.attn.add_q_proj.weight",),
    )),
    ("sdxl", (
        ("down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight",
            "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight",
        ), # Non-diffusers
        ("label_emb.0.0.weight",),
    )),
    ("sd1", (
        ("down_blocks.0.downsamplers.0.conv.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "input_blocks.9.0.op.weight",
            "output_blocks.2.1.conv.weight", "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight"
        ), # Non-diffusers
    )),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate F16 GGUF files from single UNET")
    parser.add_argument("--src", required=True, help="Source model ckpt file.")
    parser.add_argument("--dst", help="Output unet gguf file.")
    args = parser.parse_args()

    if not os.path.isfile(args.src):
        parser.error("No input provided!")

    return args

def load_state_dict(path):
    if any(path.endswith(x) for x in [".ckpt", ".pt", ".bin", ".pth"]):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = state_dict.get("model", state_dict)
    else:
        state_dict = load_file(path)

    # only keep unet with no prefix!
    sd = {}
    has_prefix = any(["model.diffusion_model." in x for x in state_dict.keys()])
    for k, v in state_dict.items():
        if has_prefix and "model.diffusion_model." not in k:
            continue
        if has_prefix:
            k = k.replace("model.diffusion_model.", "")
        sd[k] = v

    return sd

def detect_arch(state_dict):
    for arch, match_lists in MODEL_DETECTION:
        for match_list in match_lists:
            if all(key in state_dict for key in match_list):
                return arch
    breakpoint()
    raise ValueError("Unknown model architecture!")


def load_model(path):
    state_dict = load_state_dict(path)
    arch = detect_arch(state_dict)
    print(f"* Architecture detected from input: {arch}")
    if arch == "flux" and "transformer_blocks.0.attn.norm_added_k.weight" in state_dict:
        raise ValueError("The Diffusers UNET can not be used for this!")
    writer = gguf.GGUFWriter(path=None, arch=arch)
    return (writer, state_dict)

def handle_tensors(args, writer, state_dict):
    # TODO list:
    # - do something about this being awful and hacky

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
    for key, data in tqdm(state_dict.items()):
        old_dtype = data.dtype

        if data.dtype == torch.bfloat16:
            data = data.to(torch.float32).numpy()
        else:
            data = data.numpy()

        n_dims = len(data.shape)
        data_shape = data.shape
        data_qtype = getattr(
            gguf.GGMLQuantizationType,
            "BF16" if old_dtype == torch.bfloat16 else "F16"
        )

        # get number of parameters (AKA elements) in this tensor
        n_params = 1
        for dim_size in data_shape:
            n_params *= dim_size

        # keys to keep as max precision
        blacklist = {
            "time_embedding.",
            "add_embedding.",
            "time_in.",
            "txt_in.",
            "vector_in.",
            "img_in.",
            "guidance_in.",
            "final_layer.",
        }

        if old_dtype in (torch.float32, torch.bfloat16):
            if n_dims == 1:
                # one-dimensional tensors should be kept in F32
                # also speeds up inference due to not dequantizing
                data_qtype = gguf.GGMLQuantizationType.F32

            elif n_params <= QUANTIZATION_THRESHOLD:
                # very small tensors
                data_qtype = gguf.GGMLQuantizationType.F32

            elif ".weight" in key and any(x in key for x in blacklist):
                data_qtype = gguf.GGMLQuantizationType.F32

        if (    n_dims > 1                              # Skip one-dimensional tensors
            and n_params >= REARRANGE_THRESHOLD         # Only rearrange tensors meeting the size requirement
            and (n_params / 256).is_integer()           # Rearranging only makes sense if total elements is divisible by 256
            and not (data.shape[-1] / 256).is_integer() # Only need to rearrange if the last dimension is not divisible by 256
        ):
            orig_shape = data.shape
            data = data.reshape(n_params // 256, 256)
            writer.add_array(f"comfy.gguf.orig_shape.{key}", tuple(int(dim) for dim in orig_shape))

        try:
            data = gguf.quants.quantize(data, data_qtype)
        except (AttributeError, gguf.QuantError) as e:
            tqdm.write(f"falling back to F16: {e}")
            data_qtype = gguf.GGMLQuantizationType.F16
            data = gguf.quants.quantize(data, data_qtype)

        new_name = key # do we need to rename?

        shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
        tqdm.write(f"{f'%-{max_name_len + 4}s' % f'{new_name}'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

        writer.add_tensor(new_name, data, raw_dtype=data_qtype)

if __name__ == "__main__":
    args = parse_args()
    path = args.src
    writer, state_dict = load_model(path)

    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    if next(iter(state_dict.values())).dtype == torch.bfloat16:
        out_path = f"{os.path.splitext(path)[0]}-BF16.gguf"
        writer.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
    else:
        out_path = f"{os.path.splitext(path)[0]}-F16.gguf"
        writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    out_path = args.dst or out_path
    if os.path.isfile(out_path):
        input("Output exists enter to continue or ctrl+c to abort!")

    handle_tensors(path, writer, state_dict)
    writer.write_header_to_file(path=out_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
