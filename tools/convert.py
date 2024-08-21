# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import sys
import torch
import numpy as np
import gguf # This needs to be the llama.cpp one specifically!
import argparse
from tqdm import tqdm

from safetensors.torch import load_file

def parse_args():
    parser = argparse.ArgumentParser(description="Generate F16 GGUF files from single UNET")
    parser.add_argument("--src", required=True, help="Source model ckpt file.")
    parser.add_argument("--dst", help="Output  unet gguf file.")
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

def load_model(path):
    state_dict = load_state_dict(path)

    # from ComfyUI model detection
    if "transformer_blocks.0.attn.norm_added_k.weight" in state_dict:
        arch = "flux"
        raise ValueError(f"The Diffusers UNET can not be used for this!")
    elif "double_blocks.0.img_attn.proj.weight" in state_dict:
        arch = "flux" # mmdit ...?
    elif "transformer_blocks.0.attn.add_q_proj.weight" in state_dict:
        arch = "sd3"
    elif "down_blocks.0.downsamplers.0.conv.weight" in state_dict:
        if "add_embedding.linear_1.weight" in state_dict:
            arch = "sdxl"
        else:
            arch = "sd1" 
    else:
        breakpoint()
        raise ValueError(f"Unknown model architecture!")

    writer = gguf.GGUFWriter(path=None, arch=arch)
    return (writer, state_dict)

def handle_tensors(args, writer, state_dict):
    # TODO list:
    # - do something about this being awful and hacky

    max_name_len = max([len(s) for s in state_dict.keys()]) + 4
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
        blacklist = [
            "time_embedding.",
            "add_embedding.",
            "time_in.",
            "txt_in.",
            "vector_in.",
            "img_in.",
            "guidance_in.",
            "final_layer.",
        ]

        if any([x in key for x in blacklist]) and ".weight" in key:
            data_qtype = gguf.GGMLQuantizationType.F32

        if n_dims == 1: 
            # one-dimensional tensors should be kept in F32
            # also speeds up inference due to not dequantizing
            data_qtype = gguf.GGMLQuantizationType.F32
        
        elif n_params <= 1024:
            # very small tensors
            data_qtype = gguf.GGMLQuantizationType.F32
        
        elif n_dims == 4:
            if min(data.shape[:2]) == 4: # output tensor
                data_qtype = gguf.GGMLQuantizationType.F16
            elif data_shape[-1] == 3: # 3x3 kernel
                data_qtype = gguf.GGMLQuantizationType.F16
            elif data_shape[-1] == 1: # 1x1 kernel
                #data = np.squeeze(data) # don't do this
                data_qtype = gguf.GGMLQuantizationType.F16

        try:
            data = gguf.quants.quantize(data, data_qtype)
        except gguf.QuantError as e:
            tqdm.write(f"falling back to F16: {e}")
            data_qtype = gguf.GGMLQuantizationType.F16
            data = gguf.quants.quantize(data, data_qtype)
        except AttributeError as e:
            tqdm.write(f"falling back to F16: {e}")
            data_qtype = gguf.GGMLQuantizationType.F16
            data = gguf.quants.quantize(data, data_qtype)

        new_name = key # do we need to rename?

        shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
        tqdm.write(f"{f'%-{max_name_len}s' % f'{new_name}'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

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
