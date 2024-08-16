# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import torch
import numpy as np
import gguf # This needs to be the llama.cpp one specifically!
import argparse
from tqdm import tqdm

from safetensors.torch import load_file

def parse_args():
    parser = argparse.ArgumentParser(description="Generate GGUF files from single SD ckpt")
    parser.add_argument("--src", required=True, help="Source model ckpt file.")
    parser.add_argument("--dst", help="Output  unet gguf file.")
    parser.add_argument("--qtype", default="F16", help="Quant type [default: f16]")
    args = parser.parse_args()

    if not os.path.isfile(args.src):
        parser.error("No input provided!")

    if args.dst is None:
        args.dst = os.path.splitext(args.src)[0] + f"_{args.qtype}.gguf"
        args.dst = os.path.basename(args.dst)

    if os.path.isfile(args.dst):
        input("Output exists enter to continue or ctrl+c to abort!")
    
    try:
        args.qtype = getattr(gguf.GGMLQuantizationType, args.qtype)
    except AttributeError:
        parser.error(f"Unknown quant type {args.qtype}")
    
    return args

def load_state_dict(path):
    if any(path.endswith(x) for x in [".ckpt", ".pt", ".bin", ".pth"]):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = state_dict.get("model", state_dict)
    else:
        state_dict = load_file(path)
    return state_dict

def load_model(args):
    state_dict = load_state_dict(args.src)

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

def handle_metadata(args, writer, state_dict):
    # TODO: actual metadata
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(args.qtype) # TODO: gguf.LlamaFileType

def handle_tensors(args, writer, state_dict):
    # TODO list:
    # - do something about this being awful and hacky

    max_name_len = max([len(s) for s in state_dict.keys()]) + 4
    for key, data in tqdm(state_dict.items()):
        if data.dtype == torch.bfloat16:
            data = data.to(torch.float32)
        data = data.numpy()

        old_dtype = data.dtype

        n_dims = len(data.shape)
        data_qtype = args.qtype
        data_shape = data.shape

        # get number of parameters (AKA elements) in this tensor
        n_params = 1
        for dim_size in data_shape:
            n_params *= dim_size

        fallback = gguf.GGMLQuantizationType.F16

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
            data_qtype = fallback

        if n_dims == 1: 
            # one-dimensional tensors should be kept in F32
            # also speeds up inference due to not dequantizing
            data_qtype = gguf.GGMLQuantizationType.F32
        
        elif n_params <= 1024:
            # very small tensors
            data_qtype = gguf.GGMLQuantizationType.F32
        
        elif n_dims == 4:
            if min(data.shape[:2]) == 4: # output tensor
                data_qtype = fallback
            elif data_shape[-1] == 3: # 3x3 kernel
                data_qtype = fallback
            elif data_shape[-1] == 1: # 1x1 kernel
                #data = np.squeeze(data) # don't do this
                data_qtype = fallback

        # TODO: find keys to keep in higher precision(s) / qtypes
        # if "time_emb_proj.weight" in key:
        #     data_qtype = gguf.GGMLQuantizationType.F16
        # if ".to_v.weight" in key or ".to_out" in key:
        #     data_qtype = gguf.GGMLQuantizationType.F16
        # if "ff.net" in key:
        #     data_qtype = gguf.GGMLQuantizationType.F16

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
        tqdm.write(f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

        writer.add_tensor(new_name, data, raw_dtype=data_qtype)

warning = """
######################################################
      The quantized file format needs more work.
Consider **not** uploading the resulting files for now
######################################################
"""

if __name__ == "__main__":
    args = parse_args()
    writer, state_dict = load_model(args)
    
    handle_metadata(args, writer, state_dict)
    handle_tensors(args, writer, state_dict)

    writer.write_header_to_file(path=(args.dst or "test.gguf"))
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    print(warning)
