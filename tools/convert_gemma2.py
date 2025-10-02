import os
import argparse
import logging
from safetensors.torch import load_file
import torch
import gguf
from tqdm import tqdm

# Gemma2 key mapping
KEY_MAP = {
    # embedding
    "model.embed_tokens.weight": "token_embd.weight",
    # norm
    "model.norm.weight": "output_norm.weight",
    # spiece
    "spiece_model": "tokenizer.ggml.spiece_model_raw",
}

# Layer parameter mapping
LAYER_KEY_MAP = {
    # LayerNorm
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
    "pre_feedforward_layernorm.weight": "ffn_norm.weight",
    # MLP
    "mlp.down_proj.weight": "ffn_down.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    # Attention
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Gemma2 safetensors to GGUF with precision preservation")
    parser.add_argument("--src", required=True, help="Source safetensors file")
    parser.add_argument("--dst", help="Output GGUF file")
    parser.add_argument("--quantize", "--quant", "-q", 
                        choices=["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k"],
                        help="Quantization type")
    args = parser.parse_args()
    if not os.path.isfile(args.src):
        parser.error("Input file does not exist!")
    return args


def map_key(key):
    # Direct mapping
    if key in KEY_MAP:
        return KEY_MAP[key]
    # Layer parameter mapping
    import re
    m = re.match(r"model.layers.(\d+)\.(.+)", key)
    if m:
        layer_idx, subkey = m.groups()
        if subkey in LAYER_KEY_MAP:
            return f"blk.{layer_idx}.{LAYER_KEY_MAP[subkey]}"
    return key  # Keep others as-is


def get_quantization_type(quant_str):
    quant_map = {
        "f32": gguf.GGMLQuantizationType.F32,
        "f16": gguf.GGMLQuantizationType.F16,
        "bf16": gguf.GGMLQuantizationType.BF16,
        "q8_0": gguf.GGMLQuantizationType.Q8_0,
        "q4_0": gguf.GGMLQuantizationType.Q4_0,
        "q4_1": gguf.GGMLQuantizationType.Q4_1,
        "q5_0": gguf.GGMLQuantizationType.Q5_0,
        "q5_1": gguf.GGMLQuantizationType.Q5_1,
        "q2_k": gguf.GGMLQuantizationType.Q2_K,
        "q3_k": gguf.GGMLQuantizationType.Q3_K,
        "q4_k": gguf.GGMLQuantizationType.Q4_K,
        "q5_k": gguf.GGMLQuantizationType.Q5_K,
        "q6_k": gguf.GGMLQuantizationType.Q6_K,
    }
    return quant_map.get(quant_str.lower())


def should_quantize_tensor(key, quant_type):
    """Determine if a tensor should be quantized
    Rules:
    - token_embd (embedding) kept at F16 (quantization severely impacts quality)
    - norm layers kept at F32 (quantization affects stability)
    - other weights (attn/mlp) use target quantization
    """
    # Embedding always kept at F16
    if key == "token_embd.weight":
        return False, gguf.GGMLQuantizationType.F16
    
    # Norm layers kept at F32
    norm_suffixes = [
        "attn_norm.weight", 
        "post_attention_norm.weight", 
        "post_ffw_norm.weight", 
        "ffn_norm.weight", 
        "output_norm.weight"
    ]
    if any(key.endswith(suffix) for suffix in norm_suffixes):
        return False, gguf.GGMLQuantizationType.F32
    
    # Other layers (attn/mlp) use target quantization
    return True, quant_type


def main():
    args = parse_args()
    state_dict = load_file(args.src)

    if args.quantize:
        quant_type = get_quantization_type(args.quantize)
        ftype_name = args.quantize.upper()
    else:
        dtypes = [v.dtype for v in state_dict.values() if hasattr(v, 'dtype')]
        main_dtype = max(set(dtypes), key=dtypes.count) if dtypes else torch.float16
        if main_dtype == torch.float32:
            ftype_name = "F32"
            quant_type = gguf.GGMLQuantizationType.F32
        elif main_dtype == torch.bfloat16:
            ftype_name = "BF16"
            quant_type = gguf.GGMLQuantizationType.BF16
        else:
            ftype_name = "F16"
            quant_type = gguf.GGMLQuantizationType.F16
    
    dst = args.dst or f"{os.path.splitext(args.src)[0]}-{ftype_name}.gguf"
    if os.path.isfile(dst):
        input(f"Output file {dst} exists, press Enter to overwrite or Ctrl+C to cancel...")
    
    writer = gguf.GGUFWriter(path=None, arch="gemma2")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    
    print(f"Target quantization: {ftype_name}")
    print(f"Output file: {dst}")
    
    for key, value in tqdm(state_dict.items(), desc="Converting"):
        new_key = map_key(key)
        
        # Special handling for spiece_model
        if key == "spiece_model":
            arr = value.cpu().numpy().astype("int8")
            writer.add_tensor(new_key, arr, raw_dtype=gguf.GGMLQuantizationType.I8)
            tqdm.write(f"{key} -> {new_key} (spiece_model, {arr.shape[0]} bytes, I8)")
            continue
        
        if not hasattr(value, 'dtype'):
            tqdm.write(f"Skipping non-tensor: {key}")
            continue
        
        arr = value.cpu().numpy()
        
        # Determine if quantization needed + get target precision
        should_quant, target_qtype = should_quantize_tensor(new_key, quant_type)
        
        # Apply quantization or keep original precision
        if should_quant and target_qtype not in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16]:
            quantized_arr = gguf.quants.quantize(arr, target_qtype)
            writer.add_tensor(new_key, quantized_arr, raw_dtype=target_qtype)
            tqdm.write(f"{key} -> {new_key}, {value.dtype} -> {target_qtype.name}, shape={arr.shape}")
        else:
            if target_qtype == gguf.GGMLQuantizationType.F32:
                arr = arr.astype('float32')
            elif target_qtype == gguf.GGMLQuantizationType.BF16:
                # BF16 requires special handling
                pass  # gguf.quants.quantize handles this
            else:  # F16
                arr = arr.astype('float16')
            
            quantized_arr = gguf.quants.quantize(arr, target_qtype)
            writer.add_tensor(new_key, quantized_arr, raw_dtype=target_qtype)
            tqdm.write(f"{key} -> {new_key}, {value.dtype} -> {target_qtype.name}, shape={arr.shape}")
    
    print("Writing GGUF file...")
    writer.write_header_to_file(path=dst)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"Conversion complete: {dst}")
    print(f"Quantization type: {ftype_name}")
    print(f"File size: {os.path.getsize(dst) / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()
