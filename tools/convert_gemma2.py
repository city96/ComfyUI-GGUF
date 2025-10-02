import os
import argparse
import logging
from safetensors.torch import load_file
import torch
import gguf
from tqdm import tqdm

KEY_MAP = {
    # embedding
    "model.embed_tokens.weight": "token_embd.weight",
    # norm
    "model.norm.weight": "output_norm.weight",
    # spiece
    "spiece_model": "tokenizer.ggml.spiece_model_raw",
}

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
    parser = argparse.ArgumentParser(description="Convert Gemma2 safetensors to GGUF (保留全部精度和元数据)")
    parser.add_argument("--src", required=True, help="源 safetensors 文件")
    parser.add_argument("--dst", help="输出 GGUF 文件")
    args = parser.parse_args()
    if not os.path.isfile(args.src):
        parser.error("输入文件不存在！")
    return args


def map_key(key):
    if key in KEY_MAP:
        return KEY_MAP[key]
    import re
    m = re.match(r"model.layers.(\d+)\.(.+)", key)
    if m:
        layer_idx, subkey = m.groups()
        if subkey in LAYER_KEY_MAP:
            return f"blk.{layer_idx}.{LAYER_KEY_MAP[subkey]}"
    return key


def main():
    args = parse_args()
    state_dict = load_file(args.src)
    dtypes = [v.dtype for v in state_dict.values() if hasattr(v, 'dtype')]
    main_dtype = max(set(dtypes), key=dtypes.count) if dtypes else torch.float16
    if main_dtype == torch.float32:
        ftype_name = "F32"
        ftype_gguf = gguf.GGMLQuantizationType.F32
    elif main_dtype == torch.bfloat16:
        ftype_name = "BF16"
        ftype_gguf = gguf.GGMLQuantizationType.BF16
    else:
        ftype_name = "F16"
        ftype_gguf = gguf.GGMLQuantizationType.F16
    dst = args.dst or f"{os.path.splitext(args.src)[0]}-{ftype_name}.gguf"
    if os.path.isfile(dst):
        input(f"输出文件 {dst} 已存在，按回车覆盖或 Ctrl+C 取消...")
    writer = gguf.GGUFWriter(path=None, arch="gemma2")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    for key, value in tqdm(state_dict.items()):
        new_key = map_key(key)
        if key == "spiece_model":
            arr = value.cpu().numpy().astype("int8")
            writer.add_tensor(new_key, arr, raw_dtype=gguf.GGMLQuantizationType.I8)
            tqdm.write(f"{key} -> {new_key} (spiece_model, {arr.shape} bytes, int8)")
            continue
        if not hasattr(value, 'dtype'):
            tqdm.write(f"跳过非张量: {key}")
            continue
        arr = value.cpu().numpy()
        # norm 层全部 F32，embedding/attn/mlp 优先 F16
        norm_keys = [
            "attn_norm.weight", "post_attention_norm.weight", "post_ffw_norm.weight", "ffn_norm.weight", "output_norm.weight"
        ]
        # embedding key
        emb_keys = ["token_embd.weight"]
        is_norm = any(new_key.endswith(nk) for nk in norm_keys)
        is_emb = any(new_key == ek for ek in emb_keys)
        # norm 层只有原始为 float32/bfloat16 时才保留 F32，否则保持原始 dtype
        if is_norm:
            if value.dtype == torch.float32 or value.dtype == torch.bfloat16:
                qtype = gguf.GGMLQuantizationType.F32
            elif value.dtype == torch.float16:
                qtype = gguf.GGMLQuantizationType.F16
            else:
                qtype = gguf.GGMLQuantizationType.F16
        elif is_emb:
            qtype = gguf.GGMLQuantizationType.F16
        elif value.dtype == torch.bfloat16:
            qtype = gguf.GGMLQuantizationType.BF16
        else:
            qtype = gguf.GGMLQuantizationType.F16
        writer.add_tensor(new_key, gguf.quants.quantize(arr, qtype), raw_dtype=qtype)
        tqdm.write(f"{key} -> {new_key}, {value.dtype} -> {qtype.name}, shape={arr.shape}")
    writer.write_header_to_file(path=dst)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"转换完成: {dst}")

if __name__ == "__main__":
    main()

