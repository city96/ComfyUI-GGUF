"""
Convert a llama.cpp `qwen35` GGUF (e.g. unsloth/Qwen3.5-9B-GGUF) into a GGUF with ComfyUI key names.

The loader can read llama.cpp qwen35 files directly, but then it has to undo the llama.cpp
converter's changes on every load (on the CPU) and keeps `out_proj` and the norms as fp16/fp32
copies. This script does the same reversals once, offline:

  * norm.weight (except linear_attn.norm): w - 1
  * ssm_a = -exp(A_log): A_log = log(-a)
  * conv1d [C, K] -> [C, 1, K]
  * V heads grouped -> tiled undone: rows of in_proj_qkv/conv1d/in_proj_z/in_proj_a/in_proj_b/A_log/
    dt_bias are permuted directly on the quantized data (lossless, each row is quantized on its own);
    out_proj needs a column permutation, so it is dequantized, permuted and requantized to Q8_0.

All other tensors keep their original quantization. Text-only GGUFs have no vision tower, but
ComfyUI always builds it; pass --vision with a Qwen3.5 safetensors file (local path or http(s)
URL, e.g. Comfy-Org/MOSS-Image text_encoders/qwen3.5_9b_bf16.safetensors) to include it as
Q8_0/F16 instead of leaving it uninitialized. Only the vision tensors are read (HTTP range requests
for URLs).

usage: python convert_qwen35.py --src Qwen3.5-9B-Q6_K.gguf [--dst out.gguf] [--vision <safetensors>]
"""
import argparse
import json
import struct
import urllib.request

import numpy as np
import gguf
from gguf import GGMLQuantizationType as Q
from gguf import quants

# keep in sync with QWEN35_SD_MAP in ../loader.py
QWEN35_SD_MAP = {
    "blk.": "model.language_model.layers.",
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attention_layernorm",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_qkv": "linear_attn.in_proj_qkv",
    "attn_gate": "linear_attn.in_proj_z",
    "attn_q.": "self_attn.q_proj.",
    "attn_k.": "self_attn.k_proj.",
    "attn_v.": "self_attn.v_proj.",
    "attn_output": "self_attn.o_proj",
    "ssm_alpha": "linear_attn.in_proj_a",
    "ssm_beta": "linear_attn.in_proj_b",
    "ssm_out": "linear_attn.out_proj",
    "ssm_norm": "linear_attn.norm",
    "ssm_conv1d": "linear_attn.conv1d",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "token_embd": "model.language_model.embed_tokens",
    "output_norm": "model.language_model.norm",
    "output.weight": "lm_head.weight",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a llama.cpp qwen35 GGUF to ComfyUI key names.")
    parser.add_argument("--src", required=True, help="llama.cpp qwen35 GGUF")
    parser.add_argument("--dst", help="output GGUF (default: <src>-comfy.gguf)")
    parser.add_argument("--vision", help="Qwen3.5 safetensors (path or URL) to take model.visual.* from")
    args = parser.parse_args()
    args.dst = args.dst or args.src.removesuffix(".gguf") + "-comfy.gguf"
    return args


def comfy_name(name):
    for s, d in QWEN35_SD_MAP.items():
        name = name.replace(s, d)
    return name


def plain(tensor, shape):
    if tensor.tensor_type in (Q.F32, Q.F16):
        return np.array(tensor.data, dtype=np.float32).reshape(shape)
    return quants.dequantize(tensor.data, tensor.tensor_type).reshape(shape).astype(np.float32)


def read_range(src, start, end):
    """Bytes [start, end) of a local file or an http(s) URL."""
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()
    with open(src, "rb") as f:
        f.seek(start)
        return f.read(end - start)


def vision_tensors(src):
    n = struct.unpack("<Q", read_range(src, 0, 8))[0]
    header = json.loads(read_range(src, 8, 8 + n))
    header.pop("__metadata__", None)
    dtypes = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}
    for name, info in header.items():
        if not name.startswith("model.visual."):
            continue
        a, b = info["data_offsets"]
        raw = np.frombuffer(read_range(src, 8 + n + a, 8 + n + b), dtype=dtypes[info["dtype"]])
        x = (raw.astype(np.uint32) << 16).view(np.float32) if info["dtype"] == "BF16" else raw.astype(np.float32)
        x = x.reshape(info["shape"])
        if x.ndim == 2 and x.shape[1] % 32 == 0 and not name.endswith(("norm.weight", ".bias")):
            yield name, quants.quantize(np.ascontiguousarray(x), Q.Q8_0), Q.Q8_0
        elif x.size < (1 << 20) or name.endswith(("norm.weight", ".bias")):
            yield name, x, Q.F32
        else:
            yield name, x.astype(np.float16), Q.F16


def convert(args):
    reader = gguf.GGUFReader(args.src)
    arch = str(reader.fields["general.architecture"].parts[-1], encoding="utf-8")
    if arch != "qwen35":
        raise ValueError(f"Expected a qwen35 GGUF, got {arch!r}")
    num_k_heads = int(reader.fields["qwen35.ssm.group_count"].parts[-1][0])

    tensors = {comfy_name(t.name): (t, tuple(int(x) for x in reversed(t.shape))) for t in reader.tensors}
    out = {}  # name -> (array, type); quantized arrays are (rows, bytes per row)
    for name, (t, shape) in tensors.items():
        if name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            out[name] = (plain(t, shape) - 1.0, Q.F32)
        elif name.endswith("linear_attn.A_log"):
            out[name] = (np.log(-plain(t, shape)), Q.F32)
        elif name.endswith("linear_attn.conv1d.weight"):
            out[name] = (plain(t, shape)[:, None, :], Q.F32)
        elif t.tensor_type in (Q.F32, Q.F16):
            out[name] = (plain(t, shape), t.tensor_type)
        else:
            out[name] = (np.array(t.data), t.tensor_type)

    def perm(num_v_heads, d):
        return np.arange(num_v_heads * d).reshape(num_v_heads // num_k_heads, num_k_heads, d).transpose(1, 0, 2).reshape(-1)

    for key in [k for k in out if k.endswith("linear_attn.A_log")]:
        prefix = key[:-len("A_log")]
        num_v_heads = out[key][0].shape[0]
        head_v_dim = out[f"{prefix}norm.weight"][0].shape[0]
        v_perm = perm(num_v_heads, head_v_dim)
        for name in ("in_proj_qkv.weight", "conv1d.weight", "in_proj_z.weight"):
            data, qtype = out[prefix + name]
            offset = data.shape[0] - v_perm.shape[0]
            out[prefix + name] = (data[np.concatenate([np.arange(offset), offset + v_perm])], qtype)
        for name in ("in_proj_a.weight", "in_proj_b.weight", "A_log", "dt_bias"):
            data, qtype = out[prefix + name]
            out[prefix + name] = (data[perm(num_v_heads, 1)], qtype)
        t, shape = tensors[prefix + "out_proj.weight"]
        weight = plain(t, shape)[:, v_perm]
        out[prefix + "out_proj.weight"] = (quants.quantize(np.ascontiguousarray(weight), Q.Q8_0), Q.Q8_0)

    if args.vision:
        for name, data, qtype in vision_tensors(args.vision):
            out[name] = (data, qtype)

    writer = gguf.GGUFWriter(args.dst, "qwen35", use_temp_file=False)
    writer.add_name(f"{reader.fields['general.name'].parts[-1].tobytes().decode() if 'general.name' in reader.fields else 'Qwen3.5'} (ComfyUI key names)")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    for name, (data, qtype) in out.items():
        if qtype in (Q.F32, Q.F16):
            writer.add_tensor(name, np.ascontiguousarray(data.astype(np.float32 if qtype == Q.F32 else np.float16)), raw_dtype=qtype)
        else:
            writer.add_tensor(name, np.ascontiguousarray(data), raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"Wrote {len(out)} tensors to {args.dst}")


if __name__ == "__main__":
    convert(parse_args())
