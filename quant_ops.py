# GGML QuantizedTensor support for dynamic VRAM loading
import gguf
import torch
from dataclasses import dataclass

try:
    from comfy_kitchen.tensor import (
        QuantizedTensor,
        QuantizedLayout,
        BaseLayoutParams,
        register_layout_class,
    )
    _CK_AVAILABLE = True
except ImportError:
    _CK_AVAILABLE = False

    class QuantizedTensor:
        pass

    class QuantizedLayout:
        pass

    class BaseLayoutParams:
        pass

    def register_layout_class(name, cls):
        pass

from .dequant import dequantize_functions, TORCH_COMPATIBLE_QTYPES, is_quantized

COMPILED_DEQUANT_FUNCTIONS = {}
if hasattr(torch, 'compile'):
    for k, func in dequantize_functions.items():
        COMPILED_DEQUANT_FUNCTIONS[k] = torch.compile(func)
else:
    COMPILED_DEQUANT_FUNCTIONS = dequantize_functions.copy()
   
if _CK_AVAILABLE:
    @dataclass(frozen=True)
    class GGMLLayoutParams(BaseLayoutParams):
        tensor_type: int  # gguf.GGMLQuantizationType stored as int

    class GGMLLayout(QuantizedLayout):
        Params = GGMLLayoutParams

        @classmethod
        def quantize(cls, tensor, **kwargs):
            raise NotImplementedError("Quantization to GGML format is not supported")

        @classmethod
        def dequantize(cls, qdata, params):
            qtype = gguf.GGMLQuantizationType(params.tensor_type)
            oshape = params.orig_shape

            if qtype in TORCH_COMPATIBLE_QTYPES:
                return qdata.reshape(oshape).to(params.orig_dtype)

            if qtype not in dequantize_functions:
                from tqdm import tqdm
                tqdm.write(f"Falling back to numpy dequant for qtype: {qtype.name}")
                new = gguf.quants.dequantize(qdata.cpu().numpy(), qtype)
                return torch.from_numpy(new).reshape(oshape).to(device=qdata.device, dtype=params.orig_dtype)

            block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
            raw = qdata.reshape(-1).view(torch.uint8)
            n_blocks = raw.numel() // type_size
            blocks = raw.reshape((n_blocks, type_size))

            fn = COMPILED_DEQUANT_FUNCTIONS[qtype]
            try:
                blocks = fn(blocks, block_size, type_size, params.orig_dtype)
            except Exception:
                fn = COMPILED_DEQUANT_FUNCTIONS[qtype] = dequantize_functions[qtype]
                blocks = fn(blocks, block_size, type_size, params.orig_dtype)
            return blocks.reshape(oshape).to(params.orig_dtype)

        @classmethod
        def get_plain_tensors(cls, qtensor):
            return (qtensor._qdata,)

        @classmethod
        def state_dict_tensors(cls, qdata, params):
            return {"weight": qdata}

    register_layout_class("GGMLLayout", GGMLLayout)

def make_quantized(qdata, tensor_type, tensor_shape, orig_dtype=torch.float16):
    """Construct a GGML QuantizedTensor from raw packed data."""
    params = GGMLLayoutParams(
        scale=torch.ones((), dtype=torch.float32),
        orig_dtype=orig_dtype,
        orig_shape=tuple(tensor_shape),
        tensor_type=tensor_type.value if not isinstance(tensor_type, int) else tensor_type,
    )
    return QuantizedTensor(qdata, "GGMLLayout", params)
