# GGML QuantizedTensor support for ComfyUI DynamicVRAM loading.
from dataclasses import dataclass

import gguf
import torch

from comfy_kitchen.tensor import (
    BaseLayoutParams,
    QuantizedLayout,
    QuantizedTensor,
    register_layout_class,
)

from .dequant import TORCH_COMPATIBLE_QTYPES, dequantize_functions


@dataclass(frozen=True)
class GGMLLayoutParams(BaseLayoutParams):
    tensor_type: int


class GGMLLayout(QuantizedLayout):
    Params = GGMLLayoutParams

    @classmethod
    def quantize(cls, tensor, **kwargs):
        raise NotImplementedError("Quantization to GGML format is not supported")

    @classmethod
    def dequantize(cls, qdata, params):
        qtype = gguf.GGMLQuantizationType(params.tensor_type)
        orig_shape = params.orig_shape

        if qtype in TORCH_COMPATIBLE_QTYPES:
            return qdata.reshape(orig_shape).to(params.orig_dtype)

        if qtype not in dequantize_functions:
            dequantized = gguf.quants.dequantize(qdata.cpu().numpy(), qtype)
            return torch.from_numpy(dequantized).reshape(orig_shape).to(
                device=qdata.device,
                dtype=params.orig_dtype,
            )

        _, type_size = gguf.GGML_QUANT_SIZES[qtype]
        raw = qdata.reshape(-1).view(torch.uint8)
        blocks = raw.reshape((raw.numel() // type_size, type_size))
        return dequantize_functions[qtype](blocks, *gguf.GGML_QUANT_SIZES[qtype], None).reshape(
            orig_shape
        ).to(params.orig_dtype)

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return (qtensor._qdata,)

    @classmethod
    def state_dict_tensors(cls, qdata, params):
        return {"weight": qdata}


register_layout_class("GGMLLayout", GGMLLayout)


def make_quantized(qdata, tensor_type, tensor_shape, orig_dtype=torch.float16):
    params = GGMLLayoutParams(
        scale=torch.ones((), dtype=torch.float32),
        orig_dtype=orig_dtype,
        orig_shape=tuple(tensor_shape),
        tensor_type=tensor_type.value if not isinstance(tensor_type, int) else tensor_type,
    )
    return QuantizedTensor(qdata, "GGMLLayout", params)
