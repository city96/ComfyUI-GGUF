# GGML QuantizedTensor support for ComfyUI DynamicVRAM loading.
from dataclasses import dataclass, replace

import gguf
import torch

from comfy_kitchen.tensor import (
    BaseLayoutParams,
    QuantizedLayout,
    QuantizedTensor,
    register_layout_class,
)

try:
    from comfy_kitchen.tensor import register_layout_op
except ImportError:

    def register_layout_op(*_args, **_kwargs):
        return lambda function: function


from .dequant import TORCH_COMPATIBLE_QTYPES, dequantize_functions
from .kquant_backend import try_kquant_linear


@dataclass(frozen=True)
class GGMLLayoutParams(BaseLayoutParams):
    tensor_type: int
    transposed: bool = False


class GGMLLayout(QuantizedLayout):
    Params = GGMLLayoutParams

    @classmethod
    def quantize(cls, tensor, **kwargs):
        raise NotImplementedError("Quantization to GGML format is not supported")

    @classmethod
    def dequantize(cls, qdata, params):
        qtype = gguf.GGMLQuantizationType(params.tensor_type)
        orig_shape = (
            tuple(reversed(params.orig_shape))
            if params.transposed
            else params.orig_shape
        )

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


def _is_ggml_qtensor(tensor):
    return (
        isinstance(tensor, QuantizedTensor)
        and tensor._layout_cls == "GGMLLayout"
    )


def _physical_weight_shape(weight):
    shape = tuple(weight._params.orig_shape)
    return tuple(reversed(shape)) if weight._params.transposed else shape


def _try_native_linear(input_tensor, weight, bias):
    if (
        not _is_ggml_qtensor(weight)
        or weight._params.transposed
        or weight._params.orig_dtype != input_tensor.dtype
    ):
        return None
    qtype = gguf.GGMLQuantizationType(weight._params.tensor_type)
    return try_kquant_linear(
        input_tensor,
        weight._qdata,
        qtype,
        _physical_weight_shape(weight),
        bias,
        dequant_dtype=None,
    )


@register_layout_op(torch.ops.aten.t.default, GGMLLayout)
def _handle_ggml_t(qtensor, args, kwargs):
    input_tensor = args[0]
    if not _is_ggml_qtensor(input_tensor) or len(input_tensor.shape) != 2:
        return torch.ops.aten.t.default(input_tensor.dequantize())
    params = replace(
        input_tensor._params,
        orig_shape=tuple(reversed(input_tensor._params.orig_shape)),
        transposed=not input_tensor._params.transposed,
    )
    return QuantizedTensor(input_tensor._qdata, "GGMLLayout", params)


@register_layout_op(torch.ops.aten.linear.default, GGMLLayout)
def _handle_ggml_linear(qtensor, args, kwargs):
    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if not _is_ggml_qtensor(weight):
        if isinstance(weight, QuantizedTensor):
            weight = weight.dequantize()
        return torch.nn.functional.linear(input_tensor, weight, bias)
    output = _try_native_linear(input_tensor, weight, bias)
    if output is not None:
        return output
    return torch.nn.functional.linear(input_tensor, weight.dequantize(), bias)


def _resolve_transposed_rhs(rhs):
    if not _is_ggml_qtensor(rhs) or not rhs._params.transposed:
        return None
    params = replace(
        rhs._params,
        orig_shape=tuple(reversed(rhs._params.orig_shape)),
        transposed=False,
    )
    return QuantizedTensor(rhs._qdata, "GGMLLayout", params)


@register_layout_op(torch.ops.aten.mm.default, GGMLLayout)
def _handle_ggml_mm(qtensor, args, kwargs):
    left, right = args[0], args[1]
    if isinstance(left, QuantizedTensor):
        left = left.dequantize()
    weight = _resolve_transposed_rhs(right)
    if weight is not None:
        output = _try_native_linear(left, weight, None)
        if output is not None:
            return output
    if isinstance(right, QuantizedTensor):
        right = right.dequantize()
    return torch.mm(left, right)


@register_layout_op(torch.ops.aten.addmm.default, GGMLLayout)
def _handle_ggml_addmm(qtensor, args, kwargs):
    bias, left, right = args[0], args[1], args[2]
    if isinstance(left, QuantizedTensor):
        left = left.dequantize()
    weight = _resolve_transposed_rhs(right)
    if not kwargs and weight is not None:
        output = _try_native_linear(left, weight, bias)
        if output is not None:
            return output
    if isinstance(right, QuantizedTensor):
        right = right.dequantize()
    return torch.addmm(bias, left, right, **kwargs)


def make_quantized(qdata, tensor_type, tensor_shape, orig_dtype=torch.float16):
    params = GGMLLayoutParams(
        scale=torch.ones((), dtype=torch.float32),
        orig_dtype=orig_dtype,
        orig_shape=tuple(tensor_shape),
        tensor_type=tensor_type.value if not isinstance(tensor_type, int) else tensor_type,
    )
    return QuantizedTensor(qdata, "GGMLLayout", params)
