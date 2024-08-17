# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import gguf
import torch
import numpy as np

import comfy.ops
from .dequant import dequantize_tensor

class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """
    def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape

    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = self.tensor_type
        new.tensor_shape = self.tensor_shape
        return new

    @property
    def shape(self):
        return self.tensor_shape

class GGMLLayer(torch.nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = None
        self.bias = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for k,v in state_dict.items():
            if k[len(prefix):] == "weight":
                self.weight = v
            elif k[len(prefix):] == "bias":
                self.bias = v
            else:
                missing_keys.append(k)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # This is a fake state dict for vram estimation
        if self.weight is not None:
            weight = torch.zeros_like(self.weight, device=torch.device("meta"))
            destination[f"{prefix}weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[f"{prefix}bias"] = bias
        return

        # This would return the actual state dict
        weight, bias = self.get_weights()
        if weight is not None:
            destination[f"{prefix}weight"] = weight
        if bias is not None:
            destination[f"{prefix}bias"] = weight

    def _apply(self, fn):
        if self.weight is not None:
            self.weight = fn(self.weight)
        if self.bias is not None:
            self.bias = fn(self.bias)
        super()._apply(fn)
        return self

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return
        weight = dequantize_tensor(tensor, dtype)
        return weight

    def get_weights(self, dtype=torch.float16):
        weight = self.get_weight(self.weight, dtype)
        bias = self.get_weight(self.bias, dtype)
        return (weight, bias)

class GGMLOps(comfy.ops.disable_weight_init):
    """
    Dequantize weights on the fly before doing the compute
    """
    class Linear(GGMLLayer):
        comfy_cast_weights = True

        def __init__(self, *args, device=None, dtype=None, **kwargs):
            super().__init__(device=device, dtype=dtype)

        def forward(self, x):
            # lowvram hack
            device = None
            if self.weight.device != x.device:
                device = self.weight.device
                self.to(x.device)

            weight, bias = self.get_weights(x.dtype)
            x = torch.nn.functional.linear(x, weight, bias)
            del weight, bias

            if device:
                self.to(device)
            return x