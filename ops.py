# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import torch
from functools import partial

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
        self.patches = patches

    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            print(f"ignoring 'copy_' on tensor: {e}")

    def __deepcopy__(self, *args, **kwargs):
        # Intel Arc fix, ref#50
        new = super().__deepcopy__(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape

class GGMLLayer(torch.nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """
    comfy_cast_weights = True
    dequant_dtype = None
    patch_dtype = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = GGMLTensor(1, tensor_type=None, tensor_shape=None)
        self.bias = None

    def _forward_operation(self, x, weight, bias):
        raise NotImplementedError

    def forward(self, x):
        # lowvram hack
        device = None
        if self.weight.device != x.device:
            device = self.weight.device
            self.to(x.device)

        weight, bias = self.get_weights(x.dtype)
        x = self._forward_operation(x, weight, bias=bias)
        del weight, bias

        if device:
            self.to(device)
        return x

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
            try:
                self.weight = fn(self.weight)
            except TypeError:
                # TODO: Figure out why this happens
                pass
        if self.bias is not None:
            self.bias = fn(self.bias)
        super()._apply(fn)
        return self

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return

        # consolidate and load patches to GPU in async
        patch_list = []
        device = tensor.device
        t_move = lambda x: x.to(device) if torch.is_tensor(x) else x
        for function, patches, key in getattr(tensor, "patches", []):
            patch_list += move_patch_to_cuda(patches, device)

        # dequantize tensor while patches load
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)

        # apply patches
        if patch_list:
            if self.patch_dtype is None:
                weight = function(patch_list, weight, key)
            else:
                # for testing, may degrade image quality
                patch_dtype = dtype if self.patch_dtype == "target" else self.patch_dtype
                weight = function(patch_list, weight, key, patch_dtype)
        return weight

    def get_weights(self, dtype=torch.float16):
        weight = self.get_weight(self.weight, dtype)
        bias = self.get_weight(self.bias, dtype)
        return (weight, bias)


class GGMLOps(comfy.ops.manual_cast):
    """
    Dequantize weights on the fly before doing the compute
    """
    class Linear(GGMLLayer):
        _forward_operation = staticmethod(torch.nn.functional.linear)

    class Conv2d(GGMLLayer):
        def __init__(self, *args, device=None, dtype=None, **kwargs):
            super().__init__()
            _ = kwargs.pop("kernel_size", None)
            self._forward_operation = partial(staticmethod(torch.nn.functional.conv2d), **kwargs)


def move_patch_to_cuda(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_cuda(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_cuda(x, device) for x in item]
    else:
        return item
