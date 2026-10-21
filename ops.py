# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import gguf
import torch
import logging
import os
import sys
import threading
import weakref

import comfy.ops
import comfy.lora
import comfy.model_management
from .dequant import dequantize_tensor, is_quantized

def chained_hasattr(obj, chained_attr):
    probe = obj
    for attr in chained_attr.split('.'):
        if hasattr(probe, attr):
            probe = getattr(probe, attr)
        else:
            return False
    return True

# A bakcward and forward compatible way to get `torch.compiler.disable`.
def get_torch_compiler_disable_decorator():
    def dummy_decorator(*args, **kwargs):
        def noop(x):
            return x
        return noop

    from packaging import version

    if not chained_hasattr(torch, "compiler.disable"):
        logging.info("ComfyUI-GGUF: Torch too old for torch.compile - bypassing")
        return dummy_decorator # torch too old
    elif version.parse(torch.__version__) >= version.parse("2.8"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile")
        return dummy_decorator # torch compile works
    if chained_hasattr(torch, "_dynamo.config.nontraceable_tensor_subclasses"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile (nightly)")
        return dummy_decorator # torch compile works, nightly before 2.8 release
    else:
        logging.info("ComfyUI-GGUF: Partial torch compile only, consider updating pytorch")
        return torch.compiler.disable

torch_compiler_disable = get_torch_compiler_disable_decorator()

class _GGUFStagingBuffers:
    def __init__(self, device):
        self.lock = threading.Lock()
        self.buffers = [torch.empty(8 * 1024 * 1024, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        self.views = [memoryview(buffer.numpy()) for buffer in self.buffers]
        self.events = [torch.cuda.Event() for _ in range(2)]
        self.stream = torch.cuda.Stream(device=device)

GGUF_STREAM_TO_GPU = os.environ.get("COMFYUI_GGUF_STREAM_TO_GPU") == "1"
MIN_STREAM_BYTES = 8 * 1024 * 1024
_gguf_stagers = {}
_gguf_stagers_lock = threading.Lock()

def _gguf_stager(device):
    with _gguf_stagers_lock:
        if device not in _gguf_stagers:
            try:
                _gguf_stagers[device] = _GGUFStagingBuffers(device)
            except RuntimeError as error:
                logging.warning("GGUF pinned staging unavailable on %s: %s", device, error)
                _gguf_stagers[device] = None
        return _gguf_stagers[device]

class GGUFFileSource:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY)
        self.mapping_path = f"/proc/self/fd/{self.fd}" if sys.platform.startswith("linux") else path
        self._close = weakref.finalize(self, os.close, self.fd)

    def close(self):
        self._close()

class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """
    streamed_bytes = 0
    def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches

    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def set_file_source(self, source, offset, size, readonly):
        self.file_source = (source, offset, size)
        self.file_source_data_ptr = self.data_ptr()
        self.file_source_readonly = readonly
        self.file_source_version = None if self.is_inference() else self._version

    def current_file_source(self):
        source = getattr(self, "file_source", None)
        if source is None or self.data_ptr() != getattr(self, "file_source_data_ptr", None):
            return None
        version = getattr(self, "file_source_version", None)
        if version is None:
            return source if self.is_inference() and self.file_source_readonly else None
        return source if not self.is_inference() and self._version == version else None

    def to(self, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        target = torch.device(target) if isinstance(target, (str, torch.device)) else None
        dtype = kwargs.get("dtype", args[1] if len(args) > 1 and isinstance(args[1], torch.dtype) else None)
        source = self.current_file_source() if GGUF_STREAM_TO_GPU else None
        if source and source[2] >= MIN_STREAM_BYTES and hasattr(os, "preadv") and self.device.type == "cpu" \
                and target is not None and target.type == "cuda" \
                and (dtype is None or dtype == self.dtype) and kwargs.get("memory_format") is None:
            stager = _gguf_stager(target)
            new = self._stream_to_device(target, source, stager) if stager is not None else super().to(*args, **kwargs)
        else:
            new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def _stream_to_device(self, device, source, stager):
        file, offset, size = source
        chunk_size = 8 * 1024 * 1024
        tensor = torch.empty(self.size(), dtype=self.dtype, device=device)
        destination = tensor.view(torch.uint8).flatten()
        if destination.numel() != size:
            raise ValueError("GGUF tensor byte count differs from file metadata")
        with stager.lock:
            stager.stream.wait_stream(torch.cuda.current_stream(device))
            position = 0
            used = [False, False]
            while position < size:
                slot = (position // chunk_size) % 2
                if used[slot]:
                    stager.events[slot].synchronize()
                count = min(chunk_size, size - position)
                filled = 0
                while filled < count:
                    received = os.preadv(file.fd, [stager.views[slot][filled:count]], offset + position + filled)
                    if received == 0:
                        raise EOFError("GGUF tensor ended during GPU upload")
                    filled += received
                with torch.cuda.stream(stager.stream):
                    destination[position:position + count].copy_(stager.buffers[slot][:count], non_blocking=True)
                    stager.events[slot].record(stager.stream)
                used[slot] = True
                position += count
            stager.stream.synchronize()
        previous_gib = GGMLTensor.streamed_bytes // (1 << 30)
        GGMLTensor.streamed_bytes += size
        if GGMLTensor.streamed_bytes // (1 << 30) > previous_gib:
            logging.info("GGUF streamed %.1f GiB to GPU", GGMLTensor.streamed_bytes / (1 << 30))
        return GGMLTensor(tensor, tensor_type=self.tensor_type, tensor_shape=self.tensor_shape)

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            logging.warning(f"ignoring 'copy_' on tensor: {e}")

    def new_empty(self, size, *args, **kwargs):
        # Intel Arc fix, ref#50
        new_tensor = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(
                new_tensor,
                tensor_type = getattr(self, "tensor_type", None),
                tensor_shape = size,
                patches = getattr(self, "patches", []).copy()
        )

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
    largest_layer = False
    torch_compatible_tensor_types = {None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = state_dict.get(f"{prefix}weight"), state_dict.get(f"{prefix}bias")
        # NOTE: using modified load for linear due to not initializing on creation, see GGMLOps todo
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(self, torch.nn.Linear):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # Not strictly required, but fixes embedding shape mismatch. Threshold set in loader.py
        if isinstance(self, torch.nn.Embedding) and self.weight.shape[0] >= (64 * 1024):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        prefix_len = len(prefix)
        for k,v in state_dict.items():
            if k[prefix_len:] == "weight":
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                unexpected_keys.append(k)

        # For Linear layer with missing weight
        if self.weight is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix+"weight")

        # for vram estimation (TODO: less fragile logic?)
        if getattr(self.weight, "is_largest_weight", False):
            self.largest_layer = True

    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        # This is a fake state dict for vram estimation
        weight = torch.zeros_like(self.weight, device=torch.device("meta"))
        destination[prefix + "weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[prefix + "bias"] = bias

        # Take into account space required for dequantizing the largest tensor
        if self.largest_layer:
            shape = getattr(self.weight, "tensor_shape", self.weight.shape)
            dtype = self.dequant_dtype if self.dequant_dtype and self.dequant_dtype != "target" else torch.float16
            temp = torch.empty(*shape, device=torch.device("meta"), dtype=dtype)
            destination[prefix + "temp.weight"] = temp

        return
        # This would return the dequantized state dict
        destination[prefix + "weight"] = self.get_weight(self.weight)
        if bias is not None:
            destination[prefix + "bias"] = self.get_weight(self.bias)

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return

        # consolidate and load patches to GPU in async
        patch_list = []
        device = tensor.device
        for patches, key in getattr(tensor, "patches", []):
            patch_list += move_patch_to_device(patches, device)

        # dequantize tensor while patches load
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)

        # prevent propagating custom tensor class
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)

        # apply patches
        if len(patch_list) > 0:
            if self.patch_dtype is None:
                weight = comfy.lora.calculate_weight(patch_list, weight, key)
            else:
                # for testing, may degrade image quality
                patch_dtype = dtype if self.patch_dtype == "target" else self.patch_dtype
                weight = comfy.lora.calculate_weight(patch_list, weight, key, patch_dtype)
        return weight

    @torch_compiler_disable()
    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        non_blocking = comfy.model_management.device_supports_non_blocking(device)
        if s.bias is not None:
            bias = s.get_weight(s.bias.to(device), dtype)
            bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)

        weight = s.get_weight(s.weight.to(device), dtype)
        weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
        return weight, bias

    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)
        else:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)

        # non-ggml forward might still propagate custom tensor class
        if isinstance(out, GGMLTensor):
            out = torch.Tensor(out)
        return out

    def forward_ggml_cast_weights(self, input):
        raise NotImplementedError

class GGMLOps(comfy.ops.manual_cast):
    """
    Dequantize weights on the fly before doing the compute
    """
    class Linear(GGMLLayer, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            # TODO: better workaround for reserved memory spike on windows
            # Issue is with `torch.empty` still reserving the full memory for the layer
            # Windows doesn't over-commit memory so without this 24GB+ of pagefile is used
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(GGMLLayer, comfy.ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Embedding(GGMLLayer, comfy.ops.manual_cast.Embedding):
        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
                out_dtype = None
            weight, _bias = self.cast_bias_weight(self, device=input.device, dtype=out_dtype)
            return torch.nn.functional.embedding(
                input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
            ).to(dtype=output_dtype)

    class LayerNorm(GGMLLayer, comfy.ops.manual_cast.LayerNorm):
        def forward_ggml_cast_weights(self, input):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

    class GroupNorm(GGMLLayer, comfy.ops.manual_cast.GroupNorm):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item
