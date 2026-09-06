# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import copy
import gguf
import json
import torch
import logging
import os
import time
from numbers import Real

import comfy.ops
import comfy.lora
import comfy.model_management
from .dequant import dequantize_tensor, is_quantized


_PERF_LOG_ENV = "COMFYUI_GGUF_PERF_LOG"
_INT4_LORA_OFFLOAD_ENV = "COMFYUI_GGUF_INT4_LORA_OFFLOAD"


def int4_lora_offload_enabled():
    return os.environ.get(_INT4_LORA_OFFLOAD_ENV, "").strip().lower() in {
        "true",
        "1",
        "yes",
        "on",
    }


def log_cuda_oom_loaded_models(context):
    """Log ComfyUI's managed model inventory before propagating a CUDA OOM."""
    loaded_models = getattr(comfy.model_management, "current_loaded_models", ())
    logging.error("ComfyUI-GGUF CUDA OOM while %s. Loaded models:", context)
    if not loaded_models:
        logging.error("  (none)")
        return

    for loaded_model in loaded_models:
        patcher = getattr(loaded_model, "model", None)
        model = getattr(patcher, "model", None)
        model_class = type(model).__name__ if model is not None else "<unloaded>"
        model_name = getattr(model, "name", None) or getattr(patcher, "model_name", None)
        identifier = (
            f"{model_class} ({model_name})"
            if isinstance(model_name, str) and model_name
            else model_class
        )

        size_bytes = None
        model_memory = getattr(loaded_model, "model_memory", None)
        if callable(model_memory):
            try:
                size_bytes = model_memory()
            except (AttributeError, TypeError):
                pass
        if not isinstance(size_bytes, Real):
            size_bytes = getattr(patcher, "size", None)

        if isinstance(size_bytes, Real) and size_bytes >= 0:
            logging.error("  %s: %.1f MiB", identifier, size_bytes / 1024**2)
        else:
            logging.error("  %s: size unavailable", identifier)


def _configure_perf_logger():
    log_path = os.environ.get(_PERF_LOG_ENV, "").strip()
    if not log_path:
        return None
    if log_path.strip().lower() in {"1", "true", "yes", "on"}:
        log_path = "comfyui-gguf-performance.log"

    logger = logging.getLogger("comfyui_gguf.performance")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        handler = logging.FileHandler(log_path, encoding="utf-8")
    except (OSError, ValueError) as error:
        logging.getLogger(__name__).warning(
            "ComfyUI-GGUF: unable to open performance log %r: %s", log_path, error
        )
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    handler._comfyui_gguf_perf = True
    logger.addHandler(handler)
    logger.info("performance logging enabled path=%s", log_path)
    return logger


_PERF_LOGGER = _configure_perf_logger()


def _perf_sync(device):
    if device is not None and getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _perf_forward(mode, layer, input_tensor, operation):
    """Run and optionally time one quantized Linear forward.

    CUDA synchronization is deliberately limited to the opt-in diagnostic path;
    normal inference does not pay for timing or synchronization overhead.
    """
    if _PERF_LOGGER is None:
        return operation()

    device = getattr(input_tensor, "device", None)
    _perf_sync(device)
    started = time.perf_counter()
    call_number = getattr(layer, "_gguf_perf_calls", 0) + 1
    layer._gguf_perf_calls = call_number
    cache_before = getattr(layer, "_quantized_weight", None) is not None
    fused_cache_before = getattr(layer, "_fused_weight", None) is not None
    try:
        result = operation()
        _perf_sync(device)
    except BaseException:
        _perf_sync(device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        _PERF_LOGGER.exception(
            "forward mode=%s call=%d in_features=%s out_features=%s "
            "input_shape=%s device=%s elapsed_ms=%.3f failed=true",
            mode,
            call_number,
            getattr(layer, "in_features", None),
            getattr(layer, "out_features", None),
            tuple(getattr(input_tensor, "shape", ())),
            device,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    _PERF_LOGGER.info(
        "forward mode=%s call=%d in_features=%s out_features=%s "
        "input_shape=%s device=%s elapsed_ms=%.3f cache_before=%s cache_after=%s "
        "fused_cache_before=%s fused_cache_after=%s patched=%s",
        mode,
        call_number,
        getattr(layer, "in_features", None),
        getattr(layer, "out_features", None),
        tuple(getattr(input_tensor, "shape", ())),
        device,
        elapsed_ms,
        cache_before,
        getattr(layer, "_quantized_weight", None) is not None,
        fused_cache_before,
        getattr(layer, "_fused_weight", None) is not None,
        bool(getattr(layer, "weight_function", ())) or bool(getattr(layer, "bias_function", ())),
    )
    return result

def _build_regular_hadamard(size, dtype=torch.float32, device="cpu"):
    """Build a normalized regular (Sylvester) Hadamard of a power-of-4 size."""
    if size < 4 or (size & (size - 1)) != 0:
        import math
        if not math.log(size, 4).is_integer():
            raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4
    return h / (size ** 0.5)

def _valid_compute_dtype(dtype):
    return dtype in {torch.float16, torch.bfloat16, torch.float32, torch.float64}

def _infer_compute_dtype(tensor_type, fallback=None):
    if _valid_compute_dtype(fallback):
        return fallback
    if tensor_type == gguf.GGMLQuantizationType.BF16:
        return torch.bfloat16
    if tensor_type == gguf.GGMLQuantizationType.F32:
        return torch.float32
    return torch.float16

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

class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """
    def __init__(self, *args, tensor_type, tensor_shape, patches=[], compute_dtype=None, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches
        self.compute_dtype = compute_dtype

    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], compute_dtype=None, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        new.compute_dtype = getattr(self, "compute_dtype", None)
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
            logging.warning(f"ignoring 'copy_' on tensor: {e}")

    def new_empty(self, size, *args, **kwargs):
        # Intel Arc fix, ref#50
        new_tensor = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(
                new_tensor,
                tensor_type = getattr(self, "tensor_type", None),
                tensor_shape = size,
                patches = getattr(self, "patches", []).copy(),
                compute_dtype = getattr(self, "compute_dtype", None),
        )

    @property
    def dtype(self):
        qtype = getattr(self, "tensor_type", None)
        if qtype in GGMLLayer.torch_compatible_tensor_types:
            # NOTE: use the base-class descriptor instead of torch.Tensor(self):
            # constructing a Tensor from an inference-mode tensor raises
            # "Inference tensors do not track version counter" (hit via
            # low_vram_patch_estimate_vram on bias keys when LoRAs patch biases)
            return torch.Tensor.dtype.__get__(self)
        return _infer_compute_dtype(qtype, getattr(self, "compute_dtype", None))

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
                if isinstance(v, GGMLTensor):
                    v.compute_dtype = self._ggml_compute_dtype(v, "weight")
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                if isinstance(v, GGMLTensor):
                    v.compute_dtype = self._ggml_compute_dtype(v, "bias")
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

    def _ggml_compute_dtype(self, tensor, param_name):
        if self.dequant_dtype is not None and self.dequant_dtype != "target":
            return self.dequant_dtype
        model_dtype = getattr(self, f"{param_name}_comfy_model_dtype", None)
        return _infer_compute_dtype(getattr(tensor, "tensor_type", None), model_dtype)

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
            self.weight_comfy_model_dtype = dtype
            self.bias_comfy_model_dtype = dtype

        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(GGMLLayer, comfy.ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Conv3d(GGMLLayer, comfy.ops.manual_cast.Conv3d):
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

def get_gguf_q8_ops(compute_dtype=torch.bfloat16, full_precision_mm=False):
    """
    Factory for an ops class that uses ComfyUI's native mixed_precision_ops INT8 path.
    Weights are kept as INT8 and matmul uses comfy_kitchen's TensorWiseINT8Layout.
    """
    BaseOps = comfy.ops.mixed_precision_ops(
        quant_config={},
        compute_dtype=compute_dtype,
        full_precision_mm=full_precision_mm,
    )

    class GGUFQ8Ops(BaseOps):
        class Linear(BaseOps.Linear):
            def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                # Lazy init: don't allocate weight here; it will be loaded from state dict
                torch.nn.Module.__init__(self)
                self.factory_kwargs = {"device": device, "dtype": BaseOps._compute_dtype}
                self.in_features = in_features
                self.out_features = out_features
                self.weight = None
                if bias:
                    self.bias = torch.nn.Parameter(torch.empty(out_features, **self.factory_kwargs))
                else:
                    self.register_parameter("bias", None)
                self._orig_shape = (out_features, in_features)
                self.tensor_class = None
                self._full_precision_mm = BaseOps._full_precision_mm
                self._full_precision_mm_config = False

            def forward(self, *args, **kwargs):
                input_tensor = args[0] if args else kwargs.get("input")
                parent_forward = super().forward
                return _perf_forward(
                    "int8_tensorwise",
                    self,
                    input_tensor,
                    lambda: parent_forward(*args, **kwargs),
                )

            def _load_from_state_dict(self, *args):
                state_dict, prefix = args[:2]
                weight_key = f"{prefix}weight"
                # Target-size GGUFs combine native Q8_CR weights with standard
                # Q4_0 weights. The mixed-precision loader only understands
                # the former's comfy_quant metadata, so materialize standard
                # GGML weights before handing them to that loader.
                if weight_key in state_dict and f"{prefix}comfy_quant" not in state_dict:
                    weight = state_dict[weight_key]
                    if is_quantized(weight):
                        state_dict[weight_key] = dequantize_tensor(
                            weight,
                            dtype=self.factory_kwargs["dtype"],
                        )
                    elif hasattr(weight, "dequantize"):
                        state_dict[weight_key] = weight.dequantize().to(
                            dtype=self.factory_kwargs["dtype"],
                        )
                return comfy.ops._load_quantized_module(
                    self,
                    torch.nn.Module._load_from_state_dict.__get__(self, type(self)),
                    *args,
                    load_extra_params=True,
                )

    return GGUFQ8Ops


# Q4_CR_W4A4: custom W4A4 INT4 format backed by comfy_kitchen's fast ConvRot
# int4 tensor-core MMA. On-disk is kitchen-native packed int4 (N, K//2) int8
# + a per-output-row fp16 scale. The weight is pre-rotated by a block-diagonal
# Hadamard along K; at runtime the kernel rotates the activation into the same
# basis, so output is directly in the original space.
def get_gguf_q4_w4a4_ops(compute_dtype=torch.bfloat16, full_precision_mm=False):
    try:
        from comfy_kitchen.tensor.convrot_w4a4 import (
            TensorCoreConvRotW4A4Layout,
            quantize_convrot_w4a4_weight,
        )
        from comfy_kitchen.tensor.base import QuantizedTensor
        _HAVE_KITCHEN = True
    except Exception:
        TensorCoreConvRotW4A4Layout = None
        quantize_convrot_w4a4_weight = None
        QuantizedTensor = None
        _HAVE_KITCHEN = False

    class GGUFQ4W4A4Ops(comfy.ops.disable_weight_init):
        class Linear(torch.nn.Module, comfy.ops.CastWeightBiasOp):
            comfy_cast_weights = True

            def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                torch.nn.Module.__init__(self)
                self.in_features = in_features
                self.out_features = out_features
                self.weight = None
                self.register_parameter("bias", None)
                self._orig_shape = (out_features, in_features)
                self._convrot_groupsize = 256
                self._quant_group_size = 64
                self._quantized = False
                self.weight_scale = None
                self._compute_dtype = torch.bfloat16
                self._quantized_weight = None
                self._quantized_weight_device = None
                # Cache for a dequantized non-native patch. Re-quantizing a small
                # delta to INT4 can round it away; compatible LoRA/LoKr instead use
                # the packed base plus their low-rank output correction.
                self._fused_weight = None
                self._fused_patch_id = None
                self._fused_weight_device = None
                self._fused_bias = None
                self._fused_bias_patch_id = None
                self._fused_bias_device = None
                self._lora_factor_cache = {}

            def install_patch_entries(self, patches, key):
                """Expose a static patch as the same callable used by Dynamic VRAM."""
                self.evict_quantized_caches()

                def apply_patch(weight, *args, **kwargs):
                    return comfy.lora.calculate_weight(patches, weight, key)

                apply_patch._gguf_static_patch = True
                apply_patch._gguf_patch_entries = patches
                apply_patch._gguf_patch_key = key
                self.weight_function = [apply_patch]
                self._gguf_static_patch = True

            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs):
                self.evict_quantized_caches()
                weight_key = f"{prefix}weight"
                scale_key = f"{prefix}weight_scale"
                quant_key = f"{prefix}comfy_quant"

                weight = state_dict.pop(weight_key, None)
                scale = state_dict.pop(scale_key, None)
                quant_raw = state_dict.pop(quant_key, None)

                bias_key = f"{prefix}bias"
                bias = state_dict.pop(bias_key, None)
                if bias is not None:
                    self.bias = torch.nn.Parameter(bias, requires_grad=False)
                else:
                    self.bias = None
                if bias_key in missing_keys:
                    missing_keys.remove(bias_key)

                if quant_raw is None or weight is None:
                    # Plain (non-quantized) Linear: keep the raw weight.
                    if weight is None:
                        missing_keys.append(weight_key)
                        return
                    self.weight = torch.nn.Parameter(
                        weight if isinstance(weight, torch.Tensor) else torch.Tensor(weight),
                        requires_grad=False,
                    )
                    self._quantized = False
                    return

                if scale is None:
                    raise RuntimeError(f"Missing Q4_CR_W4A4 scale tensor for {prefix}")

                quant_conf = json.loads(bytes(quant_raw.tolist()).decode("utf-8"))
                if quant_conf.get("format") != "int4_cr" or quant_conf.get("backing") != "w4a4":
                    raise ValueError(
                        f"Unsupported Q4_CR_W4A4 format for {prefix}: {quant_conf.get('format')!r} "
                        f"/ {quant_conf.get('backing')!r}"
                    )
                self._convrot_groupsize = quant_conf.get("convrot_groupsize", 256)
                self._quant_group_size = quant_conf.get("quant_group_size", 64)
                self._orig_shape = tuple(quant_conf.get("orig_shape", (self.out_features, self.in_features)))
                self.out_features = self._orig_shape[0]
                self.in_features = self._orig_shape[1]

                self.weight = torch.nn.Parameter(weight, requires_grad=False)
                self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
                self._quantized = True
                self._compute_dtype = quant_conf.get("orig_dtype", torch.bfloat16)
                self._quantized_weight = None
                self._quantized_weight_device = None
                self._fused_weight = None
                self._fused_patch_id = None
                self._fused_weight_device = None
                self._fused_bias = None
                self._fused_bias_patch_id = None
                self._fused_bias_device = None

            def evict_quantized_caches(self):
                """Release every derived representation of this quantized layer."""
                self._quantized_weight = None
                self._quantized_weight_device = None
                self._fused_weight = None
                self._fused_patch_id = None
                self._fused_weight_device = None
                self._fused_bias = None
                self._fused_bias_patch_id = None
                self._fused_bias_device = None
                self._fused_patch_keepalive = ()
                self._lora_factor_cache.clear()

            def move_fused_caches(self, device):
                """Move prepared patched caches without rebuilding them."""
                if self._fused_weight is not None and self._fused_weight_device != str(device):
                    self._fused_weight = self._fused_weight.to(device=device)
                    self._fused_weight_device = str(device)
                if self._fused_bias is not None and self._fused_bias_device != str(device):
                    self._fused_bias = self._fused_bias.to(device=device)
                    self._fused_bias_device = str(device)

            def _build_quantized_weight(self, device, dtype, packed=None):
                if not _HAVE_KITCHEN:
                    return None
                if packed is None:
                    packed = self.weight.to(device=device).to(torch.int8)
                else:
                    packed = packed.to(device=device).to(torch.int8)
                scale = self.weight_scale.to(device=device, dtype=torch.float32).contiguous()
                params = TensorCoreConvRotW4A4Layout.Params(
                    scale=scale,
                    orig_dtype=dtype,
                    orig_shape=self._orig_shape,
                    convrot_groupsize=self._convrot_groupsize,
                    quant_group_size=self._quant_group_size,
                    linear_dtype="int4",
                )
                return QuantizedTensor(packed, "TensorCoreConvRotW4A4Layout", params)

            def _get_cached_quantized_weight(self, device, packed=None):
                if packed is not None:
                    # A patched/offloaded packed weight may differ from self.weight
                    # (e.g. moved to device each forward). Build fresh; don't cache,
                    # because the packed content can change per forward.
                    return self._build_quantized_weight(device, self._compute_dtype, packed=packed)
                if self._quantized_weight is not None and self._quantized_weight_device == str(device):
                    return self._quantized_weight
                if self._quantized_weight is not None:
                    self._quantized_weight = None
                    if hasattr(self, "_quantized_weight_scale") and self._quantized_weight_scale is not None:
                        # Free the previous device's cached scale/params to avoid a leak.
                        self._quantized_weight_scale = None
                qt = self._build_quantized_weight(device, self._compute_dtype)
                self._quantized_weight = qt
                self._quantized_weight_device = str(device)
                return qt

            def _fused_patch_signature(self):
                """A cheap, stable identity for the active weight/bias patch set.

                ComfyUI installs the adapter (LoRA/LoKR) functions once per model
                load/patch, so the callable objects are stable across forwards. We key
                the fused-weight cache on the callable objects themselves so a re-bound
                patch (even if a function object is reused) is detected and re-fused.

                Under Dynamic VRAM, however, ``GGUFModelPatcherDynamic.load`` promotes a
                freshly created ``LowVramPatch`` into ``weight_function`` every time the
                model is moved to device. That object is recreated on each reload, so
                keying the cache on ``id(f)`` would miss on every forward and re-run the
                expensive Hadamard rotate + int4 pack (a 12x slowdown on large layers).
                For ``LowVramPatch`` we therefore key on its stable *content*: the tensor
                key and the identity of the shared patches dict/list (both live on the
                model patcher and survive recreation). A genuine patch change (e.g. a new
                ``add_patches`` call) rebuilds that list, which is detected as a change.
                All other adapter callables are stable objects, so we key on them directly.
                """
                sig = []
                keepalive = []
                for f in list(getattr(self, "weight_function", ())) + list(getattr(self, "bias_function", ())):
                    if getattr(f, "is_lowvram_patch", False) and getattr(f, "key", None):
                        patches = getattr(f, "patches", None)
                        plist = patches.get(f.key) if patches is not None else None
                        if patches is not None:
                            keepalive.append(patches)
                        if plist is not None:
                            keepalive.append(plist)
                        sig.append(("lowvram", f.key, id(patches), id(plist)))
                    else:
                        sig.append(("callable", f))
                # Keep references alive so address reuse never aliases a stale signature.
                self._fused_patch_keepalive = tuple(keepalive)
                return tuple(sig)

            def _get_cached_fused_bias(self, device, patch_id=None):
                bias_functions = list(getattr(self, "bias_function", ()))
                if not bias_functions:
                    return self.bias.to(device=device, dtype=self._compute_dtype) if self.bias is not None else None
                patch_id = patch_id if patch_id is not None else self._fused_patch_signature()
                if (
                    self._fused_bias is not None
                    and self._fused_bias_patch_id == patch_id
                ):
                    # The cache stays on the lifecycle-managed offload device. The
                    # caller creates a temporary device/dtype copy for the operation;
                    # moving this persistent tensor here would retain every patched
                    # layer on CUDA after it executes.
                    return self._fused_bias
                if self.bias is None:
                    self._fused_bias = None
                    self._fused_bias_patch_id = patch_id
                    self._fused_bias_device = str(device)
                    return None
                bias = self.bias.to(device=device, dtype=self._compute_dtype)
                for f in bias_functions:
                    bias = f(bias)
                self._fused_bias = bias
                self._fused_bias_patch_id = patch_id
                self._fused_bias_device = str(device)
                return bias

            def prepare_fused_weight(self, device):
                """Prepare fallback caches for active patch sets before inference.

                Compatible standard LoRA/LoKr patches keep the native INT4 base and
                bypass with a low-rank output correction. Other patch forms use a
                fused compute-dtype cache, kept on one device only.
                """
                if not self._quantized or self.weight is None:
                    return False
                weight_functions = list(getattr(self, "weight_function", ()))
                bias_functions = list(getattr(self, "bias_function", ()))
                if not weight_functions and not bias_functions:
                    if self._fused_weight is not None or self._fused_bias is not None:
                        self.evict_quantized_caches()
                    return False

                patch_id = self._fused_patch_signature()
                if self._fused_patch_id is not None and self._fused_patch_id != patch_id:
                    self.evict_quantized_caches()

                fused = True
                if weight_functions:
                    # Standard LoRA can use the packed INT4 base plus an exact
                    # low-rank output correction. Do not eagerly expand all such
                    # layers into full-precision matrices during model loading.
                    native_entries = self._native_lora_patch_entries()
                    if native_entries:
                        fused = False
                        if self._fused_weight is not None:
                            self._fused_weight = None
                            self._fused_patch_id = None
                            self._fused_weight_device = None
                    else:
                        self._quantized_weight = None
                        self._quantized_weight_device = None
                        try:
                            fused = self._get_cached_fused_weight(device) is not None
                        except torch.OutOfMemoryError:
                            if torch.device(device).type != "cuda":
                                raise
                            self._move_derived_caches_to_cpu()
                            logging.warning(
                                "Q4_CR_W4A4 CUDA fallback for INT4 adapter layer: "
                                "CUDA ran out of memory while preparing the patched weight."
                            )
                            fused = self._get_cached_fused_weight(torch.device("cpu")) is not None
                if bias_functions:
                    self._get_cached_fused_bias(device, patch_id=patch_id)
                return fused

            def _get_cached_fused_weight(self, device):
                """Apply a non-native patch once and cache the compute-dtype result.

                Re-quantizing a patched INT4 matrix can erase the small LoRA delta, so
                patched layers use the dequantized compute-dtype matrix. The pristine
                path below still uses the native INT4 kernel.
                """
                # A patched layer must never retain the ordinary packed cache alongside
                # either a fused representation or the dequantized fallback.
                self._quantized_weight = None
                self._quantized_weight_device = None
                sig = self._fused_patch_signature()
                if (
                    self._fused_weight is not None
                    and self._fused_patch_id == sig
                ):
                    # Keep the persistent fused matrix on its offload device. A
                    # forward may copy it temporarily to its input device, but must
                    # not promote the cache permanently; otherwise Dynamic VRAM
                    # accumulates one full-precision matrix for every LoRA layer.
                    return self._fused_weight
                if self._fused_weight is not None:
                    self.evict_quantized_caches()
                else:
                    # Do not retain the ordinary packed device cache while the
                    # patched representation is being prepared.
                    self._quantized_weight = None
                    self._quantized_weight_device = None
                # Apply adapters in the compute (BF16) domain on the un-rotated weight.
                fused = self._dequantized_weight(device, self._compute_dtype)
                for f in getattr(self, "weight_function", ()):
                    fused = f(fused)
                fused = fused.to(device=device, dtype=self._compute_dtype)
                self._fused_weight = fused
                self._fused_patch_id = sig
                self._fused_weight_device = str(device)
                self._quantized_weight = None
                self._quantized_weight_device = None
                return fused

            def _native_lora_patch_entries(self):
                """Return active low-rank patches that can bypass native INT4.

                A native base matmul plus the adapter's additive output is exact for
                ordinary LoRA/LoKr patches and avoids expanding the complete weight.
                Other patch forms retain the full-precision fusion path because they
                can transform the base weight or otherwise change its shape.
                """
                entries = []
                for patch_function in getattr(self, "weight_function", ()):
                    if getattr(patch_function, "is_lowvram_patch", False):
                        patches = getattr(patch_function, "patches", None)
                        key = getattr(patch_function, "key", None)
                        patch_list = patches.get(key) if patches is not None else None
                    elif getattr(patch_function, "_gguf_static_patch", False):
                        patch_list = getattr(patch_function, "_gguf_patch_entries", None)
                    else:
                        return None
                    if patch_list is None:
                        return None

                    prepared = getattr(patch_function, "prepared_patches", None)
                    if prepared is not None:
                        patch_list = prepared
                    for patch in patch_list:
                        if len(patch) != 5:
                            return None
                        strength, adapter, strength_model, offset, function = patch
                        if (
                            strength_model != 1.0
                            or offset is not None
                            or function is not None
                            or isinstance(adapter, list)
                            or getattr(adapter, "name", None) not in {"lora", "lokr"}
                            or not callable(getattr(adapter, "h", None))
                        ):
                            return None
                        weights = getattr(adapter, "weights", ())
                        # The current adapter h() implementations do not implement
                        # DoRA rescaling. Keep those patches on the exact fusion path.
                        dora_index = {"lora": 4, "lokr": 8}.get(adapter.name)
                        if dora_index is not None and len(weights) > dora_index and weights[dora_index] is not None:
                            return None
                        entries.append((strength, adapter))
                return entries

            def _get_cached_lora_factor(self, factor, device, dtype):
                """Move a CPU adapter factor to CUDA, retaining it only with headroom."""
                if not isinstance(factor, torch.Tensor):
                    return factor
                device = torch.device(device)
                try:
                    if factor.device.type != "cpu" or device.type != "cuda":
                        return comfy.model_management.cast_to_device(factor, device, dtype)

                    key = (id(factor), str(device), dtype)
                    cached = self._lora_factor_cache.get(key)
                    if cached is not None and cached[0] is factor:
                        return cached[1]

                    free_memory, _total_memory = torch.cuda.mem_get_info(device)
                    cache_bytes = factor.numel() * torch.empty((), dtype=dtype).element_size()
                    if (
                        free_memory
                        <= comfy.model_management.extra_reserved_memory() + cache_bytes
                    ):
                        return comfy.model_management.cast_to_device(factor, device, dtype)
                    cached_factor = comfy.model_management.cast_to_device(factor, device, dtype)
                except torch.OutOfMemoryError:
                    log_cuda_oom_loaded_models("caching INT4 LoRA/LoKr factors")
                    raise
                self._lora_factor_cache[key] = (factor, cached_factor)
                return cached_factor

            def _native_lora_bypass(self, input, base_out):
                entries = self._native_lora_patch_entries()
                if not entries:
                    return None
                out = base_out
                for strength, adapter in entries:
                    if adapter.name == "lora":
                        up, down, alpha, mid, dora_scale, reshape = adapter.weights
                        if mid is not None or dora_scale is not None or reshape is not None:
                            return None
                        up = self._get_cached_lora_factor(up, input.device, input.dtype)
                        down = self._get_cached_lora_factor(down, input.device, input.dtype)
                        rank = down.shape[0]
                        scale = (alpha / rank) if alpha is not None else 1.0
                        delta = torch.nn.functional.linear(
                            torch.nn.functional.linear(input, down), up
                        ) * scale
                    else:
                        # LoKr's h() is already a low-rank/Kronecker matmul. Its
                        # implementation casts factors to the input dtype but does
                        # not move them to the input device, so use a shallow,
                        # device-local adapter view instead of mutating the
                        # persistent CPU/offload adapter.
                        local_adapter = copy.copy(adapter)
                        local_adapter.weights = tuple(
                            self._get_cached_lora_factor(
                                weight, input.device, input.dtype
                            )
                            if isinstance(weight, torch.Tensor) else weight
                            for weight in getattr(adapter, "weights", ())
                        )
                        delta = local_adapter.h(input, base_out)
                    # Avoid materializing a scaled copy of delta and a second output
                    # tensor. This matters for large Flux activations where the
                    # residual can be hundreds of MiB even though the base is INT4.
                    out.add_(delta, alpha=strength)
                return out

            def _move_derived_caches_to_cpu(self):
                """Release CUDA derived caches before a system-memory retry."""
                self._quantized_weight = None
                self._quantized_weight_device = None
                self.move_fused_caches(torch.device("cpu"))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            def _cpu_forward_fallback(self, input, bias, native_entries=None, use_fused_weight=False):
                """Run one failed INT4 layer through CPU system memory."""
                original_device = input.device
                cpu = torch.device("cpu")
                if original_device.type == "cuda":
                    self._move_derived_caches_to_cpu()
                cpu_input = input.to(device=cpu)
                cpu_bias = bias.to(device=cpu, dtype=cpu_input.dtype) if bias is not None else None

                if native_entries:
                    cpu_weight = self._dequantized_weight(cpu, cpu_input.dtype)
                    cpu_out = torch.nn.functional.linear(cpu_input, cpu_weight)
                    if cpu_bias is not None:
                        cpu_out = cpu_out + cpu_bias
                    cpu_out = self._native_lora_bypass(cpu_input, cpu_out)
                elif use_fused_weight:
                    cpu_weight = self._get_cached_fused_weight(cpu)
                    cpu_out = torch.nn.functional.linear(
                        cpu_input,
                        cpu_weight.to(device=cpu, dtype=cpu_input.dtype),
                        cpu_bias,
                    )
                else:
                    cpu_weight = self._dequantized_weight(cpu, cpu_input.dtype)
                    cpu_out = torch.nn.functional.linear(cpu_input, cpu_weight, cpu_bias)

                return cpu_out.to(device=original_device, dtype=input.dtype)

            def forward_comfy_cast_weights(self, input, *args, **kwargs):
                has_weight_functions = bool(getattr(self, "weight_function", ()))
                has_bias_functions = bool(getattr(self, "bias_function", ()))
                if not has_weight_functions and not has_bias_functions and (
                        self._fused_weight is not None or self._fused_bias is not None
                ):
                    self.evict_quantized_caches()
                patch_id = self._fused_patch_signature() if (has_weight_functions or has_bias_functions) else None
                if has_bias_functions and self._fused_bias_patch_id == patch_id:
                    bias = (
                        self._fused_bias.to(device=input.device, dtype=input.dtype)
                        if self._fused_bias is not None else None
                    )
                else:
                    bias = self.bias.to(device=input.device, dtype=input.dtype) if self.bias is not None else None
                    if bias is not None:
                        for f in getattr(self, "bias_function", ()):
                            bias = f(bias)
                if not self._quantized or self.weight is None:
                    # Plain Linear path
                    weight = self.weight.to(device=input.device, dtype=input.dtype)
                    return torch.nn.functional.linear(input, weight, bias)
                if has_weight_functions or has_bias_functions:
                    # Compatible LoRA/LoKr uses the native base plus a low-rank output
                    # correction. Other patch forms use a cached un-rotated BF16
                    # fallback rather than re-quantizing a delta that INT4 could erase.
                    # Bias-only patches retain the native INT4 weight path.
                    if has_weight_functions:
                        native_entries = self._native_lora_patch_entries()
                        if (
                            native_entries is None
                            and self._fused_weight is not None
                            and self._fused_weight.device.type == "cpu"
                            and input.device.type == "cuda"
                        ):
                            return self._cpu_forward_fallback(
                                input, bias, use_fused_weight=True
                            )
                        try:
                            weight_qt = self._get_cached_quantized_weight(input.device)
                        except torch.OutOfMemoryError:
                            if input.device.type != "cuda":
                                raise
                            logging.warning(
                                "Q4_CR_W4A4 CUDA fallback for INT4 adapter layer: "
                                "CUDA ran out of memory; retrying this layer in system memory."
                            )
                            return self._cpu_forward_fallback(
                                input,
                                bias,
                                native_entries=native_entries,
                                use_fused_weight=not bool(native_entries),
                            )
                        if weight_qt is not None:
                            base_out = None
                            try:
                                base_out = torch.nn.functional.linear(input, weight_qt)
                                if bias is not None:
                                    base_out = base_out + bias
                                bypass_out = self._native_lora_bypass(input, base_out)
                                if bypass_out is not None:
                                    return bypass_out
                            except torch.OutOfMemoryError:
                                if input.device.type != "cuda":
                                    raise
                                del base_out
                                del weight_qt
                                logging.warning(
                                    "Q4_CR_W4A4 CUDA fallback for INT4 adapter layer: "
                                    "CUDA ran out of memory; retrying this layer in system memory."
                                )
                                return self._cpu_forward_fallback(
                                    input, bias, native_entries=native_entries
                                )
                        try:
                            fused_weight = self._get_cached_fused_weight(input.device)
                        except torch.OutOfMemoryError:
                            if input.device.type != "cuda":
                                raise
                            logging.warning(
                                "Q4_CR_W4A4 CUDA fallback for INT4 adapter layer: "
                                "CUDA ran out of memory; retrying this layer in system memory."
                            )
                            return self._cpu_forward_fallback(input, bias, use_fused_weight=True)
                        if fused_weight.device.type == "cpu" and input.device.type == "cuda":
                            return self._cpu_forward_fallback(input, bias, use_fused_weight=True)
                        try:
                            out = torch.nn.functional.linear(
                                input,
                                fused_weight.to(device=input.device, dtype=input.dtype),
                            )
                            if bias is not None:
                                out = out + bias
                            return out
                        except torch.OutOfMemoryError:
                            if input.device.type != "cuda":
                                raise
                            del fused_weight
                            logging.warning(
                                "Q4_CR_W4A4 CUDA fallback for INT4 adapter layer: "
                                "CUDA ran out of memory; retrying this layer in system memory."
                            )
                            return self._cpu_forward_fallback(input, bias, use_fused_weight=True)

                try:
                    weight_qt = self._get_cached_quantized_weight(input.device)
                    if weight_qt is None:
                        # No comfy_kitchen / non-CUDA: dequant fallback.
                        _orig_w = self._dequantized_weight(input.device, input.dtype)
                        return torch.nn.functional.linear(input, _orig_w, bias)

                    out = torch.nn.functional.linear(input, weight_qt)
                    if bias is not None:
                        out = out + bias
                    return out
                except torch.OutOfMemoryError:
                    if input.device.type != "cuda":
                        raise
                    logging.warning(
                        "Q4_CR_W4A4 CUDA fallback for INT4 layer: "
                        "CUDA ran out of memory; retrying this layer in system memory."
                    )
                    return self._cpu_forward_fallback(input, bias)

            def forward(self, *args, **kwargs):
                input_tensor = args[0] if args else kwargs.get("input")

                def run_forward():
                    comfy.ops.run_every_op()
                    if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                        return self.forward_comfy_cast_weights(*args, **kwargs)
                    return torch.nn.functional.linear(
                        input=input_tensor,
                        weight=self.weight,
                        bias=self.bias,
                    )

                return _perf_forward("int4_convrot_w4a4", self, input_tensor, run_forward)

            def _dequantized_weight(self, device, dtype):
                packed = self.weight.to(device=device).to(torch.int8)
                scale = self.weight_scale.to(device=device, dtype=dtype)
                n, k_half = packed.shape
                k = k_half * 2
                x32 = packed.to(torch.int32)
                lo = (x32 & 0xF).to(torch.int8)
                hi = ((x32 >> 4) & 0xF).to(torch.int8)
                nibbles = torch.stack([lo, hi], dim=-1).reshape(n, k).to(torch.float32)
                # Signed two's-complement int4 -> [-8, 7], scaled per output row.
                nibbles = torch.where(nibbles >= 8, nibbles - 16, nibbles)
                w_rot = nibbles * scale.to(torch.float32).reshape(-1, 1)
                # The weight was pre-rotated by a block-diagonal Hadamard along K;
                # un-rotate so this dequant matches the kernel's effective weight.
                cg = self._convrot_groupsize
                if k % cg == 0:
                    h = _build_regular_hadamard(cg, dtype=torch.float32, device=device)
                    n_groups = k // cg
                    w_rot = (w_rot.reshape(n, n_groups, cg) @ h).reshape(n, k)
                return w_rot.to(dtype)

        def reset_parameters(self):
            return None

    return GGUFQ4W4A4Ops


# Retired experimental Q4_PT implementation. No loader or node runtime path
# references this class until a performant W4A16 backend is available.
class RetiredGGUFQ4Ops(comfy.ops.manual_cast):
    """
    Ops class for PyTorch's native compact INT4 GEMM.

    Packed weights are created transiently at invocation time. This keeps
    low-VRAM offloading viable without retaining a second copy of all weights.
    """
    class Linear(torch.nn.Module, comfy.ops.CastWeightBiasOp):
        comfy_cast_weights = True

        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.register_parameter("bias", None)
            self._orig_shape = (out_features, in_features)
            self._group_size = None
            self._pad = 0
            self._orig_in_features = in_features
            self._is_int4 = False

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            weight_key = f"{prefix}weight"
            scale_key = f"{prefix}weight_scale"
            quant_key = f"{prefix}comfy_quant"

            weight = state_dict.pop(weight_key, None)
            scale = state_dict.pop(scale_key, None)
            quant_raw = state_dict.pop(quant_key, None)

            bias_key = f"{prefix}bias"
            bias = state_dict.pop(bias_key, None)
            if quant_raw is None:
                if weight is None:
                    missing_keys.append(weight_key)
                    return
                self.weight = torch.nn.Parameter(torch.Tensor(weight), requires_grad=False)
                if bias is not None:
                    self.bias = torch.nn.Parameter(torch.Tensor(bias), requires_grad=False)
                self._is_int4 = False
                return

            if weight is None or scale is None:
                raise RuntimeError(f"Missing INT4 tensors for {prefix}")

            quant_conf = json.loads(bytes(quant_raw.tolist()).decode("utf-8"))
            if quant_conf.get("format") not in {"int4_compact_gemm", "int4_pytorch"}:
                raise ValueError(f"Unsupported INT4 format for {prefix}")
            self._group_size = quant_conf["group_size"]
            self._pad = quant_conf.get("pad", 0)
            orig_shape = tuple(quant_conf["orig_shape"])
            self._orig_in_features = orig_shape[1]
            self._orig_shape = orig_shape
            self.out_features = orig_shape[0]

            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
            self._is_int4 = True

            if bias is not None:
                self.bias = torch.nn.Parameter(bias, requires_grad=False)
            for key in (weight_key, scale_key, quant_key, bias_key):
                if key in missing_keys:
                    missing_keys.remove(key)

        def forward(self, input):
            if self.weight is None:
                raise RuntimeError("Q4_PT weight was not loaded.")
            if not self._is_int4:
                weight = self.weight.to(device=input.device, dtype=input.dtype)
                bias = self.bias.to(device=input.device, dtype=input.dtype) if self.bias is not None else None
                return torch.nn.functional.linear(input, weight, bias)
            if self.weight_function or self.bias_function:
                raise RuntimeError("Q4_PT does not support weight patches or LoRAs without dequantization.")
            input_shape = input.shape
            input_2d = input.reshape(-1, input_shape[-1])
            if self._pad:
                input_2d = torch.nn.functional.pad(input_2d, (0, self._pad))

            if self._group_size != 64:
                raise RuntimeError(
                    f"Q4_PT requires PyTorch's group-size-64 INT4 operator, got {self._group_size}."
                )
            if input_2d.device.type != "cuda":
                raise RuntimeError("Q4_PT requires a CUDA device.")
            if input_2d.dtype != torch.bfloat16:
                raise RuntimeError(
                    f"Q4_PT requires BF16 activations for PyTorch's native INT4 operator, got {input_2d.dtype}."
                )

            weight = torch.Tensor(
                self.weight.to(device=input.device, dtype=torch.uint8, non_blocking=True)
            )
            scale_and_offset = self.weight_scale.to(
                device=input.device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )
            packed_features = weight.size(-1) * 2
            native_padding = (-packed_features) % 128
            if native_padding:
                # PyTorch's INT4 packer needs K to be a multiple of 128.
                # A zero-valued group and zero input features preserve the
                # original result for K=64 projections such as Krea2's input.
                if native_padding % self._group_size:
                    raise RuntimeError(
                        f"Cannot pad Q4_PT input width {packed_features} to PyTorch's INT4 tile size."
                    )
                input_2d = torch.nn.functional.pad(input_2d, (0, native_padding))
                weight = torch.nn.functional.pad(weight, (0, native_padding // 2))
                scale_and_offset = torch.nn.functional.pad(
                    scale_and_offset,
                    (0, 0, 0, 0, 0, native_padding // self._group_size),
                )

            packed_weight = torch._convert_weight_to_int4pack(weight, 8)
            output = torch._weight_int4pack_mm(
                input_2d,
                packed_weight,
                self._group_size,
                scale_and_offset,
            )
            if self.bias is not None:
                output = output + self.bias.to(device=input.device, dtype=input.dtype)
            return output.reshape(*input_shape[:-1], self.out_features)
