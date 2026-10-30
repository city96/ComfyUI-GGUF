# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Accelerated Linear backend for GGML K-quant weights.

An accelerator package may either call :func:`register_kquant_backend` during
import or expose ``linear`` (and optionally ``supports``) from the module named
by ``COMFYUI_GGUF_KQUANT_BACKEND``. The backend receives the original GGUF
block bytes; this module never repacks or dequantizes them.
"""

import importlib
import logging
import os
import threading

import gguf
import torch


SUPPORTED_K_QUANTS = frozenset(
    {
        gguf.GGMLQuantizationType.Q3_K,
        gguf.GGMLQuantizationType.Q4_K,
        gguf.GGMLQuantizationType.Q5_K,
        gguf.GGMLQuantizationType.Q6_K,
    }
)
_BACKEND_ENV = "COMFYUI_GGUF_KQUANT_BACKEND"
_DEFAULT_BACKEND = "bundled"
_LOCAL_BACKEND_MODULE = ".kquant_triton_backend"
_backend = None
_backend_checked = False
_backend_lock = threading.RLock()
_failed_routes = set()


def register_kquant_backend(backend):
    """Register an object implementing ``linear`` and optionally ``supports``."""
    if backend is not None and not callable(getattr(backend, "linear", None)):
        raise TypeError("A K-quant backend must provide a callable linear()")

    global _backend, _backend_checked
    with _backend_lock:
        _backend = backend
        _backend_checked = backend is not None
        _failed_routes.clear()


def _module_backend(module):
    get_backend = getattr(module, "get_backend", None)
    backend = get_backend() if callable(get_backend) else getattr(module, "backend", module)
    if not callable(getattr(backend, "linear", None)):
        raise TypeError(
            f"K-quant backend module {module.__name__!r} does not expose linear()"
        )
    return backend


def _load_backend():
    global _backend, _backend_checked
    if _backend_checked:
        return _backend

    with _backend_lock:
        if _backend_checked:
            return _backend

        configured_module = os.environ.get(_BACKEND_ENV, _DEFAULT_BACKEND).strip()
        if not configured_module:
            configured_module = _DEFAULT_BACKEND
        if configured_module.lower() in {"none", "off", "disabled", "false"}:
            _backend_checked = True
            logging.info(
                "ComfyUI-GGUF: K-quant backend opted out via %s=%r; "
                "using portable dequantization",
                _BACKEND_ENV,
                configured_module,
            )
            return None

        candidates = (
            [_LOCAL_BACKEND_MODULE]
            if configured_module.lower() == "bundled"
            else [configured_module]
        )
        last_error = None
        for module_name in candidates:
            try:
                module = importlib.import_module(module_name, package=__package__)
                if _backend is None:
                    _backend = _module_backend(module)
                logging.info("ComfyUI-GGUF: enabled K-quant backend %s", module_name)
                break
            except ModuleNotFoundError as error:
                last_error = error
                if configured_module or error.name not in {module_name, module_name.lstrip('.')}:
                    logging.warning(
                        "ComfyUI-GGUF: unable to import K-quant backend %r: %s",
                        module_name,
                        error,
                    )
            except Exception as error:
                last_error = error
                logging.warning(
                    "ComfyUI-GGUF: unable to initialize K-quant backend %r: %s",
                    module_name,
                    error,
                )
        if _backend is None and last_error is not None:
            logging.warning(
                "ComfyUI-GGUF: using dequantized Linear fallback for K-quants"
            )
        _backend_checked = True
        return _backend


def _storage_nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def _valid_storage(qdata, qtype, weight_shape):
    if len(weight_shape) != 2:
        return False
    out_features, in_features = weight_shape
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    if out_features < 0 or in_features <= 0 or in_features % block_size:
        return False
    expected = out_features * (in_features // block_size) * type_size
    return _storage_nbytes(qdata) == expected


def _route_key(backend, qtype, input_tensor):
    device = input_tensor.device
    return (
        id(backend),
        qtype,
        device.type,
        device.index,
        input_tensor.dtype,
    )


def _backend_supports(backend, *, qtype, input_tensor, qdata, weight_shape):
    supports = getattr(backend, "supports", None)
    if not callable(supports):
        return True
    return bool(
        supports(
            qtype=qtype.name,
            device=input_tensor.device,
            input_dtype=input_tensor.dtype,
            weight_shape=weight_shape,
            weight_dtype=qdata.dtype,
        )
    )


def _backend_should_use(backend, *, qtype, input_tensor, qdata, weight_shape, dequant_dtype):
    should_use = getattr(backend, "should_use", None)
    precision_kwargs = (
        {"dequant_dtype": dequant_dtype}
        if getattr(backend, "supports_dequant_dtype", False)
        else {}
    )
    if not callable(should_use):
        return True
    return bool(
        should_use(
            qtype=qtype.name,
            device=input_tensor.device,
            input_dtype=input_tensor.dtype,
            input_shape=tuple(input_tensor.shape),
            weight_shape=weight_shape,
            weight_dtype=qdata.dtype,
            **precision_kwargs,
        )
    )


def try_kquant_linear(input_tensor, qdata, qtype, weight_shape, bias=None, *, dequant_dtype="target"):
    """Return accelerated Linear output, or ``None`` when fallback should be used."""
    if qtype not in SUPPORTED_K_QUANTS or not torch.is_tensor(qdata):
        return None

    if type(qdata) is not torch.Tensor:
        qdata = qdata.as_subclass(torch.Tensor)
    weight_shape = tuple(int(dim) for dim in weight_shape)
    if len(weight_shape) != 2 or input_tensor.ndim == 0:
        return None
    if input_tensor.shape[-1] != weight_shape[-1]:
        return None
    if qdata.device != input_tensor.device or not _valid_storage(qdata, qtype, weight_shape):
        return None

    # Gate on backend availability before paying for any repacking: the
    # reshape/contiguous below copies whole weight storages, so it must not
    # run on routes that fall back to portable dequantization anyway.
    backend = _load_backend()
    if backend is None:
        return None
    decode_dtype = input_tensor.dtype if dequant_dtype == "target" else dequant_dtype
    decode_dtype = torch.float16 if decode_dtype is None else decode_dtype
    precision_aware = getattr(backend, "supports_dequant_dtype", False)
    if not precision_aware and decode_dtype != input_tensor.dtype:
        return None

    route = _route_key(backend, qtype, input_tensor)
    if route in _failed_routes:
        return None

    try:
        if not _backend_supports(
            backend,
            qtype=qtype,
            input_tensor=input_tensor,
            qdata=qdata,
            weight_shape=weight_shape,
        ) or not _backend_should_use(
            backend,
            qtype=qtype,
            input_tensor=input_tensor,
            qdata=qdata,
            weight_shape=weight_shape,
            dequant_dtype=decode_dtype,
        ):
            return None
        out_features, in_features = weight_shape
        _, type_size = gguf.GGML_QUANT_SIZES[qtype]
        row_bytes = (in_features // 256) * type_size
        packed = qdata.reshape((out_features, row_bytes)).contiguous()
        precision_kwargs = {"dequant_dtype": decode_dtype} if precision_aware else {}
        output = backend.linear(
            input=input_tensor,
            weight=packed,
            qtype=qtype.name,
            weight_shape=weight_shape,
            bias=bias,
            **precision_kwargs,
        )
        expected_shape = (*input_tensor.shape[:-1], weight_shape[0])
        if not torch.is_tensor(output) or tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"backend returned {type(output).__name__} with shape "
                f"{getattr(output, 'shape', None)}, expected {expected_shape}"
            )
        if output.device != input_tensor.device:
            raise RuntimeError(
                f"backend returned output on {output.device}, expected {input_tensor.device}"
            )
        if output.dtype != input_tensor.dtype:
            raise RuntimeError(
                f"backend returned {output.dtype}, expected {input_tensor.dtype}"
            )
        return output
    except Exception as error:
        _failed_routes.add(route)
        logging.warning(
            "ComfyUI-GGUF: K-quant backend failed for %s on %s (%s); "
            "using dequantized Linear fallback for this route",
            qtype.name,
            input_tensor.device,
            error,
        )
        return None
