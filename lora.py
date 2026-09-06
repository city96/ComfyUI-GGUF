import json
import logging
import os

import gguf
import torch
from safetensors import safe_open


_SUPPORTED_FACTOR_TYPES = {
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
    gguf.GGMLQuantizationType.BF16,
    gguf.GGMLQuantizationType.Q8_0,
}
_FP8_SOURCE_TYPES = {
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    )
    if dtype is not None
}


def _read_int8_quant_config(state_dict, weight_key):
    quant_key = f"{weight_key[:-len('weight')]}comfy_quant"
    quant_data = state_dict.get(quant_key)
    if quant_data is None:
        return None, quant_key
    if not isinstance(quant_data, torch.Tensor) or quant_data.dtype != torch.uint8:
        raise ValueError(
            f"INT8 source quantization metadata {quant_key!r} must be a uint8 JSON tensor."
        )
    try:
        return json.loads(bytes(quant_data.detach().cpu().tolist()).decode("utf-8")), quant_key
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"INT8 source quantization metadata {quant_key!r} is invalid JSON."
        ) from error


def _unrotate_convrot_weight(weight, group_size):
    from comfy_kitchen.tensor.int8_utils import _build_hadamard

    if group_size <= 0 or weight.shape[1] % group_size:
        raise ValueError(
            f"ConvRot group size {group_size} does not divide weight shape {tuple(weight.shape)}."
        )
    hadamard = _build_hadamard(group_size, device=weight.device, dtype=weight.dtype)
    return torch.matmul(
        weight.reshape(weight.shape[0], -1, group_size),
        hadamard,
    ).reshape_as(weight)


def materialize_int8_source_weights(state_dict):
    """Restore scaled INT8 weights, reversing ConvRot before LoRA fusion or export."""
    int8_keys = [
        key for key, value in state_dict.items()
        if isinstance(value, torch.Tensor) and value.dtype == torch.int8
    ]
    for weight_key in int8_keys:
        if not weight_key.endswith(".weight"):
            raise ValueError(
                f"INT8 source tensor {weight_key!r} is not a Linear weight. "
                "Only scaled INT8 Linear checkpoint weights are supported."
            )
        scale_key = f"{weight_key}_scale"
        scale = state_dict.get(scale_key)
        if scale is None:
            raise ValueError(
                f"INT8 source weight {weight_key!r} is missing its scale tensor "
                f"{scale_key!r}."
            )
        if not isinstance(scale, torch.Tensor) or not scale.is_floating_point():
            raise ValueError(
                f"INT8 source weight scale {scale_key!r} must be a floating-point tensor."
            )

        weight = state_dict[weight_key]
        scale = scale.to(dtype=torch.float32)
        if scale.ndim == 1 and scale.shape[0] == weight.shape[0]:
            scale = scale.unsqueeze(1)
        try:
            restored = weight.to(dtype=torch.float32) * scale
        except RuntimeError as error:
            raise ValueError(
                f"INT8 source weight scale {scale_key!r} with shape {tuple(scale.shape)} "
                f"cannot be broadcast over {weight_key!r} with shape {tuple(weight.shape)}."
            ) from error

        quant_conf, quant_key = _read_int8_quant_config(state_dict, weight_key)
        if quant_conf is not None:
            if quant_conf.get("format") != "int8_tensorwise":
                raise ValueError(
                    f"INT8 source quantization metadata {quant_key!r} uses unsupported "
                    f"format {quant_conf.get('format')!r}."
                )
            if quant_conf.get("convrot"):
                restored = _unrotate_convrot_weight(
                    restored,
                    int(quant_conf.get("convrot_groupsize", 256)),
                )
            del state_dict[quant_key]

        state_dict[weight_key] = restored.to(dtype=torch.float16)
        del state_dict[scale_key]
    return len(int8_keys)


def _read_string_field(reader, name, required=True):
    field = reader.get_field(name)
    if field is None:
        if required:
            raise ValueError(f"GGUF LoRA is missing required metadata {name!r}.")
        return None
    if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
        raise ValueError(f"GGUF LoRA metadata {name!r} must be a string.")
    return str(field.parts[field.data[-1]], encoding="utf-8")


def _read_float_field(reader, name):
    field = reader.get_field(name)
    if field is None:
        return None
    if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.FLOAT32:
        raise ValueError(f"GGUF LoRA metadata {name!r} must be a float32.")
    return float(field.parts[field.data[-1]].item())


def _tensor_to_float(tensor):
    if tensor.tensor_type not in _SUPPORTED_FACTOR_TYPES:
        raise ValueError(
            f"GGUF LoRA tensor {tensor.name!r} uses unsupported {tensor.tensor_type.name}. "
            "Only F32, F16, BF16, and Q8_0 factors are supported."
        )
    shape = tuple(int(value) for value in reversed(tensor.shape))
    data = torch.from_numpy(tensor.data.copy())
    if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
        return data.view(torch.float32).reshape(shape)
    if tensor.tensor_type == gguf.GGMLQuantizationType.F16:
        return data.view(torch.float16).reshape(shape)
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        return data.view(torch.bfloat16).reshape(shape)
    return torch.from_numpy(
        gguf.quants.dequantize(tensor.data, tensor.tensor_type)
    ).reshape(shape).to(torch.float16)


def _build_targets(pairs, path, architecture=None, default_alpha=None):
    if not pairs:
        raise ValueError("LoRA contains no factor tensors.")

    lora = {}
    targets = {}
    for base_name, pair in pairs.items():
        if not {"a", "b"}.issubset(pair):
            raise ValueError(f"LoRA target {base_name!r} is missing one factor.")
        down, up = pair["a"], pair["b"]
        if down.ndim != 2 or up.ndim != 2:
            raise ValueError(
                f"LoRA target {base_name!r} is not 2-D; convolutional and other "
                "adapter layouts are not supported."
            )
        if down.shape[0] != up.shape[1]:
            raise ValueError(
                f"LoRA target {base_name!r} has incompatible factor shapes "
                f"{tuple(down.shape)} and {tuple(up.shape)}."
            )
        target_name = base_name.removesuffix(".weight")
        source_name = base_name if base_name.endswith(".weight") else f"{base_name}.weight"
        alpha = pair.get("alpha", default_alpha)
        lora[f"{target_name}.lora_A.weight"] = down
        lora[f"{target_name}.lora_B.weight"] = up
        if alpha is not None:
            lora[f"{target_name}.alpha"] = torch.tensor(alpha, dtype=torch.float32)
        targets[target_name] = {
            "base_name": source_name,
            "down": down,
            "up": up,
            "alpha": alpha,
        }

    metadata = {
        "path": os.path.abspath(path),
        "architecture": architecture,
        "alpha": default_alpha,
        "target_count": len(targets),
    }
    return lora, targets, metadata


def load_gguf_lora(path):
    """Read a standard GGUF LoRA into ComfyUI's lora_A/lora_B dictionary form."""
    reader = gguf.GGUFReader(path)
    try:
        if _read_string_field(reader, "general.type") != "adapter":
            raise ValueError("GGUF file is not an adapter (general.type must be 'adapter').")
        if _read_string_field(reader, "adapter.type") != "lora":
            raise ValueError("GGUF adapter is not a LoRA (adapter.type must be 'lora').")

        pairs = {}
        for tensor in reader.tensors:
            if tensor.name.endswith(".lora_a"):
                base_name = tensor.name[:-len(".lora_a")]
                pairs.setdefault(base_name, {})["a"] = _tensor_to_float(tensor)
            elif tensor.name.endswith(".lora_b"):
                base_name = tensor.name[:-len(".lora_b")]
                pairs.setdefault(base_name, {})["b"] = _tensor_to_float(tensor)
            else:
                raise ValueError(
                    f"Unsupported GGUF LoRA tensor {tensor.name!r}; only paired "
                    "'.lora_a' and '.lora_b' factors are supported."
                )

        return _build_targets(
            pairs,
            path,
            architecture=_read_string_field(reader, "general.architecture", required=False),
            default_alpha=_read_float_field(reader, "adapter.lora.alpha"),
        )
    finally:
        reader.tensors.clear()
        reader.fields.clear()
        reader.data._mmap.close()


_SAFETENSORS_FACTOR_SUFFIXES = (
    (".lora_A.weight", "a"),
    (".lora_B.weight", "b"),
    (".lora_down.weight", "a"),
    (".lora_up.weight", "b"),
)
_SAFETENSORS_LOKR_SUFFIXES = (
    (".lokr_w1", "w1"),
    (".lokr_w2", "w2"),
)


def load_safetensors_lora(path):
    """Load Linear LoRA and direct-factor LyCORIS LoKr safetensors adapters."""
    pairs = {}
    lokr_pairs = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        for key in keys:
            match = next(
                ((suffix, factor) for suffix, factor in _SAFETENSORS_FACTOR_SUFFIXES if key.endswith(suffix)),
                None,
            )
            if match is None:
                match = next(
                    (
                        (suffix, factor)
                        for suffix, factor in _SAFETENSORS_LOKR_SUFFIXES
                        if key.endswith(suffix)
                    ),
                    None,
                )
                if match is None:
                    continue
                suffix, factor = match
                base_name = key[:-len(suffix)]
                lokr_pairs.setdefault(base_name, {})[factor] = checkpoint.get_tensor(key)
            else:
                suffix, factor = match
                base_name = key[:-len(suffix)]
                pairs.setdefault(base_name, {})[factor] = checkpoint.get_tensor(key)

        for base_name, pair in pairs.items():
            for alpha_key in (f"{base_name}.alpha", f"{base_name}.lora_alpha"):
                if alpha_key not in keys:
                    continue
                alpha = checkpoint.get_tensor(alpha_key)
                if alpha.numel() != 1:
                    raise ValueError(f"LoRA alpha {alpha_key!r} must be a scalar.")
                pair["alpha"] = float(alpha.item())
                break

        for base_name, pair in lokr_pairs.items():
            for alpha_key in (f"{base_name}.alpha", f"{base_name}.lora_alpha"):
                if alpha_key not in keys:
                    continue
                alpha = checkpoint.get_tensor(alpha_key)
                if alpha.numel() != 1:
                    raise ValueError(f"LoKr alpha {alpha_key!r} must be a scalar.")
                pair["alpha"] = float(alpha.item())
                break

    if pairs:
        lora, targets, metadata = _build_targets(pairs, path)
    else:
        lora = {}
        targets = {}
        metadata = {
            "path": os.path.abspath(path),
            "architecture": None,
            "alpha": None,
            "target_count": 0,
        }

    for base_name, pair in lokr_pairs.items():
        if not {"w1", "w2"}.issubset(pair):
            raise ValueError(
                f"LoKr target {base_name!r} is missing a direct lokr_w1 or lokr_w2 factor. "
                "Factorized and Tucker LoKr layouts are not supported for offline fusion."
            )
        w1, w2 = pair["w1"], pair["w2"]
        if w1.ndim != 2 or w2.ndim != 2:
            raise ValueError(
                f"LoKr target {base_name!r} is not a 2-D Linear adapter."
            )
        target_name = base_name.removesuffix(".weight")
        source_name = base_name if base_name.endswith(".weight") else f"{base_name}.weight"
        lora[f"{target_name}.lokr_w1"] = w1
        lora[f"{target_name}.lokr_w2"] = w2
        targets[target_name] = {
            "base_name": source_name,
            "kind": "lokr",
            "w1": w1,
            "w2": w2,
            "alpha": pair.get("alpha"),
        }

    metadata["target_count"] = len(targets)
    if not targets:
        raise ValueError("LoRA contains no supported factor tensors.")
    return lora, targets, metadata


def load_lora(path):
    """Load a GGUF or safetensors LoRA suitable for offline fusion."""
    extension = os.path.splitext(path)[1].lower()
    if extension == ".gguf":
        return load_gguf_lora(path)
    if extension == ".safetensors":
        return load_safetensors_lora(path)
    raise ValueError(f"LoRA fusion accepts .gguf or .safetensors adapters, got {path!r}.")


def _find_state_key(state_dict, candidates):
    return next((candidate for candidate in candidates if candidate in state_dict), None)


def _comfy_model_lora_target_map(state_dict):
    """Build ComfyUI's current diffusion-model LoRA map without loading weights."""
    import comfy.lora
    import comfy.model_detection

    model_config = comfy.model_detection.model_config_from_unet(state_dict, "")
    if model_config is None:
        return {}

    model = model_config.get_model({}, device=torch.device("meta"))
    source_state = {
        f"diffusion_model.{key}": value
        for key, value in state_dict.items()
    }
    object.__setattr__(
        model,
        "state_dict",
        lambda *args, **kwargs: source_state,
    )
    return comfy.lora.model_lora_keys_unet(model, {})


def _target_shape(target):
    if target.get("kind") == "lokr":
        return (
            target["w1"].shape[0] * target["w2"].shape[0],
            target["w1"].shape[1] * target["w2"].shape[1],
        )
    return (target["up"].shape[0], target["down"].shape[1])


def resolve_fusion_targets(state_dict, targets):
    resolved = {}
    missing = []
    comfy_map = None
    for target_name, target in targets.items():
        candidates = (
            target["base_name"],
            f"{target_name}.weight",
            target_name,
            target["base_name"].removeprefix("diffusion_model."),
            f"{target_name.removeprefix('diffusion_model.')}.weight",
        )
        state_key = _find_state_key(state_dict, candidates)
        target_slice = None
        if state_key is None:
            if comfy_map is None:
                comfy_map = _comfy_model_lora_target_map(state_dict)
            mapped = comfy_map.get(target_name)
            if isinstance(mapped, tuple):
                mapped, target_slice = mapped[:2]
            if mapped is not None:
                bare_mapped = mapped.removeprefix("model.diffusion_model.").removeprefix(
                    "diffusion_model."
                )
                state_key = _find_state_key(
                    state_dict,
                    (
                        mapped,
                        bare_mapped,
                        f"diffusion_model.{bare_mapped}",
                        f"model.diffusion_model.{bare_mapped}",
                    ),
                )
        if state_key is None:
            missing.append(target["base_name"])
            continue
        weight = state_dict[state_key]
        target_weight = weight
        if target_slice is not None:
            target_weight = weight.narrow(*target_slice)
        if target_weight.ndim != 2:
            raise ValueError(
                f"GGUF LoRA target {target['base_name']!r} resolves to {state_key!r}, "
                f"which is {target_weight.ndim}-D. Only 2-D Linear weights can be fused."
            )
        if tuple(target_weight.shape) != _target_shape(target):
            raise ValueError(
                f"GGUF LoRA target {target['base_name']!r} does not match {state_key!r}: "
                f"base shape {tuple(target_weight.shape)}, adapter delta shape "
                f"{_target_shape(target)}."
            )
        resolved.setdefault(state_key, []).append((target, target_slice))
    if missing:
        logging.warning(
            "Skipping %d LoRA target(s) not present in the source checkpoint: %s",
            len(missing),
            ", ".join(sorted(missing)),
        )
    return resolved


def fuse_targets_into_state_dict(state_dict, targets, strength, device):
    """Apply one adapter's linear deltas in GPU/CPU FP32, returning the target count."""
    resolved = resolve_fusion_targets(state_dict, targets)
    if not resolved:
        logging.warning(
            "No compatible LoRA targets were found in the source checkpoint; "
            "this adapter does not change the fused cache."
        )
    fused_count = 0
    for state_key, target_entries in resolved.items():
        source = state_dict[state_key]
        state_dict[state_key], count = fuse_target_entries_into_tensor(
            source,
            target_entries,
            strength,
            device,
            state_dict.get(f"{state_key}_scale"),
        )
        fused_count += count
    return fused_count


def fuse_target_entries_into_tensor(source, target_entries, strength, device, source_scale=None):
    """Fuse resolved LoRA targets into one tensor without retaining unrelated weights."""
    source_dtype = source.dtype
    if source_dtype not in {torch.float16, torch.bfloat16, torch.float32, *_FP8_SOURCE_TYPES}:
        raise ValueError(
            f"Fusion target has unsupported dtype {source.dtype}. "
            "Fuse from an FP16, BF16, FP32, or scaled FP8 checkpoint."
        )
    fused = source.to(device=device, dtype=torch.float32)
    if source_dtype in _FP8_SOURCE_TYPES and source_scale is not None:
        if source_scale.numel() != 1:
            raise ValueError("FP8 scale tensor must be a scalar.")
        fused.mul_(source_scale.to(device=device, dtype=torch.float32))
    for target, target_slice in target_entries:
        target_fused = fused if target_slice is None else fused.narrow(*target_slice)
        if target.get("kind") == "lokr":
            w1 = target["w1"].to(device=device, dtype=torch.float32)
            w2 = target["w2"].to(device=device, dtype=torch.float32)
            target_fused.add_(torch.kron(w1, w2), alpha=strength)
            del w1, w2
        else:
            down = target["down"].to(device=device, dtype=torch.float32)
            up = target["up"].to(device=device, dtype=torch.float32)
            alpha = target["alpha"] if target["alpha"] is not None else down.shape[0]
            target_fused.add_(up.matmul(down), alpha=strength * alpha / down.shape[0])
            del down, up
    output_dtype = torch.float16 if source_dtype in _FP8_SOURCE_TYPES else source_dtype
    return fused.to(device="cpu", dtype=output_dtype), len(target_entries)
