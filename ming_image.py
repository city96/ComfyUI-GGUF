# Ming-Image GGUF compatibility helpers.
#
# Ming-Image's Diffusers-format checkpoint stores attention Q/K/V separately.
# ComfyUI's generic conversion merges those tensors with ordinary torch tensor
# assignment. For quantized GGUF tensors, the physical packed byte width can
# differ from the logical tensor width, so that merge can fail.
#
# This helper performs the same Ming/Z-Image key mapping while preserving the
# GGML tensor type and logical shape of packed tensors.

import logging
import torch
import comfy.utils

from .ops import GGMLTensor
from .dequant import is_quantized


def _count_blocks(keys, prefix):
    i = 0
    while any(key.startswith(prefix.format(i)) for key in keys):
        i += 1
    return i


def _base_tensor(tensor):
    if isinstance(tensor, GGMLTensor):
        return torch.Tensor(tensor)
    return tensor


def _merge_offset_parts(target, parts):
    parts = sorted(parts, key=lambda item: item[0][1])

    if len(parts) != 3:
        raise RuntimeError(
            f"Ming-Image GGUF expected 3 Q/K/V parts for {target}, got {len(parts)}"
        )

    axes = {int(item[0][0]) for item in parts}
    if axes != {0}:
        raise RuntimeError(
            f"Ming-Image GGUF only supports axis-0 QKV merges, got {axes}"
        )

    cursor = 0
    for offset, _value in parts:
        _axis, start, length = map(int, offset)
        if start != cursor:
            raise RuntimeError(
                f"Ming-Image GGUF non-contiguous merge for {target}: "
                f"expected start {cursor}, got {start}"
            )
        cursor += length

    values = [item[1] for item in parts]
    quantized = [is_quantized(value) for value in values]

    if any(quantized):
        if not all(quantized):
            raise RuntimeError(
                f"Ming-Image GGUF mixed quantized/unquantized QKV for {target}"
            )

        qtypes = [getattr(value, "tensor_type", None) for value in values]
        if not all(qtype == qtypes[0] for qtype in qtypes):
            raise RuntimeError(
                f"Ming-Image GGUF Q/K/V qtypes differ for {target}: {qtypes}"
            )

        logical_shapes = [
            tuple(int(x) for x in value.tensor_shape) for value in values
        ]

        if any(len(shape) != 2 for shape in logical_shapes):
            raise RuntimeError(
                f"Ming-Image GGUF expected 2D Q/K/V tensors for {target}: "
                f"{logical_shapes}"
            )

        input_dim = logical_shapes[0][1]
        if any(shape[1] != input_dim for shape in logical_shapes):
            raise RuntimeError(
                f"Ming-Image GGUF Q/K/V input dims differ for {target}: "
                f"{logical_shapes}"
            )

        packed = [_base_tensor(value) for value in values]
        packed_tail = tuple(packed[0].size()[1:])

        if any(tuple(value.size()[1:]) != packed_tail for value in packed):
            raise RuntimeError(
                f"Ming-Image GGUF packed Q/K/V row widths differ for {target}: "
                f"{[tuple(value.size()) for value in packed]}"
            )

        merged_data = torch.cat(packed, dim=0)
        merged_shape = torch.Size(
            (sum(shape[0] for shape in logical_shapes), input_dim)
        )

        merged = GGMLTensor(
            merged_data,
            tensor_type=qtypes[0],
            tensor_shape=merged_shape,
        )

        if any(getattr(value, "is_largest_weight", False) for value in values):
            merged.is_largest_weight = True

        return merged

    return torch.cat([_base_tensor(value) for value in values], dim=0)


def convert_ming_image_diffusers_gguf(state_dict):
    """Convert Ming-Image Diffusers keys without unpacking GGUF tensors."""

    required = (
        "all_x_embedder.2-1.weight",
        "all_final_layer.2-1.linear.weight",
        "noise_refiner.0.attention.to_q.weight",
        "noise_refiner.0.attention.to_k.weight",
        "noise_refiner.0.attention.to_v.weight",
        "layers.0.attention.to_q.weight",
    )

    if not all(key in state_dict for key in required):
        return state_dict

    keys = list(state_dict.keys())
    n_layers = _count_blocks(keys, "layers.{}.")

    if n_layers <= 0:
        raise RuntimeError(
            "Ming-Image GGUF could not determine transformer layer count"
        )

    dim = int(state_dict["noise_refiner.0.attention.to_k.weight"].shape[0])

    # Ming currently uses ComfyUI's Z-Image Diffusers key mapping internally.
    state_dict_map = comfy.utils.z_image_to_diffusers(
        {"n_layers": n_layers, "dim": dim},
        output_prefix="",
    )

    # Preserve Ming-specific keys not handled by the shared mapping.
    for key in keys:
        state_dict_map.setdefault(key, key)

    output = {}
    pending = {}
    merged_count = 0

    for key in list(state_dict.keys()):
        value = state_dict.pop(key)
        target = state_dict_map.get(key, key)

        if isinstance(target, str):
            output[target] = value
            continue

        transform = target[2] if len(target) > 2 else (lambda x: x)
        offset = target[1]
        destination = target[0]

        if offset is None:
            output[destination] = transform(value)
            continue

        bucket = pending.setdefault(destination, [])
        bucket.append((offset, transform(value)))

        if len(bucket) == 3:
            output[destination] = _merge_offset_parts(destination, bucket)
            del pending[destination]
            merged_count += 1

    if pending:
        detail = {key: len(value) for key, value in list(pending.items())[:8]}
        raise RuntimeError(
            f"Ming-Image GGUF incomplete QKV groups remain: {detail}"
        )

    logging.info(
        "ComfyUI-GGUF: Ming-Image quant-safe conversion complete "
        f"({merged_count} packed QKV groups merged, {n_layers} main layers)"
    )

    return output
