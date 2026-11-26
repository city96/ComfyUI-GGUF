"""Triton bit decoding and BF16-activation execution for Q1_CR_XL."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


CONVROT_GROUP_SIZE = 256
QUANT_GROUP_SIZE = 64


def _build_regular_hadamard(size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if size < 4 or (size & (size - 1)) != 0 or not math.log(size, 4).is_integer():
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


def _validate_q1_tensors(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    orig_shape: tuple[int, int] | None,
    quant_group_size: int,
) -> tuple[int, int]:
    if qweight.ndim != 2 or qweight.dtype not in {torch.int8, torch.uint8}:
        raise ValueError("Q1_CR_XL packed weights must be a 2D int8/uint8 tensor")
    if scales.ndim != 2 or scales.dtype not in {torch.float16, torch.float32}:
        raise ValueError("Q1_CR_XL scales must be a 2D FP16/FP32 tensor")
    if not isinstance(quant_group_size, int) or isinstance(quant_group_size, bool) or quant_group_size <= 0:
        raise ValueError("Q1_CR_XL quant_group_size must be a positive integer")

    n, k_bytes = qweight.shape
    if n <= 0 or k_bytes <= 0:
        raise ValueError("Q1_CR_XL packed weights must have non-empty dimensions")
    k = k_bytes * 8
    if orig_shape is not None and tuple(orig_shape) != (n, k):
        raise ValueError(
            f"Q1_CR_XL packed shape {(n, k)} does not match metadata {tuple(orig_shape)}"
        )
    if k % quant_group_size:
        raise ValueError(
            f"Q1_CR_XL K={k} must be divisible by quant_group_size={quant_group_size}"
        )
    if quant_group_size % 8:
        raise ValueError("Q1_CR_XL quant_group_size must be divisible by 8")
    if tuple(scales.shape) != (n, k // quant_group_size):
        raise ValueError(
            f"Q1_CR_XL scales have shape {tuple(scales.shape)}, expected "
            f"{(n, k // quant_group_size)}"
        )
    return n, k


if triton is not None:

    @triton.jit
    def _q1_cr_decode_kernel(
        qweight_ptr,
        scales_ptr,
        output_ptr,
        n_bytes,
        bytes_per_row,
        group_bytes: tl.constexpr,
        groups_per_row: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        BLOCK_BYTES: tl.constexpr,
    ):
        byte_index = tl.program_id(0) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
        valid = byte_index < n_bytes
        row = byte_index // bytes_per_row
        col_byte = byte_index % bytes_per_row
        group = col_byte // group_bytes

        packed = tl.load(qweight_ptr + byte_index, mask=valid, other=0).to(tl.int32) & 0xFF
        scale = tl.load(
            scales_ptr + row * groups_per_row + group,
            mask=valid,
            other=0.0,
        ).to(OUT_DTYPE)
        shifts = tl.arange(0, 8)[None, :]
        bits = (packed[:, None] >> shifts) & 1
        values = tl.where(bits != 0, scale[:, None], -scale[:, None])

        output_index = byte_index * 8
        offsets = output_index[:, None] + tl.arange(0, 8)[None, :]
        tl.store(output_ptr + offsets, values, mask=valid[:, None])


def unpack_int1_sign_bits(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    quant_group_size: int = QUANT_GROUP_SIZE,
) -> torch.Tensor:
    """Decode packed sign weights into the rotated dense matrix."""
    n, k = _validate_q1_tensors(qweight, scales, None, quant_group_size)
    qweight = qweight.contiguous()
    scales = scales.to(device=qweight.device, dtype=dtype).contiguous()
    output = torch.empty((n, k), device=qweight.device, dtype=dtype)

    if qweight.device.type == "cuda" and triton is not None:
        dtype_map = {
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
            torch.float32: tl.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported Q1_CR_XL decode dtype: {dtype}")
        qbytes = qweight.view(torch.uint8).reshape(-1)
        n_bytes = qbytes.numel()
        block_bytes = 256
        _q1_cr_decode_kernel[(triton.cdiv(n_bytes, block_bytes),)](
            qbytes,
            scales,
            output,
            n_bytes,
            k // 8,
            quant_group_size // 8,
            k // quant_group_size,
            OUT_DTYPE=dtype_map[dtype],
            BLOCK_BYTES=block_bytes,
            num_warps=4,
            num_stages=1,
        )
        return output

    qbytes = qweight.view(torch.uint8)
    shifts = torch.arange(8, device=qweight.device, dtype=torch.int32)
    output_elements_per_chunk = 4 * 1024 * 1024
    rows_per_chunk = max(1, output_elements_per_chunk // k)
    output = torch.empty((n, k), device=qweight.device, dtype=dtype)
    for row_start in range(0, n, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, n)
        packed_chunk = qbytes[row_start:row_end].to(torch.int32)
        bits = (packed_chunk.unsqueeze(-1) >> shifts) & 1
        signs = bits.to(dtype).mul_(2).sub_(1).reshape(row_end - row_start, k)
        scales_chunk = scales[row_start:row_end].unsqueeze(-1)
        output[row_start:row_end] = (
            signs.reshape(row_end - row_start, k // quant_group_size, quant_group_size)
            * scales_chunk
        ).reshape(row_end - row_start, k)
    return output


def pack_int1_sign_bits(
    weight_rotated: torch.Tensor,
    quant_group_size: int = QUANT_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack rotated floating weights as little-endian 1-bit signs and FP16 scales."""
    if weight_rotated.ndim != 2 or not weight_rotated.is_floating_point():
        raise ValueError("Q1_CR_XL quantization expects a 2D floating-point weight")
    n, k = weight_rotated.shape
    if (
        not isinstance(quant_group_size, int)
        or isinstance(quant_group_size, bool)
        or quant_group_size <= 0
        or quant_group_size % 8
        or k % quant_group_size
    ):
        raise ValueError(
            f"Q1_CR_XL K={k} must be divisible by a positive quant_group_size "
            "that is itself divisible by 8"
        )

    grouped = weight_rotated.to(torch.float32).reshape(n, k // quant_group_size, quant_group_size)
    scales = grouped.abs().mean(dim=-1).to(torch.float16)
    positive = (grouped >= 0).reshape(n, k // 8, 8).to(torch.uint8)
    shifts = torch.arange(8, device=weight_rotated.device, dtype=torch.uint8)
    packed = (positive << shifts).sum(dim=-1).to(torch.uint8).view(torch.int8)
    return packed.contiguous(), scales.contiguous()


def _rotate_activation(x: torch.Tensor, convrot_group_size: int) -> torch.Tensor:
    if (
        not isinstance(convrot_group_size, int)
        or isinstance(convrot_group_size, bool)
        or convrot_group_size <= 0
        or x.shape[-1] % convrot_group_size
    ):
        raise ValueError(
            f"Q1_CR_XL input K={x.shape[-1]} must be divisible by "
            f"convrot_group_size={convrot_group_size}"
        )
    original_shape = x.shape
    x2d = x.reshape(-1, original_shape[-1]).contiguous()
    if x2d.device.type == "cuda":
        try:
            from comfy_kitchen.backends.cuda import rotate_int8_convrot_weight
        except ImportError:
            rotate_int8_convrot_weight = None
        if rotate_int8_convrot_weight is not None:
            return rotate_int8_convrot_weight(x2d, convrot_group_size).reshape(original_shape)

    hadamard = _build_regular_hadamard(
        convrot_group_size,
        dtype=torch.float32,
        device=x2d.device,
    )
    rotated = (
        x2d.to(torch.float32)
        .reshape(x2d.shape[0], -1, convrot_group_size)
        @ hadamard
    ).reshape_as(x2d)
    return rotated.to(x.dtype).reshape(original_shape)


def dequantize_q1_cr_xl_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    orig_shape: tuple[int, int],
    *,
    dtype: torch.dtype = torch.bfloat16,
    convrot_group_size: int = CONVROT_GROUP_SIZE,
    quant_group_size: int = QUANT_GROUP_SIZE,
) -> torch.Tensor:
    """Decode Q1_CR_XL weights and rotate them back to the original basis."""
    n, k = _validate_q1_tensors(qweight, scales, orig_shape, quant_group_size)
    if (
        not isinstance(convrot_group_size, int)
        or isinstance(convrot_group_size, bool)
        or convrot_group_size <= 0
        or k % convrot_group_size
    ):
        raise ValueError(
            f"Q1_CR_XL K={k} must be divisible by convrot_group_size={convrot_group_size}"
        )
    rotated = unpack_int1_sign_bits(
        qweight,
        scales,
        dtype=dtype,
        quant_group_size=quant_group_size,
    )
    hadamard = _build_regular_hadamard(
        convrot_group_size,
        dtype=torch.float32,
        device=rotated.device,
    )
    return (
        rotated.to(torch.float32)
        .reshape(n, k // convrot_group_size, convrot_group_size)
        .matmul(hadamard)
        .reshape(n, k)
        .to(dtype)
    )


def q1_cr_xl_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    orig_shape: tuple[int, int] | None = None,
    convrot_group_size: int = CONVROT_GROUP_SIZE,
    quant_group_size: int = QUANT_GROUP_SIZE,
) -> torch.Tensor:
    """Run W1A16 linear using Triton sign decoding followed by a BF16/FP16 GEMM."""
    n, k = _validate_q1_tensors(qweight, scales, orig_shape, quant_group_size)
    if (
        not isinstance(convrot_group_size, int)
        or isinstance(convrot_group_size, bool)
        or convrot_group_size <= 0
        or k % convrot_group_size
    ):
        raise ValueError(
            f"Q1_CR_XL K={k} must be divisible by convrot_group_size={convrot_group_size}"
        )
    if x.shape[-1] != k:
        raise ValueError(f"Input K={x.shape[-1]} does not match Q1_CR_XL K={k}")
    if x.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"Unsupported Q1_CR_XL activation dtype: {x.dtype}")

    original_shape = x.shape
    x2d = x.reshape(-1, k).contiguous()
    qweight = qweight.to(device=x.device)
    scales = scales.to(device=x.device)
    x_rotated = _rotate_activation(x2d, convrot_group_size).contiguous()
    weight_rotated = unpack_int1_sign_bits(
        qweight,
        scales,
        dtype=x.dtype,
        quant_group_size=quant_group_size,
    )
    bias_arg = bias.to(device=x.device, dtype=x.dtype) if bias is not None else None
    output = torch.nn.functional.linear(x_rotated, weight_rotated, bias_arg)
    return output.reshape(*original_shape[:-1], n)


__all__ = [
    "CONVROT_GROUP_SIZE",
    "QUANT_GROUP_SIZE",
    "dequantize_q1_cr_xl_weight",
    "pack_int1_sign_bits",
    "q1_cr_xl_linear",
    "unpack_int1_sign_bits",
]
