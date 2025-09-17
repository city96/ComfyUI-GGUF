from typing import Callable, NamedTuple

import torch

import triton
import triton.language as tl

from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

GQT = GGMLQuantizationType

K_SCALE_SIZE = 12

TORCH_TO_TRITON_DTYPE_MAP: dict[torch.dtype, tl.dtype] = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

_DEFAULT_AUTOTUNE_CONFIGS: list[triton.Config] = [
    triton.Config({"N_BLOCKS_PER_PROG": 1}, num_warps=4),
    triton.Config({"N_BLOCKS_PER_PROG": 2}, num_warps=4),
    triton.Config({"N_BLOCKS_PER_PROG": 4}, num_warps=4),
    triton.Config({"N_BLOCKS_PER_PROG": 1}, num_warps=8),
    triton.Config({"N_BLOCKS_PER_PROG": 2}, num_warps=8),
    triton.Config({"N_BLOCKS_PER_PROG": 4}, num_warps=8),
]

_AUTOTUNE_CONFIGS: dict[str, list[triton.Config]] = {}


class KernelDefinition(NamedTuple):
    qtype: GGMLQuantizationType
    kernel: triton.runtime.jit.JITFunction
    block_size: int
    type_size: int
    kernel_kwargs: dict

    @classmethod
    def build(
        cls,
        qtype: GGMLQuantizationType,
        kernel: triton.runtime.jit.JITFunction,
        **kwargs,
    ) -> NamedTuple:
        block_size, type_size = GGML_QUANT_SIZES[qtype]
        return cls(
            qtype=qtype,
            kernel=kernel,
            block_size=block_size,
            type_size=type_size,
            kernel_kwargs=kwargs,
        )

    def __call__(
        self,
        blocks: torch.Tensor,
        block_size: int,
        type_size: int,
        dtype=None,
    ) -> torch.Tensor:
        qtype, ggml_type_size = self.qtype, self.type_size
        if blocks.dtype != torch.uint8:
            raise ValueError(
                f"GGUF Triton {qtype.name}: Blocks tensor dtype must be uint8 but got {blocks.dtype}"
            )
        if not blocks.is_cuda:
            raise ValueError(f"GGUF Triton {qtype.name}: Blocks tensor must be CUDA")
        if not blocks.is_contiguous():
            raise ValueError(
                f"GGUF Triton {qtype.name}: Blocks tensor must be contiguous"
            )

        n_elements = blocks.numel()
        if n_elements % ggml_type_size != 0:
            raise ValueError(
                f"GGUF Triton {qtype.name}: Blocks tensor must have a number of elements ({n_elements}) divisible by the type size {ggml_type_size}"
            )
        n_total_blocks = n_elements // ggml_type_size

        dtype = dtype or torch.float32
        if (triton_dtype := TORCH_TO_TRITON_DTYPE_MAP.get(dtype)) is None:
            raise TypeError(
                f"GGUF Triton {qtype.name}: Unsupported output dtype {dtype}"
            )

        out_tensor = torch.empty(
            n_total_blocks * self.block_size, dtype=dtype, device=blocks.device
        )

        def grid(meta: dict) -> tuple[int]:
            return (triton.cdiv(n_total_blocks, meta["N_BLOCKS_PER_PROG"]),)

        self.kernel[grid](
            blocks,
            out_tensor,
            n_total_blocks,
            BLOCK_SIZE=self.block_size,
            TYPE_SIZE=ggml_type_size,
            OUT_DTYPE=triton_dtype,
            **self.kernel_kwargs,
        )

        return out_tensor


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS.get("q3_k", _DEFAULT_AUTOTUNE_CONFIGS),
    key=["n_total_blocks"],
)
@triton.jit
def dequantize_Q3_K_kernel(
    q_tensor_ptr,
    out_tensor_ptr,
    n_total_blocks,
    OUT_DTYPE: tl.constexpr,
    N_BLOCKS_PER_PROG: tl.constexpr,
    TYPE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    out_dtype = OUT_DTYPE.value
    pid = tl.program_id(axis=0)
    start_block_idx = pid * N_BLOCKS_PER_PROG

    # Vector of offsets for a 16-element chunk (one row of the output matrix)
    offsets_16 = tl.arange(0, 16)

    for i in tl.static_range(N_BLOCKS_PER_PROG):
        current_block_idx = start_block_idx + i
        if current_block_idx < n_total_blocks:
            # --- Set up pointers for the current block ---
            block_start_ptr = q_tensor_ptr + current_block_idx * TYPE_SIZE
            output_start_ptr = out_tensor_ptr + current_block_idx * BLOCK_SIZE

            hmask_ptr = block_start_ptr
            qs_ptr = block_start_ptr + 32
            scales_ptr = block_start_ptr + 96
            d_ptr = block_start_ptr + 108

            # --- Load the super-scale 'd' ---
            d_super_scale = tl.load(d_ptr.to(tl.pointer_type(tl.float16))).to(out_dtype)

            # --- Process block in 16 chunks of 16 values ---
            for chunk_idx in tl.static_range(16):
                # 1. Unpack the 6-bit scale for this chunk.
                # Low 4 bits of the scale (lscale_nibble) - THIS WAS THE FINAL BUG
                # Python logic: read all 8 low nibbles, then all 8 high nibbles.
                lscale_byte_index = chunk_idx % 8
                lscale_shift = (chunk_idx // 8) * 4
                lscale_byte = tl.load(scales_ptr + lscale_byte_index)
                lscale_nibble = (lscale_byte >> lscale_shift) & 0x0F

                # High 2 bits of the scale (hscale_2bit) - This logic is correct.
                hscale_byte_index = chunk_idx % 4
                hscale_shift_index = chunk_idx // 4
                hscale_byte = tl.load(scales_ptr + 8 + hscale_byte_index)
                hscale_2bit = (hscale_byte >> (hscale_shift_index * 2)) & 0x03

                scale_6bit = lscale_nibble | (hscale_2bit << 4)
                final_scale = d_super_scale * (scale_6bit.to(tl.int8) - 32).to(
                    out_dtype
                )

                # --- Map the 16 output elements to their source data ---
                # This logic correctly models the Python reshape from a flat 256-element array.
                flat_indices = chunk_idx * 16 + offsets_16

                # 2. Unpack ql (lower 2 bits).
                ql_source_row = flat_indices // 32
                ql_source_col = flat_indices % 32

                ql_segment = ql_source_row // 4
                ql_shift_group = ql_source_row % 4

                ql_ptr = qs_ptr + ql_segment * 32 + ql_source_col
                ql_byte = tl.load(ql_ptr)
                ql_vec = ((ql_byte >> (ql_shift_group * 2)) & 3).to(tl.int8)

                # 3. Unpack qh (higher 1 bit, inverted).
                qh_source_row = flat_indices // 32
                qh_source_col = flat_indices % 32

                qh_ptr = hmask_ptr + qh_source_col
                qh_byte = tl.load(qh_ptr)
                qh_vec = (((qh_byte >> qh_source_row) & 1) ^ 1).to(tl.int8)

                # 4. Combine to get the final 3-bit quantized value.
                q_vec = ql_vec - (qh_vec << 2)

                # 5. Dequantize and store the 16 results.
                dequant_16 = final_scale * q_vec.to(out_dtype)
                output_ptr = output_start_ptr + chunk_idx * 16
                tl.store(output_ptr + offsets_16, dequant_16)


@triton.jit
def dequantize_Q4_K_get_scales_min(
    k_idx: int,
    d_sc_word: tl.tensor,
    m_word: tl.tensor,
    m_sc_word: tl.tensor,
) -> tl.tuple:
    if k_idx < 4:
        k_idx_x8 = k_idx * 8
        d_sc_byte = d_sc_word >> k_idx_x8
        m_byte = m_word >> k_idx_x8
        sc = d_sc_byte & 0x3F
        m = m_byte & 0x3F
    else:
        k_prime_x8 = (k_idx - 4) * 8
        d_sc_byte = d_sc_word >> k_prime_x8
        m_byte = m_word >> k_prime_x8
        m_sc_byte = m_sc_word >> k_prime_x8
        sc = (m_sc_byte & 0x0F) | ((d_sc_byte >> 2) & 0x30)
        m = ((m_sc_byte & 0xFF) >> 4) | ((m_byte >> 2) & 0x30)
    return tl.tuple((sc, m))


# Same as Q4_K
dequantize_Q5_K_get_scales_min = dequantize_Q4_K_get_scales_min


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS.get("q4_k", _DEFAULT_AUTOTUNE_CONFIGS),
    key=["n_total_blocks"],
)
@triton.jit
def dequantize_Q4_K_kernel(
    q_tensor_ptr,
    out_tensor_ptr,
    n_total_blocks,
    OUT_DTYPE: tl.constexpr,
    N_BLOCKS_PER_PROG: tl.constexpr,
    TYPE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    K_SCALE_SIZE: tl.constexpr,
):
    out_dtype = OUT_DTYPE.value
    pid = tl.program_id(axis=0)
    start_block_idx = pid * N_BLOCKS_PER_PROG

    offsets_32 = tl.arange(0, 32)
    offsets_scale = offsets_32 + 4 + K_SCALE_SIZE

    for i in tl.static_range(N_BLOCKS_PER_PROG):
        current_block_idx = start_block_idx + i
        if current_block_idx < n_total_blocks:
            block_start_ptr = q_tensor_ptr + current_block_idx * TYPE_SIZE
            output_start_ptr = (
                out_tensor_ptr + current_block_idx * BLOCK_SIZE + offsets_32
            )

            d = tl.load(block_start_ptr.to(tl.pointer_type(tl.float16))).to(out_dtype)
            dmin = tl.load((block_start_ptr + 2).to(tl.pointer_type(tl.float16))).to(
                out_dtype
            )

            scales_ptr_u32 = (block_start_ptr + 4).to(tl.pointer_type(tl.uint32))
            d_sc_word = tl.load(scales_ptr_u32 + 0)
            m_word = tl.load(scales_ptr_u32 + 1)
            m_sc_word = tl.load(scales_ptr_u32 + 2)

            qs_start_ptr = block_start_ptr + offsets_scale

            # Process in 4 chunks of 64 values
            for k_chunk in tl.static_range(4):
                k_idx = 2 * k_chunk

                # --- Get scale A (for low nibbles) ---
                sc_a, m_a = dequantize_Q4_K_get_scales_min(
                    k_idx, d_sc_word, m_word, m_sc_word
                )

                # --- Get scale B (for high nibbles) ---
                sc_b, m_b = dequantize_Q4_K_get_scales_min(
                    k_idx + 1, d_sc_word, m_word, m_sc_word
                )

                current_d_a = d * sc_a.to(out_dtype)
                current_dm_a = dmin * m_a.to(out_dtype)
                current_d_b = d * sc_b.to(out_dtype)
                current_dm_b = dmin * m_b.to(out_dtype)

                # Load 32 bytes of quantized data
                chunk_qs_ptr = qs_start_ptr + k_chunk * 32
                qs_bytes_chunk = tl.load(chunk_qs_ptr)

                qs_low = (qs_bytes_chunk & 0x0F).to(out_dtype)
                qs_high = (qs_bytes_chunk >> 4).to(out_dtype)

                dequant_low = current_d_a * qs_low - current_dm_a
                dequant_high = current_d_b * qs_high - current_dm_b

                # Store results contiguously
                output_chunk_ptr = output_start_ptr + k_chunk * 64
                output_chunk_ptr.store(dequant_low)
                (output_chunk_ptr + 32).store(dequant_high)


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS.get("q5_k", _DEFAULT_AUTOTUNE_CONFIGS),
    key=["n_total_blocks"],
)
@triton.jit
def dequantize_Q5_K_kernel(
    q_tensor_ptr,
    out_tensor_ptr,
    n_total_blocks,
    OUT_DTYPE: tl.constexpr,
    N_BLOCKS_PER_PROG: tl.constexpr,
    TYPE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    K_SCALE_SIZE: tl.constexpr,
):
    out_dtype = OUT_DTYPE.value
    pid = tl.program_id(axis=0)
    start_block_idx = pid * N_BLOCKS_PER_PROG

    offsets_32 = tl.arange(0, 32)
    offsets_scale = offsets_32 + 4 + K_SCALE_SIZE

    for i in tl.static_range(N_BLOCKS_PER_PROG):
        current_block_idx = start_block_idx + i
        if current_block_idx < n_total_blocks:
            # Pointers and initial loads
            block_start_ptr = q_tensor_ptr + current_block_idx * TYPE_SIZE
            output_start_ptr = (
                out_tensor_ptr + current_block_idx * BLOCK_SIZE + offsets_32
            )
            d = tl.load(block_start_ptr.to(tl.pointer_type(tl.float16))).to(out_dtype)
            dmin = tl.load((block_start_ptr + 2).to(tl.pointer_type(tl.float16))).to(
                out_dtype
            )

            scales_ptr_u32 = (block_start_ptr + 4).to(tl.pointer_type(tl.uint32))
            d_sc_word = tl.load(scales_ptr_u32 + 0)
            m_word = tl.load(scales_ptr_u32 + 1)
            m_sc_word = tl.load(scales_ptr_u32 + 2)

            qh_start_ptr = block_start_ptr + offsets_scale
            qs_start_ptr = qh_start_ptr + BLOCK_SIZE // 8

            qh_bytes_all = tl.load(qh_start_ptr)

            # Process in 8 chunks of 32 values
            for chunk_idx in tl.static_range(8):
                # # 1. Unpack scale and min for this chunk
                sc, m = dequantize_Q5_K_get_scales_min(
                    chunk_idx, d_sc_word, m_word, m_sc_word
                )

                final_d = d * sc.to(out_dtype)
                final_dm = dmin * m.to(out_dtype)

                # 2. Unpack ql (lower 4 bits) for this chunk
                qs_byte_offset = (chunk_idx // 2) * 32
                qs_bytes = tl.load(qs_start_ptr + qs_byte_offset)
                use_low_nibbles = chunk_idx % 2 == 0
                ql = tl.where(use_low_nibbles, qs_bytes & 0x0F, qs_bytes >> 4)

                # 3. Unpack qh (higher 1 bit) for this chunk
                qh = (qh_bytes_all >> chunk_idx) & 0x01

                # 4. Combine, dequantize, and store
                q = ql.to(tl.uint8) | (qh.to(tl.uint8) << 4)
                dequant_32 = final_d * q.to(out_dtype) - final_dm

                output_ptr = output_start_ptr + chunk_idx * 32
                output_ptr.store(dequant_32)


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS.get("q6_k", _DEFAULT_AUTOTUNE_CONFIGS),
    key=["n_total_blocks"],
)
@triton.jit
def dequantize_Q6_K_kernel(
    q_tensor_ptr,
    out_tensor_ptr,
    n_total_blocks,
    OUT_DTYPE: tl.constexpr,
    N_BLOCKS_PER_PROG: tl.constexpr,
    TYPE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    out_dtype = OUT_DTYPE.value
    pid = tl.program_id(axis=0)
    start_block_idx = pid * N_BLOCKS_PER_PROG
    offsets_32 = tl.arange(0, 32)
    mask_16 = offsets_32 < 16

    for i in tl.static_range(N_BLOCKS_PER_PROG):
        current_block_idx = start_block_idx + i
        if current_block_idx < n_total_blocks:
            block_start_ptr = q_tensor_ptr + current_block_idx * TYPE_SIZE
            output_start_ptr = out_tensor_ptr + current_block_idx * BLOCK_SIZE

            d_ptr = block_start_ptr + 208
            scales_ptr = block_start_ptr + 192
            d_super_scale = tl.load(d_ptr.to(tl.pointer_type(tl.float16))).to(out_dtype)

            # Process block in 8 chunks of 32 values
            for chunk_idx in tl.static_range(8):
                # 1. Calculate ql source data and unpack
                ql_byte_offset = (chunk_idx % 2) * 32 + (chunk_idx // 4) * 64
                ql_ptr = block_start_ptr + ql_byte_offset
                ql_32_bytes = tl.load(ql_ptr + offsets_32)

                use_low_nibbles = (chunk_idx // 2) % 2 == 0
                if use_low_nibbles:
                    ql_vec_32 = (ql_32_bytes & 0x0F).to(tl.int8)
                else:
                    ql_vec_32 = (ql_32_bytes >> 4).to(tl.int8)

                # 2. Calculate qh source data and unpack
                qh_byte_offset = (chunk_idx // 4) * 32
                qh_ptr = block_start_ptr + 128 + qh_byte_offset
                qh_32_bytes = tl.load(qh_ptr + offsets_32)

                bit_shift = (chunk_idx % 4) * 2
                qh_vec_32 = ((qh_32_bytes >> bit_shift) & 0x03).to(tl.int8)

                # 3. Combine and dequantize
                q_vec_32 = (ql_vec_32 | (qh_vec_32 << 4)) - 32

                # 4. Load and apply correct scales
                scale_0_ptr = scales_ptr + chunk_idx * 2
                scale_1_ptr = scales_ptr + chunk_idx * 2 + 1
                scale_0 = d_super_scale * tl.load(scale_0_ptr).to(tl.int8).to(out_dtype)
                scale_1 = d_super_scale * tl.load(scale_1_ptr).to(tl.int8).to(out_dtype)

                scales_32 = tl.where(mask_16, scale_0, scale_1)
                dequant_32 = q_vec_32.to(out_dtype) * scales_32

                # 5. Store result
                output_ptr = output_start_ptr + chunk_idx * 32
                tl.store(output_ptr + offsets_32, dequant_32)


dequantize_functions = {
    GQT.Q3_K: KernelDefinition.build(GQT.Q3_K, dequantize_Q3_K_kernel),
    GQT.Q4_K: KernelDefinition.build(
        GQT.Q4_K, dequantize_Q4_K_kernel, K_SCALE_SIZE=K_SCALE_SIZE
    ),
    GQT.Q5_K: KernelDefinition.build(
        GQT.Q5_K, dequantize_Q5_K_kernel, K_SCALE_SIZE=K_SCALE_SIZE
    ),
    GQT.Q6_K: KernelDefinition.build(GQT.Q6_K, dequantize_Q6_K_kernel),
}

__all__ = ("dequantize_functions",)
