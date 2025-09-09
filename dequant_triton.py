import torch

import triton
import triton.language as tl

import gguf

TORCH_TO_TRITON_DTYPE_MAP = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

# K Quants #
QK_K = 256
K_SCALE_SIZE = 12


@triton.autotune(
    configs=[
        # Test different numbers of GGUF blocks per program instance
        triton.Config({"N_BLOCKS_PER_PROG": 1}, num_warps=4),
        triton.Config({"N_BLOCKS_PER_PROG": 2}, num_warps=4),
        triton.Config({"N_BLOCKS_PER_PROG": 4}, num_warps=4),
        triton.Config({"N_BLOCKS_PER_PROG": 8}, num_warps=4),
        triton.Config({"N_BLOCKS_PER_PROG": 1}, num_warps=8),
        triton.Config({"N_BLOCKS_PER_PROG": 2}, num_warps=8),
        triton.Config({"N_BLOCKS_PER_PROG": 4}, num_warps=8),
        triton.Config({"N_BLOCKS_PER_PROG": 8}, num_warps=8),
    ],
    key=["n_total_blocks"],  # Tune based on the total number of blocks
)
@triton.jit
def dequantize_q8_0_kernel(
    q_tensor_ptr,
    out_tensor_ptr,
    n_total_blocks,
    GGUF_BLOCK_SIZE: tl.constexpr,
    GGUF_TYPE_SIZE: tl.constexpr,
    N_BLOCKS_PER_PROG: tl.constexpr,  # How many blocks each program handles
    OUT_DTYPE: tl.constexpr,
):
    out_dtype = OUT_DTYPE.value

    # Each program is responsible for a chunk of N_BLOCKS_PER_PROG blocks
    pid = tl.program_id(axis=0)

    # Starting GGUF block index for this program
    start_block_idx = pid * N_BLOCKS_PER_PROG

    # Create offsets for the weights within a GGUF block (0, 1, ..., 31)
    weight_indices = tl.arange(0, GGUF_BLOCK_SIZE)

    # Loop over the N blocks assigned to this program
    for i in tl.static_range(N_BLOCKS_PER_PROG):
        current_block_idx = start_block_idx + i

        # Boundary check to avoid processing padding blocks
        if current_block_idx < n_total_blocks:
            # Pointer to the start of the current GGUF block in the input tensor
            block_start_ptr = q_tensor_ptr + current_block_idx * GGUF_TYPE_SIZE

            # Load scale (d)
            uint16_ptr = block_start_ptr.to(tl.pointer_type(tl.uint16))
            uint16_val = tl.load(uint16_ptr)
            scale_fp16 = tl.cast(uint16_val, tl.float16, bitcast=True)
            scale = scale_fp16.to(out_dtype)

            # Load weights (x)
            q_weights_ptr = block_start_ptr + 2
            uint8_weights = tl.load(q_weights_ptr + weight_indices)
            q_weights = uint8_weights.to(tl.int8)

            # Dequantize
            dequantized_weights = q_weights.to(out_dtype) * scale

            # Store the result
            output_start_ptr = out_tensor_ptr + current_block_idx * GGUF_BLOCK_SIZE
            tl.store(output_start_ptr + weight_indices, dequantized_weights)


def dequantize_blocks_Q8_0_triton(
    blocks: torch.Tensor,
    block_size: int,
    type_size: int,
    dtype=None,
) -> torch.Tensor:
    assert blocks.dtype == torch.uint8 and blocks.is_cuda and blocks.is_contiguous()

    n_elements = blocks.numel()
    assert n_elements % type_size == 0
    n_total_blocks = n_elements // type_size

    dtype = dtype or torch.float32
    triton_dtype = TORCH_TO_TRITON_DTYPE_MAP.get(dtype)
    if triton_dtype is None:
        raise TypeError(f"Unsupported output dtype {dtype}")

    out_tensor = torch.empty(
        (n_total_blocks * block_size,),
        dtype=dtype,
        device=blocks.device,
    )

    def grid(meta):
        return (triton.cdiv(n_total_blocks, meta["N_BLOCKS_PER_PROG"]),)

    dequantize_q8_0_kernel[grid](
        blocks,
        out_tensor,
        n_total_blocks,
        GGUF_BLOCK_SIZE=block_size,
        GGUF_TYPE_SIZE=type_size,
        OUT_DTYPE=triton_dtype,
    )

    return out_tensor.reshape(n_total_blocks, -1)


@triton.autotune(
    configs=[
        triton.Config({"N_BLOCKS_PER_PROG": 1}, num_warps=4),
        triton.Config({"N_BLOCKS_PER_PROG": 2}, num_warps=4),
        triton.Config({"N_BLOCKS_PER_PROG": 4}, num_warps=4),
        triton.Config({"N_BLOCKS_PER_PROG": 1}, num_warps=8),
        triton.Config({"N_BLOCKS_PER_PROG": 2}, num_warps=8),
        triton.Config({"N_BLOCKS_PER_PROG": 4}, num_warps=8),
    ],
    key=["n_total_blocks"],
)
@triton.jit
def dequantize_q4_k_kernel(
    q_tensor_ptr,
    out_tensor_ptr,
    n_total_blocks,
    QK_K: tl.constexpr,
    TYPE_SIZE: tl.constexpr,
    K_SCALE_SIZE: tl.constexpr,
    N_BLOCKS_PER_PROG: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    out_dtype = OUT_DTYPE.value
    pid = tl.program_id(axis=0)
    start_block_idx = pid * N_BLOCKS_PER_PROG

    qs_chunk_offsets = tl.arange(0, 32)
    store_offsets = tl.arange(0, 32)

    for i in tl.static_range(N_BLOCKS_PER_PROG):
        current_block_idx = start_block_idx + i
        if current_block_idx < n_total_blocks:
            block_start_ptr = q_tensor_ptr + current_block_idx * TYPE_SIZE
            output_start_ptr = out_tensor_ptr + current_block_idx * QK_K

            d = tl.load(block_start_ptr.to(tl.pointer_type(tl.float16))).to(out_dtype)
            dmin = tl.load((block_start_ptr + 2).to(tl.pointer_type(tl.float16))).to(
                out_dtype
            )

            scales_ptr_u32 = (block_start_ptr + 4).to(tl.pointer_type(tl.uint32))
            d_sc_word = tl.load(scales_ptr_u32 + 0)
            m_word = tl.load(scales_ptr_u32 + 1)
            m_sc_word = tl.load(scales_ptr_u32 + 2)

            qs_start_ptr = block_start_ptr + 4 + K_SCALE_SIZE

            # Process in 4 chunks of 64 values
            for k_chunk in tl.static_range(4):
                # Scale indices for low (a) and high (b) nibbles
                k_idx_a = 2 * k_chunk
                k_idx_b = 2 * k_chunk + 1

                # --- Calculate Scale A (for low nibbles) ---
                if k_idx_a < 4:
                    d_sc_byte_a = (d_sc_word >> (k_idx_a * 8)) & 0xFF
                    m_byte_a = (m_word >> (k_idx_a * 8)) & 0xFF
                    sc_a = d_sc_byte_a & 0x3F
                    m_a = m_byte_a & 0x3F
                else:
                    k_prime_a = k_idx_a - 4
                    d_sc_byte_a = (d_sc_word >> (k_prime_a * 8)) & 0xFF
                    m_byte_a = (m_word >> (k_prime_a * 8)) & 0xFF
                    m_sc_byte_a = (m_sc_word >> (k_prime_a * 8)) & 0xFF
                    sc_a = (m_sc_byte_a & 0x0F) | ((d_sc_byte_a >> 2) & 0x30)
                    m_a = (m_sc_byte_a >> 4) | ((m_byte_a >> 2) & 0x30)

                # --- Calculate Scale B (for high nibbles) ---
                if k_idx_b < 4:
                    d_sc_byte_b = (d_sc_word >> (k_idx_b * 8)) & 0xFF
                    m_byte_b = (m_word >> (k_idx_b * 8)) & 0xFF
                    sc_b = d_sc_byte_b & 0x3F
                    m_b = m_byte_b & 0x3F
                else:
                    k_prime_b = k_idx_b - 4
                    d_sc_byte_b = (d_sc_word >> (k_prime_b * 8)) & 0xFF
                    m_byte_b = (m_word >> (k_prime_b * 8)) & 0xFF
                    m_sc_byte_b = (m_sc_word >> (k_prime_b * 8)) & 0xFF
                    sc_b = (m_sc_byte_b & 0x0F) | ((d_sc_byte_b >> 2) & 0x30)
                    m_b = (m_sc_byte_b >> 4) | ((m_byte_b >> 2) & 0x30)

                current_d_a = d * sc_a.to(out_dtype)
                current_dm_a = dmin * m_a.to(out_dtype)
                current_d_b = d * sc_b.to(out_dtype)
                current_dm_b = dmin * m_b.to(out_dtype)

                # Load 32 bytes of quantized data
                chunk_qs_ptr = qs_start_ptr + k_chunk * 32
                qs_bytes_chunk = tl.load(chunk_qs_ptr + qs_chunk_offsets)

                qs_low = (qs_bytes_chunk & 0x0F).to(out_dtype)
                qs_high = (qs_bytes_chunk >> 4).to(out_dtype)

                dequant_low = current_d_a * qs_low - current_dm_a
                dequant_high = current_d_b * qs_high - current_dm_b

                # Store results contiguously
                output_chunk_ptr = output_start_ptr + k_chunk * 64
                tl.store(output_chunk_ptr + store_offsets, dequant_low)
                tl.store(output_chunk_ptr + 32 + store_offsets, dequant_high)


def dequantize_blocks_Q4_K_triton(
    blocks: torch.Tensor,
    block_size: int,
    type_size: int,
    dtype=None,
) -> torch.Tensor:
    assert blocks.dtype == torch.uint8 and blocks.is_cuda and blocks.is_contiguous()

    n_elements = blocks.numel()
    assert n_elements % type_size == 0
    n_total_blocks = n_elements // type_size

    dtype = dtype or torch.float32
    triton_dtype = TORCH_TO_TRITON_DTYPE_MAP.get(dtype)
    if triton_dtype is None:
        raise TypeError(f"Unsupported output dtype {dtype}")

    out_tensor = torch.empty(
        (n_total_blocks * QK_K,), dtype=dtype, device=blocks.device
    )

    def grid(meta):
        return (triton.cdiv(n_total_blocks, meta["N_BLOCKS_PER_PROG"]),)

    dequantize_q4_k_kernel[grid](
        blocks,
        out_tensor,
        n_total_blocks,
        QK_K=QK_K,
        TYPE_SIZE=type_size,
        K_SCALE_SIZE=K_SCALE_SIZE,
        OUT_DTYPE=triton_dtype,
    )

    return out_tensor.reshape(n_total_blocks, -1)


@triton.autotune(
    configs=[
        triton.Config({"N_BLOCKS_PER_PROG": 1}, num_warps=4),
        triton.Config({"N_BLOCKS_PER_PROG": 2}, num_warps=4),
        triton.Config({"N_BLOCKS_PER_PROG": 4}, num_warps=4),
        triton.Config({"N_BLOCKS_PER_PROG": 1}, num_warps=8),
        triton.Config({"N_BLOCKS_PER_PROG": 2}, num_warps=8),
        triton.Config({"N_BLOCKS_PER_PROG": 4}, num_warps=8),
    ],
    key=["n_total_blocks"],
)
@triton.jit
def dequantize_q5_k_kernel(
    q_tensor_ptr,
    out_tensor_ptr,
    n_total_blocks,
    QK_K: tl.constexpr,
    TYPE_SIZE: tl.constexpr,
    K_SCALE_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    N_BLOCKS_PER_PROG: tl.constexpr,
):
    out_dtype = OUT_DTYPE.value
    pid = tl.program_id(axis=0)
    start_block_idx = pid * N_BLOCKS_PER_PROG

    offsets_32 = tl.arange(0, 32)

    for i in tl.static_range(N_BLOCKS_PER_PROG):
        current_block_idx = start_block_idx + i
        if current_block_idx < n_total_blocks:
            # Pointers and initial loads
            block_start_ptr = q_tensor_ptr + current_block_idx * TYPE_SIZE
            output_start_ptr = out_tensor_ptr + current_block_idx * QK_K
            d = tl.load(block_start_ptr.to(tl.pointer_type(tl.float16))).to(out_dtype)
            dmin = tl.load((block_start_ptr + 2).to(tl.pointer_type(tl.float16))).to(
                out_dtype
            )

            scales_ptr_u32 = (block_start_ptr + 4).to(tl.pointer_type(tl.uint32))
            d_sc_word = tl.load(scales_ptr_u32 + 0)
            m_word = tl.load(scales_ptr_u32 + 1)
            m_sc_word = tl.load(scales_ptr_u32 + 2)

            qh_start_ptr = block_start_ptr + 4 + K_SCALE_SIZE
            qs_start_ptr = qh_start_ptr + QK_K // 8

            qh_bytes_all = tl.load(qh_start_ptr + offsets_32)

            # Process in 8 chunks of 32 values
            for chunk_idx in tl.static_range(8):
                # 1. Unpack scale and min for this chunk
                if chunk_idx < 4:
                    sc = ((d_sc_word >> (chunk_idx * 8)) & 0xFF) & 0x3F
                    m = ((m_word >> (chunk_idx * 8)) & 0xFF) & 0x3F
                else:
                    k_prime = chunk_idx - 4
                    d_sc_byte = (d_sc_word >> (k_prime * 8)) & 0xFF
                    m_byte = (m_word >> (k_prime * 8)) & 0xFF
                    m_sc_byte = (m_sc_word >> (k_prime * 8)) & 0xFF
                    sc = (m_sc_byte & 0x0F) | ((d_sc_byte >> 2) & 0x30)
                    m = (m_sc_byte >> 4) | ((m_byte >> 2) & 0x30)

                final_d = d * sc.to(out_dtype)
                final_dm = dmin * m.to(out_dtype)

                # 2. Unpack ql (lower 4 bits) for this chunk
                qs_byte_offset = (chunk_idx // 2) * 32
                qs_bytes = tl.load(qs_start_ptr + qs_byte_offset + offsets_32)
                use_low_nibbles = chunk_idx % 2 == 0
                ql = tl.where(use_low_nibbles, qs_bytes & 0x0F, qs_bytes >> 4)

                # 3. Unpack qh (higher 1 bit) for this chunk
                qh = (qh_bytes_all >> chunk_idx) & 0x01

                # 4. Combine, dequantize, and store
                q = ql.to(tl.uint8) | (qh.to(tl.uint8) << 4)
                dequant_32 = final_d * q.to(out_dtype) - final_dm

                output_ptr = output_start_ptr + chunk_idx * 32
                tl.store(output_ptr + offsets_32, dequant_32)


def dequantize_blocks_Q5_K_triton(
    blocks: torch.Tensor,
    block_size: int,
    type_size: int,
    dtype=None,
) -> torch.Tensor:
    assert blocks.dtype == torch.uint8 and blocks.is_cuda and blocks.is_contiguous()

    n_elements = blocks.numel()
    assert n_elements % type_size == 0
    n_total_blocks = n_elements // type_size

    dtype = dtype or torch.float32
    triton_dtype = TORCH_TO_TRITON_DTYPE_MAP.get(dtype)
    if triton_dtype is None:
        raise TypeError(f"Unsupported output dtype {dtype}")

    out_tensor = torch.empty(
        (n_total_blocks * QK_K,), dtype=dtype, device=blocks.device
    )

    def grid(meta):
        return (triton.cdiv(n_total_blocks, meta["N_BLOCKS_PER_PROG"]),)

    dequantize_q5_k_kernel[grid](
        blocks,
        out_tensor,
        n_total_blocks,
        QK_K=QK_K,
        TYPE_SIZE=type_size,
        K_SCALE_SIZE=K_SCALE_SIZE,
        OUT_DTYPE=triton_dtype,
    )

    return out_tensor.reshape(n_total_blocks, -1)


@triton.autotune(
    configs=[
        triton.Config({"N_BLOCKS_PER_PROG": 1}, num_warps=4),
        triton.Config({"N_BLOCKS_PER_PROG": 2}, num_warps=4),
        triton.Config({"N_BLOCKS_PER_PROG": 4}, num_warps=4),
        triton.Config({"N_BLOCKS_PER_PROG": 1}, num_warps=8),
        triton.Config({"N_BLOCKS_PER_PROG": 2}, num_warps=8),
        triton.Config({"N_BLOCKS_PER_PROG": 4}, num_warps=8),
    ],
    key=["n_total_blocks"],
)
@triton.jit
def dequantize_q6_k_kernel(
    q_tensor_ptr,
    out_tensor_ptr,
    n_total_blocks,
    QK_K: tl.constexpr,
    TYPE_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    N_BLOCKS_PER_PROG: tl.constexpr,
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
            output_start_ptr = out_tensor_ptr + current_block_idx * QK_K

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


def dequantize_blocks_Q6_K_triton(
    blocks: torch.Tensor,
    block_size: int,
    type_size: int,
    dtype=None,
) -> torch.Tensor:
    assert blocks.dtype == torch.uint8 and blocks.is_cuda and blocks.is_contiguous()

    n_elements = blocks.numel()
    assert n_elements % type_size == 0
    n_total_blocks = n_elements // type_size

    dtype = dtype or torch.float32
    triton_dtype = TORCH_TO_TRITON_DTYPE_MAP.get(dtype)
    if triton_dtype is None:
        raise TypeError(f"Unsupported output dtype {dtype}")

    out_tensor = torch.empty(
        (n_total_blocks * QK_K,), dtype=dtype, device=blocks.device
    )

    def grid(meta):
        return (triton.cdiv(n_total_blocks, meta["N_BLOCKS_PER_PROG"]),)

    dequantize_q6_k_kernel[grid](
        blocks,
        out_tensor,
        n_total_blocks,
        QK_K=QK_K,
        TYPE_SIZE=type_size,
        OUT_DTYPE=triton_dtype,
    )

    return out_tensor.reshape(n_total_blocks, -1)


dequantize_functions = {
    # Q8_0 simply seems than the PyTorch implementation.
    # gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0_triton,
    gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K_triton,
    gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K_triton,
    gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K_triton,
}

__all__ = ("dequantize_functions",)
