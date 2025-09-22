from __future__ import annotations

from dataclasses import dataclass, field as dcfield
from typing import Any, Callable, NamedTuple, TypeVar, cast

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


@dataclass
class KernelImpl:
    type_size: tl.constexpr
    block_size: tl.constexpr

    def get_autotuner(self, **kwargs: dict) -> triton.runtime.Autotuner:
        return triton.autotune(**kwargs)(self.dequantize_kernel)

    @staticmethod
    @triton.jit
    def dequantize_kernel(
        q_tensor_ptr,
        out_tensor_ptr,
        n_total_blocks,
        OUT_DTYPE: tl.constexpr,
        N_BLOCKS_PER_PROG: tl.constexpr,
        CTX: tl.constexpr,
    ) -> None:
        pid = tl.program_id(axis=0)
        start_block_idx = pid * N_BLOCKS_PER_PROG

        n_blocks = n_total_blocks - start_block_idx

        if n_blocks > 0:
            block_start_ptr = q_tensor_ptr + start_block_idx * CTX.type_size
            output_start_ptr = out_tensor_ptr + start_block_idx * CTX.block_size

            for i in tl.static_range(N_BLOCKS_PER_PROG):
                if i < n_blocks:
                    # Pointer to the i-th quantized block
                    current_q_ptr = block_start_ptr + i * CTX.type_size
                    # Pointer to the i-th output block
                    current_out_ptr = output_start_ptr + i * CTX.block_size

                    # Call the core helper with a stride of 1 for contiguous output
                    CTX.dequantize_block_kernel(
                        current_q_ptr,
                        current_out_ptr,
                        CTX=CTX,
                        OUT_DTYPE=OUT_DTYPE,
                    )


class KernelDefinition(NamedTuple):
    qtype: GGMLQuantizationType
    block_size: int
    type_size: int
    kernel: KernelImpl
    autotuner_kernel: triton.runtime.Autotuner

    @classmethod
    def build(
        cls,
        qtype: GGMLQuantizationType,
        kernel_class: type[KernelImpl],
    ) -> "KernelDefinition":
        block_size, type_size = GGML_QUANT_SIZES[qtype]
        kernel_instance = kernel_class(
            block_size=tl.constexpr(block_size),
            type_size=tl.constexpr(type_size),
        )
        autotuner_kernel = kernel_instance.get_autotuner(
            configs=_AUTOTUNE_CONFIGS.get(
                qtype.name.lower(), _DEFAULT_AUTOTUNE_CONFIGS
            ),
            key=["n_total_blocks"],
        )
        return cls(
            qtype=qtype,
            block_size=block_size,
            type_size=type_size,
            kernel=kernel_instance,
            autotuner_kernel=autotuner_kernel,
        )

    def __call__(
        self,
        blocks: torch.Tensor,
        block_size: int,
        type_size: int,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        qtype, ggml_type_size = self.qtype, self.type_size
        if blocks.dtype != torch.uint8:
            if blocks.dtype == torch.int8:
                blocks = blocks.view(dtype=torch.uint8)
            else:
                raise ValueError(
                    f"GGUF Triton {qtype.name}: Blocks tensor dtype must be uint8 or int8 but got {blocks.dtype}"
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

        def grid(meta: dict[str, Any]) -> tuple[int]:
            return (triton.cdiv(n_total_blocks, meta["N_BLOCKS_PER_PROG"]),)

        self.autotuner_kernel[grid](
            blocks,
            out_tensor,
            n_total_blocks,
            CTX=self.kernel,
            OUT_DTYPE=triton_dtype,
        )

        return out_tensor


### K-quants


@dataclass
class KernelImpl_K_Quant(KernelImpl):
    k_scale_size: tl.constexpr = dcfield(
        default_factory=lambda: tl.constexpr(K_SCALE_SIZE)
    )


@dataclass
class KernelImpl_Q2_K(KernelImpl_K_Quant):
    @staticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr,
        out_tensor_ptr,
        CTX: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ) -> None:
        # Vector of offsets for a 16-element chunk
        offsets_16 = tl.arange(0, 16)

        # Data layout for Q2_K (TYPE_SIZE = 84 bytes)
        scales_ptr = block_start_ptr
        qs_ptr = block_start_ptr + 16
        d_ptr = block_start_ptr + 80
        dmin_ptr = block_start_ptr + 82

        # --- Load the super-scales 'd' and 'dmin' ---
        d = tl.load(d_ptr.to(tl.pointer_type(tl.float16))).to(OUT_DTYPE)
        dmin = tl.load(dmin_ptr.to(tl.pointer_type(tl.float16))).to(OUT_DTYPE)

        # --- Process block in 16 chunks of 16 values ---
        for chunk_idx in tl.static_range(16):
            # 1. Unpack the scales for this chunk.
            # Each of the 16 scale bytes corresponds to a 16-element chunk.
            # The low nibble scales 'd', the high nibble scales 'dmin'.
            scale_byte = tl.load(scales_ptr + chunk_idx)

            dl = d * (scale_byte & 0x0F).to(OUT_DTYPE)
            ml = dmin * (scale_byte >> 4).to(OUT_DTYPE)

            # --- Map the 16 output elements to their source data ---
            # This logic correctly models the Python reshape from a flat 256-element array.
            flat_indices = chunk_idx * 16 + offsets_16

            # 2. Unpack the 2-bit quantized values (qs).
            # The logical source array for qs is (2 segments * 4 shifts * 32 bytes).
            source_row = flat_indices // 32
            source_col = flat_indices % 32

            segment = source_row // 4
            shift_group = source_row % 4

            # Gather bytes from their calculated source pointers
            ptr = qs_ptr + segment * 32 + source_col
            byte = tl.load(ptr)

            # Apply the correct bit shift to extract the 2-bit value
            q_vec = (byte >> (shift_group * 2)) & 3

            # 3. Dequantize and store the 16 results.
            dequant_16 = dl * q_vec.to(OUT_DTYPE) - ml

            output_ptr = out_tensor_ptr + chunk_idx * 16
            tl.store(output_ptr + offsets_16, dequant_16)


@dataclass
class KernelImpl_Q3_K(KernelImpl_K_Quant):
    @staticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr,
        out_tensor_ptr,
        CTX: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ) -> None:
        # Vector of offsets for a 16-element chunk (one row of the output matrix)
        offsets_16 = tl.arange(0, 16)

        hmask_ptr = block_start_ptr
        qs_ptr = block_start_ptr + 32
        scales_ptr = block_start_ptr + 96
        d_ptr = block_start_ptr + 108

        # --- Load the super-scale 'd' ---
        d_super_scale = tl.load(d_ptr.to(tl.pointer_type(tl.float16))).to(OUT_DTYPE)

        # --- Process block in 16 chunks of 16 values ---
        for chunk_idx in tl.static_range(16):
            # 1. Unpack the 6-bit scale for this chunk.

            # Low 4 bits of the scale (lscale_nibble)
            lscale_byte_index = chunk_idx % 8
            lscale_shift = (chunk_idx // 8) * 4
            lscale_byte = tl.load(scales_ptr + lscale_byte_index)
            lscale_nibble = (lscale_byte >> lscale_shift) & 0x0F

            # High 2 bits of the scale (hscale_2bit)
            hscale_byte_index = chunk_idx % 4
            hscale_shift_index = chunk_idx // 4
            hscale_byte = tl.load(scales_ptr + 8 + hscale_byte_index)
            hscale_2bit = (hscale_byte >> (hscale_shift_index * 2)) & 0x03

            scale_6bit = lscale_nibble | (hscale_2bit << 4)
            final_scale = d_super_scale * (
                scale_6bit.to(tl.int8, bitcast=True) - 32
            ).to(OUT_DTYPE)

            # --- Map the 16 output elements to their source data ---
            flat_indices = chunk_idx * 16 + offsets_16

            # 2. Unpack ql (lower 2 bits).
            ql_source_row = flat_indices // 32
            ql_source_col = flat_indices % 32

            ql_segment = ql_source_row // 4
            ql_shift_group = ql_source_row % 4

            ql_ptr = qs_ptr + ql_segment * 32 + ql_source_col
            ql_byte = tl.load(ql_ptr)
            ql_vec = ((ql_byte >> (ql_shift_group * 2)) & 3).to(tl.int8, bitcast=True)

            # 3. Unpack qh (higher 1 bit, inverted).
            qh_source_row = flat_indices // 32
            qh_source_col = flat_indices % 32

            qh_ptr = hmask_ptr + qh_source_col
            qh_byte = tl.load(qh_ptr)
            qh_vec = (((qh_byte >> qh_source_row) & 1) ^ 1).to(tl.int8, bitcast=True)

            # 4. Combine to get the final 3-bit quantized value.
            q_vec = ql_vec - (qh_vec << 2)

            # 5. Dequantize and store the 16 results.
            dequant_16 = final_scale * q_vec.to(OUT_DTYPE)
            output_ptr = out_tensor_ptr + chunk_idx * 16 + offsets_16
            tl.store(output_ptr, dequant_16)


@dataclass
class KernelImpl_Q4_K(KernelImpl_K_Quant):
    # Helper function, shared by Q4_K and Q5_K.
    @staticmethod
    @triton.jit
    def get_scales_min(
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

    @staticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr,
        out_tensor_ptr,
        CTX: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ) -> None:
        offsets_32 = tl.arange(0, 32)
        offsets_scale = offsets_32 + 4 + CTX.k_scale_size

        d = tl.load(block_start_ptr.to(tl.pointer_type(tl.float16))).to(OUT_DTYPE)
        dmin = tl.load((block_start_ptr + 2).to(tl.pointer_type(tl.float16))).to(
            OUT_DTYPE
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
            sc_a, m_a = CTX.get_scales_min(k_idx, d_sc_word, m_word, m_sc_word)

            # --- Get scale B (for high nibbles) ---
            sc_b, m_b = CTX.get_scales_min(k_idx + 1, d_sc_word, m_word, m_sc_word)

            current_d_a = d * sc_a.to(OUT_DTYPE)
            current_dm_a = dmin * m_a.to(OUT_DTYPE)
            current_d_b = d * sc_b.to(OUT_DTYPE)
            current_dm_b = dmin * m_b.to(OUT_DTYPE)

            # Load 32 bytes of quantized data
            chunk_qs_ptr = qs_start_ptr + k_chunk * 32
            qs_bytes_chunk = tl.load(chunk_qs_ptr)

            qs_low = (qs_bytes_chunk & 0x0F).to(OUT_DTYPE)
            qs_high = (qs_bytes_chunk >> 4).to(OUT_DTYPE)

            dequant_low = current_d_a * qs_low - current_dm_a
            dequant_high = current_d_b * qs_high - current_dm_b

            # Store results contiguously
            output_chunk_ptr = out_tensor_ptr + k_chunk * 64 + offsets_32
            output_chunk_ptr.store(dequant_low)
            (output_chunk_ptr + 32).store(dequant_high)


@dataclass
class KernelImpl_Q5_K(KernelImpl_Q4_K):
    @staticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr,
        out_tensor_ptr,
        CTX: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ) -> None:
        offsets_32 = tl.arange(0, 32)
        offsets_scale = offsets_32 + 4 + CTX.k_scale_size

        # Pointers and initial loads
        d = tl.load(block_start_ptr.to(tl.pointer_type(tl.float16))).to(OUT_DTYPE)
        dmin = tl.load((block_start_ptr + 2).to(tl.pointer_type(tl.float16))).to(
            OUT_DTYPE
        )

        scales_ptr_u32 = (block_start_ptr + 4).to(tl.pointer_type(tl.uint32))
        d_sc_word = tl.load(scales_ptr_u32 + 0)
        m_word = tl.load(scales_ptr_u32 + 1)
        m_sc_word = tl.load(scales_ptr_u32 + 2)

        qh_start_ptr = block_start_ptr + offsets_scale
        qs_start_ptr = qh_start_ptr + CTX.block_size // 8

        qh_bytes_all = tl.load(qh_start_ptr)

        # Process in 8 chunks of 32 values
        for chunk_idx in tl.static_range(8):
            # # 1. Unpack scale and min for this chunk
            sc, m = CTX.get_scales_min(chunk_idx, d_sc_word, m_word, m_sc_word)

            final_d = d * sc.to(OUT_DTYPE)
            final_dm = dmin * m.to(OUT_DTYPE)

            # 2. Unpack ql (lower 4 bits) for this chunk
            qs_byte_offset = (chunk_idx // 2) * 32
            qs_bytes = tl.load(qs_start_ptr + qs_byte_offset)
            use_low_nibbles = chunk_idx % 2 == 0
            ql = tl.where(use_low_nibbles, qs_bytes & 0x0F, qs_bytes >> 4)

            # 3. Unpack qh (higher 1 bit) for this chunk
            qh = (qh_bytes_all >> chunk_idx) & 0x01

            # 4. Combine, dequantize, and store
            q = ql | (qh << 4)
            dequant_32 = final_d * q.to(OUT_DTYPE) - final_dm

            output_ptr = out_tensor_ptr + chunk_idx * 32 + offsets_32
            output_ptr.store(dequant_32)


@dataclass
class KernelImpl_Q6_K(KernelImpl_K_Quant):
    @staticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr,
        out_tensor_ptr,
        CTX: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ) -> None:
        offsets_32 = tl.arange(0, 32)
        mask_16 = offsets_32 < 16

        d_ptr = block_start_ptr + 208
        scales_ptr = block_start_ptr + 192
        d_super_scale = tl.load(d_ptr.to(tl.pointer_type(tl.float16))).to(OUT_DTYPE)

        # Process block in 8 chunks of 32 values
        for chunk_idx in tl.static_range(8):
            # 1. Calculate ql source data and unpack
            ql_byte_offset = (chunk_idx % 2) * 32 + (chunk_idx // 4) * 64
            ql_ptr = block_start_ptr + ql_byte_offset
            ql_32_bytes = tl.load(ql_ptr + offsets_32)

            use_low_nibbles = (chunk_idx // 2) % 2 == 0
            ql_vec_32 = ql_32_bytes & 0x0F if use_low_nibbles else ql_32_bytes >> 4
            ql_vec_32 = ql_vec_32.to(tl.int8, bitcast=True)

            # 2. Calculate qh source data and unpack
            qh_byte_offset = (chunk_idx // 4) * 32
            qh_ptr = block_start_ptr + 128 + qh_byte_offset

            bit_shift = (chunk_idx % 4) * 2
            qh_32_bytes = tl.load(qh_ptr + offsets_32)
            qh_vec_32 = (qh_32_bytes.to(tl.int8, bitcast=True) >> bit_shift) & 0x03

            # 3. Combine and dequantize
            q_vec_32 = ((ql_vec_32 | (qh_vec_32 << 4)) - 32).to(OUT_DTYPE)

            # 4. Load and apply correct scales
            scale_0_ptr = scales_ptr + chunk_idx * 2
            scales_0_1 = (
                tl.where(
                    mask_16,
                    tl.load(scale_0_ptr),
                    tl.load(scale_0_ptr + 1),
                )
                .to(tl.int8, bitcast=True)
                .to(OUT_DTYPE)
            )
            scales_32 = d_super_scale * scales_0_1
            dequant_32 = q_vec_32 * scales_32

            # 5. Store result
            output_ptr = out_tensor_ptr + chunk_idx * 32
            tl.store(output_ptr + offsets_32, dequant_32)


### Legacy quants


@dataclass
class KernelImpl_Legacy(KernelImpl):
    @staticmethod
    @triton.jit
    def store_output(out_tensor_ptr, dequant_low, dequant_high) -> None:
        offsets_16 = tl.arange(0, 16)

        out_ptrs_low = out_tensor_ptr + offsets_16
        out_ptrs_high = out_tensor_ptr + 16 + offsets_16

        # Store the 32 dequantized results.
        out_ptrs_low.store(dequant_low)
        out_ptrs_high.store(dequant_high)


@dataclass
class KernelImpl_Q4_0(KernelImpl_Legacy):
    @staticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr,
        out_tensor_ptr,
        CTX: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ) -> None:
        # Vector of offsets for the 16 bytes of quantized data
        offsets_16 = tl.arange(0, 16)

        # 1. Load the float16 scale 'd'. It's the first 2 bytes of the block.
        d = tl.load(block_start_ptr.to(tl.pointer_type(tl.float16))).to(OUT_DTYPE)

        # 2. Load the 16 bytes of quantized data ('qs').
        qs_ptr = block_start_ptr + 2
        qs_bytes_16 = tl.load(qs_ptr + offsets_16)

        # 3. Unpack the 16 bytes into 32 4-bit values (nibbles).
        # The low nibbles form the first 16 values of the block.
        qs_low = (qs_bytes_16 & 0x0F).to(tl.int8, bitcast=True)
        # The high nibbles form the second 16 values of the block.
        qs_high = (qs_bytes_16 >> 4).to(tl.int8, bitcast=True)

        # 4. Dequantize the values from unsigned 0-15 to signed -8 to 7.
        q_low = qs_low - 8
        q_high = qs_high - 8

        # 5. Apply the scale and store the 32 dequantized results.
        dequant_low = d * q_low.to(OUT_DTYPE)
        dequant_high = d * q_high.to(OUT_DTYPE)

        CTX.store_output(out_tensor_ptr, dequant_low, dequant_high)


@dataclass
class KernelImpl_Q4_1(KernelImpl_Legacy):
    @staticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr,
        out_tensor_ptr,
        CTX: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ) -> None:
        # Vector of offsets for the 16 bytes of quantized data
        offsets_16 = tl.arange(0, 16)

        # 1. Load scale 'd' (first 2 bytes) and min 'm' (next 2 bytes).
        d_ptr = block_start_ptr
        m_ptr = block_start_ptr + 2

        d = tl.load(d_ptr.to(tl.pointer_type(tl.float16))).to(OUT_DTYPE)
        m = tl.load(m_ptr.to(tl.pointer_type(tl.float16))).to(OUT_DTYPE)

        # 2. Load the 16 bytes of quantized data ('qs').
        qs_ptr = block_start_ptr + 4
        qs_bytes_16 = tl.load(qs_ptr + offsets_16)

        # 3. Unpack the 16 bytes into 32 4-bit values (0-15).
        qs_low = (qs_bytes_16 & 0x0F).to(OUT_DTYPE)
        qs_high = (qs_bytes_16 >> 4).to(OUT_DTYPE)

        # 4. Dequantize: (d * qs) + m
        dequant_low = d * qs_low + m
        dequant_high = d * qs_high + m

        CTX.store_output(out_tensor_ptr, dequant_low, dequant_high)


@dataclass
class KernelImpl_Q5_0(KernelImpl_Legacy):
    @staticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr,
        out_tensor_ptr,
        CTX: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ) -> None:
        offsets_16 = tl.arange(0, 16)
        offsets_4 = tl.arange(0, 4)

        d_ptr = block_start_ptr
        qh_ptr = block_start_ptr + 2
        qs_ptr = block_start_ptr + 6

        d = tl.load(d_ptr.to(tl.pointer_type(tl.float16))).to(OUT_DTYPE)
        qh_word = (tl.load(qh_ptr + offsets_4).to(tl.uint32) << (offsets_4 << 3)).sum()
        qs_bytes_16 = tl.load(qs_ptr + offsets_16)

        ql_low = qs_bytes_16 & 0x0F
        qh_low = (qh_word >> offsets_16) & 1
        q_low = (ql_low | (qh_low << 4)).to(tl.int8, bitcast=True) - 16
        dequant_low = d * q_low.to(OUT_DTYPE)  # Shape: [16]

        ql_high = qs_bytes_16 >> 4
        qh_high = (qh_word >> (offsets_16 + 16)) & 1
        q_high = (ql_high | (qh_high << 4)).to(tl.int8, bitcast=True) - 16
        dequant_high = d * q_high.to(OUT_DTYPE)  # Shape: [16]

        CTX.store_output(out_tensor_ptr, dequant_low, dequant_high)


@dataclass
class KernelImpl_Q5_1(KernelImpl_Legacy):
    @staticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr,
        out_tensor_ptr,
        CTX: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ) -> None:
        offsets_16 = tl.arange(0, 16)

        # Data layout: 2 bytes 'd', 2 bytes 'm', 4 bytes 'qh', 16 bytes 'qs'
        d_ptr = block_start_ptr
        m_ptr = block_start_ptr + 2
        qh_ptr = block_start_ptr + 4
        qs_ptr = block_start_ptr + 8

        # 1. Load the scales 'd', 'm' and the high-bit mask 'qh'.
        d = tl.load(d_ptr.to(tl.pointer_type(tl.float16))).to(OUT_DTYPE)
        m = tl.load(m_ptr.to(tl.pointer_type(tl.float16))).to(OUT_DTYPE)
        # This is a safe aligned load because TYPE_SIZE (24) and qh offset (4) are multiples of 4.
        qh_word = tl.load(qh_ptr.to(tl.pointer_type(tl.uint32)))

        # 2. Load the 16 bytes of low-bits 'qs'.
        qs_bytes_16 = tl.load(qs_ptr + offsets_16)

        # --- Process the first 16 values ---
        ql_low = qs_bytes_16 & 0x0F
        qh_low = (qh_word >> offsets_16) & 1
        q_low = (ql_low | (qh_low << 4)).to(OUT_DTYPE)
        dequant_low = d * q_low + m

        # --- Process the second 16 values ---
        ql_high = qs_bytes_16 >> 4
        qh_high = (qh_word >> (offsets_16 + 16)) & 1
        q_high = (ql_high | (qh_high << 4)).to(OUT_DTYPE)
        dequant_high = d * q_high + m

        CTX.store_output(out_tensor_ptr, dequant_low, dequant_high)


dequantize_functions: dict[GGMLQuantizationType, KernelDefinition] = {
    GQT.Q4_0: KernelDefinition.build(GQT.Q4_0, KernelImpl_Q4_0),
    GQT.Q4_1: KernelDefinition.build(GQT.Q4_1, KernelImpl_Q4_1),
    GQT.Q5_0: KernelDefinition.build(GQT.Q5_0, KernelImpl_Q5_0),
    GQT.Q5_1: KernelDefinition.build(GQT.Q5_1, KernelImpl_Q5_1),
    GQT.Q2_K: KernelDefinition.build(GQT.Q2_K, KernelImpl_Q2_K),
    GQT.Q3_K: KernelDefinition.build(GQT.Q3_K, KernelImpl_Q3_K),
    GQT.Q4_K: KernelDefinition.build(GQT.Q4_K, KernelImpl_Q4_K),
    GQT.Q5_K: KernelDefinition.build(GQT.Q5_K, KernelImpl_Q5_K),
    GQT.Q6_K: KernelDefinition.build(GQT.Q6_K, KernelImpl_Q6_K),
}

__all__ = ("dequantize_functions",)
