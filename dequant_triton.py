from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dcfield
from typing import Any, TypeVar

import torch
import triton
import triton.language as tl
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

C = TypeVar("C")


def passthroughdecorator(c: C) -> C:
    return c


nocompiledecorator = (
    getattr(getattr(torch, "compiler", None), "disable", None) or passthroughdecorator
)

TRITON_MAJOR, TRITON_MINOR = (
    int(part) for part in triton.__version__.split(".", 3)[:2]
)

# This static method stuff may not be necessary. Right now, Triton doesn't pass self
# in 3.3 or 3.4 whether or not the method is decorated with staticmethod. Just afraid of that
# changing and breaking stuff in future versions. Triton 3.4+ can deal with the staticmethod decorator.
if TRITON_MAJOR == 3 and TRITON_MINOR <= 3:
    maybestaticmethod = passthroughdecorator
elif TRITON_MAJOR == 3 and TRITON_MINOR >= 4:
    maybestaticmethod = staticmethod
elif TRITON_MAJOR < 3:
    raise RuntimeError(
        f"Triton major versions less than 3 not supported, you have {triton.__version__}"
    )
else:
    print(
        f"\n*** GGUF Triton: Your Triton version of {triton.__version__} has not been tested and may not work correctly."
    )
    maybestaticmethod = staticmethod


K_SCALE_SIZE = 12

TORCH_TO_TRITON_DTYPE_MAP: dict[torch.dtype, tl.dtype] = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

_DEFAULT_AUTOTUNE_CONFIGS: list[triton.Config] = [
    triton.Config({"N_BLOCKS_PER_PROG": 1}, num_warps=2),
    triton.Config({"N_BLOCKS_PER_PROG": 2}, num_warps=2),
    triton.Config({"N_BLOCKS_PER_PROG": 4}, num_warps=2),
    triton.Config({"N_BLOCKS_PER_PROG": 1}, num_warps=4),
    triton.Config({"N_BLOCKS_PER_PROG": 2}, num_warps=4),
    triton.Config({"N_BLOCKS_PER_PROG": 4}, num_warps=4),
    triton.Config({"N_BLOCKS_PER_PROG": 1}, num_warps=8),
    triton.Config({"N_BLOCKS_PER_PROG": 2}, num_warps=8),
    triton.Config({"N_BLOCKS_PER_PROG": 4}, num_warps=8),
]

_QUANT_AUTOTUNE_CONFIGS: dict[str, list[triton.Config]] = {}

_DEFAULT_VECTORIZED_AUTOTUNE_CONFIGS: list[triton.Config] = [
    triton.Config({"BLOCK_N": 16}, num_warps=2),
    triton.Config({"BLOCK_N": 32}, num_warps=2),
    triton.Config({"BLOCK_N": 64}, num_warps=4),
    triton.Config({"BLOCK_N": 128}, num_warps=4),
    triton.Config({"BLOCK_N": 256}, num_warps=8),
]

_VECTORIZED_QUANT_AUTOTUNE_CONFIGS: dict[str, list[triton.Config]] = {}


@dataclass(frozen=True)
class KernelImpl:
    type_size: tl.constexpr
    block_size: tl.constexpr
    # Vectorized kernel related parameters.
    vectorized: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(False))
    chunk_size: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(32))
    num_chunks: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(1))

    @property
    def have_vectorized(self) -> bool:
        return bool(getattr(self.vectorized, "value", self.vectorized))

    def get_autotuner(
        self,
        *,
        use_vectorized: bool = True,
        **kwargs: Any,
    ) -> triton.runtime.Autotuner:
        kernel_fn = (
            self.dequantize_kernel_vectorized
            if self.have_vectorized and use_vectorized
            else self.dequantize_kernel
        )
        return triton.autotune(**kwargs)(kernel_fn)

    @maybestaticmethod
    @triton.jit
    def dequantize_kernel(
        q_tensor_ptr,
        out_tensor_ptr,
        n_total_blocks,
        DTYPE: tl.constexpr,
        N_BLOCKS_PER_PROG: tl.constexpr,
        CTX: tl.constexpr,
    ) -> None:
        pid = tl.program_id(axis=0)
        start_block_idx = pid * N_BLOCKS_PER_PROG
        n_blocks = n_total_blocks - start_block_idx

        if n_blocks > 0:
            for i in tl.static_range(N_BLOCKS_PER_PROG):
                if i < n_blocks:
                    block_offset = start_block_idx + i
                    quantized_block_ptr = (
                        q_tensor_ptr + block_offset * CTX.value.type_size
                    )
                    output_ptr = out_tensor_ptr + block_offset * CTX.value.block_size

                    CTX.value.dequantize_block_kernel(
                        quantized_block_ptr,
                        output_ptr,
                        CTX=tl.constexpr(CTX),
                        DTYPE=DTYPE,
                    )

    @maybestaticmethod
    @triton.jit
    def dequantize_kernel_vectorized(
        q_tensor_ptr,
        out_tensor_ptr,
        n_total_blocks,
        DTYPE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CTX: tl.constexpr,
    ) -> None:
        pid = tl.program_id(axis=0)
        block_indices = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_1d = block_indices < n_total_blocks

        out_base_ptrs = out_tensor_ptr + block_indices * CTX.value.block_size
        offsets_chunk = tl.arange(0, CTX.value.chunk_size)
        base_ptrs = q_tensor_ptr + block_indices * CTX.value.type_size

        for chunk_idx in tl.static_range(CTX.value.num_chunks):
            dequant = CTX.value.dequantize_chunk_to_registers(
                base_ptrs, mask_1d, chunk_idx, CTX, DTYPE
            )
            # Retain original pointer layouts for the one-shot kernel
            out_chunk_ptrs = (
                out_base_ptrs[:, None]
                + chunk_idx * CTX.value.chunk_size
                + offsets_chunk[None, :]
            )
            tl.store(out_chunk_ptrs, dequant, mask=mask_1d[:, None])


class KernelDefinition:
    qtype: GGMLQuantizationType
    block_size: int
    type_size: int
    kernel: KernelImpl
    autotuner_kernel: triton.runtime.Autotuner
    use_vectorized: bool

    def __init__(
        self,
        qtype: GGMLQuantizationType,
        *,
        kernel_class: type[KernelImpl] | None = None,
        kernel_instance: KernelImpl | None = None,
        use_vectorized: bool = True,
        **_kwargs: Any,
    ):
        block_size, type_size = GGML_QUANT_SIZES[qtype]
        if kernel_instance is None:
            if kernel_class is None:
                raise ValueError(
                    "At least one of kernel_class or kernel_instance must be set",
                )
            kernel_instance = self.kernel = kernel_class(
                block_size=tl.constexpr(block_size),
                type_size=tl.constexpr(type_size),
            )
        elif kernel_class is not None:
            raise ValueError(
                "Only one of kernel_class or kernel_instance may be set",
            )
        self.use_vectorized = use_vectorized and self.have_vectorized
        if self.use_vectorized:
            default_configs = _DEFAULT_VECTORIZED_AUTOTUNE_CONFIGS
            quant_configs = _VECTORIZED_QUANT_AUTOTUNE_CONFIGS
        else:
            default_configs = _DEFAULT_AUTOTUNE_CONFIGS
            quant_configs = _QUANT_AUTOTUNE_CONFIGS
        autotuner_kernel = kernel_instance.get_autotuner(
            use_vectorized=self.use_vectorized,
            configs=quant_configs.get(qtype.name.lower(), default_configs),
            key=["n_total_blocks"],
        )
        self.qtype = qtype
        self.block_size = block_size
        self.type_size = type_size
        self.autotuner_kernel = autotuner_kernel
        # print(
        #     f"DEFINED({qtype}): use vectorized={self.use_vectorized}, have vectorized={self.have_vectorized}, kernel={self.kernel}",
        # )

    @property
    def have_vectorized(self) -> bool:
        return self.kernel.have_vectorized

    @nocompiledecorator
    def __call__(
        self,
        blocks: torch.Tensor,
        block_size: int = -1,
        type_size: int = -1,
        dtype: torch.dtype | None = None,
        *,
        _math_dtype: tl.dtype | None = tl.float32,
        _use_vectorized: bool = True,
        out_tensor: torch.Tensor | None = None,
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
        if _math_dtype is not None:
            triton_dtype = _math_dtype
        elif (triton_dtype := TORCH_TO_TRITON_DTYPE_MAP.get(dtype)) is None:
            raise TypeError(
                f"GGUF Triton {qtype.name}: Unsupported output dtype {dtype}"
            )

        if out_tensor is None:
            out_tensor = torch.empty(
                n_total_blocks * self.block_size, dtype=dtype, device=blocks.device
            )

        if self.use_vectorized:

            def grid(meta: dict[str, Any]) -> tuple[int]:
                return (triton.cdiv(n_total_blocks, meta["BLOCK_N"]),)
        else:

            def grid(meta: dict[str, Any]) -> tuple[int]:
                return (triton.cdiv(n_total_blocks, meta["N_BLOCKS_PER_PROG"]),)

        self.autotuner_kernel[grid](
            blocks,
            out_tensor,
            n_total_blocks,
            CTX=self.kernel,
            DTYPE=triton_dtype,
        )

        return out_tensor


### K-quants


@dataclass(frozen=True)
class KernelImpl_K_Quant(KernelImpl):
    k_scale_size: tl.constexpr = dcfield(
        default_factory=lambda: tl.constexpr(K_SCALE_SIZE)
    )


@dataclass(frozen=True)
class KernelImpl_Q2_K(KernelImpl_K_Quant):
    vectorized: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(True))
    chunk_size: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(32))
    num_chunks: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(8))

    @maybestaticmethod
    @triton.jit
    def dequantize_chunk_to_registers(
        base_ptrs, mask_1d, chunk_idx, CTX: tl.constexpr, DTYPE: tl.constexpr
    ):
        mask_2d = mask_1d[:, None]
        offsets_32 = tl.arange(0, 32)

        # 1. Load super-scales
        d_ptrs = (base_ptrs + 80).to(tl.pointer_type(tl.float16))
        dmin_ptrs = (base_ptrs + 82).to(tl.pointer_type(tl.float16))

        d = tl.load(d_ptrs, mask=mask_1d, other=0.0).to(DTYPE)
        dmin = tl.load(dmin_ptrs, mask=mask_1d, other=0.0).to(DTYPE)

        # 2. Block scales (1 scale byte handles 16 elements)
        scale_idx = chunk_idx * 2 + (offsets_32 // 16)
        scale_ptrs = base_ptrs[:, None] + scale_idx[None, :]
        scale_bytes = tl.load(scale_ptrs, mask=mask_2d, other=0)

        dl_scale = (scale_bytes & 0x0F).to(DTYPE)
        ml_scale = (scale_bytes >> 4).to(DTYPE)

        dl = d[:, None] * dl_scale
        ml = dmin[:, None] * ml_scale

        # 3. Quantized values (qs)
        qs_offset = 16 + (chunk_idx // 4) * 32
        qs_ptrs = base_ptrs[:, None] + qs_offset + offsets_32[None, :]
        qs_bytes = tl.load(qs_ptrs, mask=mask_2d, other=0)

        shift = (chunk_idx % 4) * 2
        q_vec = ((qs_bytes >> shift) & 3).to(DTYPE)

        # 4. Dequantize
        return dl * q_vec - ml

    @maybestaticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr, out_tensor_ptr, CTX: tl.constexpr, DTYPE: tl.constexpr
    ) -> None:
        # Vector of offsets for a 16-element chunk
        offsets_16 = tl.arange(0, 16)

        # Data layout for Q2_K (TYPE_SIZE = 84 bytes)
        scales_ptr = block_start_ptr
        qs_ptr = block_start_ptr + 16
        d_ptr = block_start_ptr + 80
        dmin_ptr = block_start_ptr + 82

        # --- Load the super-scales 'd' and 'dmin' ---
        d = tl.load(d_ptr.to(tl.pointer_type(tl.float16))).to(DTYPE)
        dmin = tl.load(dmin_ptr.to(tl.pointer_type(tl.float16))).to(DTYPE)

        # --- Process block in 16 chunks of 16 values ---
        for chunk_idx in tl.static_range(16):
            # 1. Unpack the scales for this chunk.
            # Each of the 16 scale bytes corresponds to a 16-element chunk.
            # The low nibble scales 'd', the high nibble scales 'dmin'.
            scale_byte = tl.load(scales_ptr + chunk_idx)

            dl = d * (scale_byte & 0x0F).to(DTYPE)
            ml = dmin * (scale_byte >> 4).to(DTYPE)

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
            dequant_16 = dl * q_vec.to(DTYPE) - ml

            output_ptr = out_tensor_ptr + chunk_idx * 16
            tl.store(output_ptr + offsets_16, dequant_16)


@dataclass(frozen=True)
class KernelImpl_Q3_K(KernelImpl_K_Quant):
    vectorized: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(True))
    chunk_size: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(32))
    num_chunks: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(8))

    @maybestaticmethod
    @triton.jit
    def dequantize_chunk_to_registers(
        base_ptrs, mask_1d, chunk_idx, CTX: tl.constexpr, DTYPE: tl.constexpr
    ):
        mask_2d = mask_1d[:, None]
        offsets_32 = tl.arange(0, 32)

        # 1. Super-scale
        d_ptrs = (base_ptrs + 108).to(tl.pointer_type(tl.float16))
        d_super_scale = tl.load(d_ptrs, mask=mask_1d, other=0.0).to(DTYPE)

        # 2. Block scales
        chunk_16 = chunk_idx * 2 + (offsets_32 // 16)

        lscale_idx = chunk_16 % 8
        lscale_shift = (chunk_16 // 8) * 4
        lscale_ptrs = base_ptrs[:, None] + 96 + lscale_idx[None, :]
        lscale_bytes = tl.load(lscale_ptrs, mask=mask_2d, other=0)
        lscale_nibble = (lscale_bytes >> lscale_shift) & 0x0F

        hscale_idx = chunk_16 % 4
        hscale_shift = (chunk_16 // 4) * 2
        hscale_ptrs = base_ptrs[:, None] + 104 + hscale_idx[None, :]
        hscale_bytes = tl.load(hscale_ptrs, mask=mask_2d, other=0)
        hscale_2bit = (hscale_bytes >> hscale_shift) & 0x03

        scale_6bit = lscale_nibble | (hscale_2bit << 4)
        final_scale = d_super_scale[:, None] * (scale_6bit.to(tl.int8) - 32).to(DTYPE)

        # 3. ql (lower 2 bits)
        ql_offset = 32 + (chunk_idx // 4) * 32
        ql_ptrs = base_ptrs[:, None] + ql_offset + offsets_32[None, :]
        ql_bytes = tl.load(ql_ptrs, mask=mask_2d, other=0)
        ql_shift = (chunk_idx % 4) * 2
        ql_vec = (ql_bytes >> ql_shift) & 3

        # 4. qh (higher 1 bit, inverted)
        qh_ptrs = base_ptrs[:, None] + offsets_32[None, :]
        qh_bytes = tl.load(qh_ptrs, mask=mask_2d, other=0)
        # Using chunk_idx as the bit-shift natively replaces the old `qh_source_row` logic!
        qh_vec = ((qh_bytes >> chunk_idx) & 1) ^ 1

        # 5. Combine and dequantize
        q_vec = ql_vec.to(tl.int8) - (qh_vec.to(tl.int8) << 2)

        return final_scale * q_vec.to(DTYPE)

    @maybestaticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr, out_tensor_ptr, CTX: tl.constexpr, DTYPE: tl.constexpr
    ) -> None:
        # Vector of offsets for a 16-element chunk (one row of the output matrix)
        offsets_16 = tl.arange(0, 16)

        hmask_ptr = block_start_ptr
        qs_ptr = block_start_ptr + 32
        scales_ptr = block_start_ptr + 96
        d_ptr = block_start_ptr + 108

        # --- Load the super-scale 'd' ---
        d_super_scale = tl.load(d_ptr.to(tl.pointer_type(tl.float16))).to(DTYPE)

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
            final_scale = d_super_scale * (scale_6bit.to(tl.int8) - 32).to(DTYPE)

            # --- Map the 16 output elements to their source data ---
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
            dequant_16 = final_scale * q_vec.to(DTYPE)
            output_ptr = out_tensor_ptr + chunk_idx * 16 + offsets_16
            tl.store(output_ptr, dequant_16)


@dataclass(frozen=True)
class KernelImpl_Q4_K(KernelImpl_K_Quant):
    vectorized: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(True))
    chunk_size: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(64))
    num_chunks: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(4))

    # Helper function, shared by Q4_K and Q5_K.
    @maybestaticmethod
    @triton.jit
    def get_scales_min(
        k_idx: int, d_sc_word: tl.tensor, m_word: tl.tensor, m_sc_word: tl.tensor
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

    @maybestaticmethod
    @triton.jit
    def dequantize_chunk_to_registers(
        base_ptrs,
        mask_1d,
        chunk_idx,
        CTX: tl.constexpr,
        DTYPE: tl.constexpr,
    ):
        d_ptrs = base_ptrs.to(tl.pointer_type(tl.float16))
        d = tl.load(d_ptrs, mask=mask_1d, other=0.0).to(DTYPE)
        dmin = tl.load(d_ptrs + 1, mask=mask_1d, other=0.0).to(DTYPE)

        scales_ptrs_u32 = (base_ptrs + 4).to(tl.pointer_type(tl.uint32))
        d_sc_word = tl.load(scales_ptrs_u32, mask=mask_1d, other=0)
        m_word = tl.load(scales_ptrs_u32 + 1, mask=mask_1d, other=0)
        m_sc_word = tl.load(scales_ptrs_u32 + 2, mask=mask_1d, other=0)

        k_idx = 2 * chunk_idx
        sc_a, m_a = CTX.value.get_scales_min(k_idx, d_sc_word, m_word, m_sc_word)
        sc_b, m_b = CTX.value.get_scales_min(k_idx + 1, d_sc_word, m_word, m_sc_word)

        # Modulo trick for 64 elements (wrapping the 32 bytes)
        offsets_64 = tl.arange(0, 64)
        qs_byte_idx = offsets_64 % 32
        qs_ptrs = base_ptrs[:, None] + 16 + chunk_idx * 32 + qs_byte_idx[None, :]
        qs_bytes = tl.load(qs_ptrs, mask=mask_1d[:, None], other=0)

        is_high = offsets_64[None, :] >= 32
        qs_nibbles = tl.where(is_high, qs_bytes >> 4, qs_bytes) & 0x0F
        qs_vals = qs_nibbles.to(DTYPE)

        # Select the correct scales for the high vs low sides
        current_d = tl.where(
            is_high, (d * sc_b.to(DTYPE))[:, None], (d * sc_a.to(DTYPE))[:, None]
        )
        current_dm = tl.where(
            is_high, (dmin * m_b.to(DTYPE))[:, None], (dmin * m_a.to(DTYPE))[:, None]
        )

        return current_d * qs_vals - current_dm

    @maybestaticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr, out_tensor_ptr, CTX: tl.constexpr, DTYPE: tl.constexpr
    ) -> None:
        offsets_32 = tl.arange(0, 32)
        offsets_scale = offsets_32 + 4 + CTX.value.k_scale_size

        d = tl.load(block_start_ptr.to(tl.pointer_type(tl.float16))).to(DTYPE)
        dmin = tl.load((block_start_ptr + 2).to(tl.pointer_type(tl.float16))).to(DTYPE)

        scales_ptr_u32 = (block_start_ptr + 4).to(tl.pointer_type(tl.uint32))
        d_sc_word = tl.load(scales_ptr_u32 + 0)
        m_word = tl.load(scales_ptr_u32 + 1)
        m_sc_word = tl.load(scales_ptr_u32 + 2)

        qs_start_ptr = block_start_ptr + offsets_scale

        # Process in 4 chunks of 64 values
        for k_chunk in tl.static_range(4):
            k_idx = 2 * k_chunk

            # --- Get scale A (for low nibbles) ---
            sc_a, m_a = CTX.value.get_scales_min(k_idx, d_sc_word, m_word, m_sc_word)

            # --- Get scale B (for high nibbles) ---
            sc_b, m_b = CTX.value.get_scales_min(
                k_idx + 1, d_sc_word, m_word, m_sc_word
            )

            current_d_a = d * sc_a.to(DTYPE)
            current_dm_a = dmin * m_a.to(DTYPE)
            current_d_b = d * sc_b.to(DTYPE)
            current_dm_b = dmin * m_b.to(DTYPE)

            # Load 32 bytes of quantized data
            chunk_qs_ptr = qs_start_ptr + k_chunk * 32
            qs_bytes_chunk = tl.load(chunk_qs_ptr)

            qs_low = (qs_bytes_chunk & 0x0F).to(DTYPE)
            # qs_high = (qs_bytes_chunk >> 4).to(DTYPE)
            qs_high = ((qs_bytes_chunk >> 4) & 0x0F).to(DTYPE)

            dequant_low = current_d_a * qs_low - current_dm_a
            dequant_high = current_d_b * qs_high - current_dm_b

            # Store results contiguously
            output_chunk_ptr = out_tensor_ptr + k_chunk * 64 + offsets_32
            output_chunk_ptr.store(dequant_low)
            (output_chunk_ptr + 32).store(dequant_high)


@dataclass(frozen=True)
class KernelImpl_Q5_K(KernelImpl_Q4_K):
    vectorized: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(True))
    chunk_size: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(32))
    num_chunks: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(8))

    @maybestaticmethod
    @triton.jit
    def dequantize_chunk_to_registers(
        base_ptrs,
        mask_1d,
        chunk_idx,
        CTX: tl.constexpr,
        DTYPE: tl.constexpr,
    ):
        mask_2d = mask_1d[:, None]
        offsets_32 = tl.arange(0, 32)

        # 1. Super-scales
        d_ptrs = base_ptrs.to(tl.pointer_type(tl.float16))
        d = tl.load(d_ptrs, mask=mask_1d, other=0.0).to(DTYPE)
        dmin = tl.load(d_ptrs + 1, mask=mask_1d, other=0.0).to(DTYPE)

        # 2. Block scales
        scales_ptrs_u32 = (base_ptrs + 4).to(tl.pointer_type(tl.uint32))
        d_sc_word = tl.load(scales_ptrs_u32, mask=mask_1d, other=0)
        m_word = tl.load(scales_ptrs_u32 + 1, mask=mask_1d, other=0)
        m_sc_word = tl.load(scales_ptrs_u32 + 2, mask=mask_1d, other=0)

        sc, m = CTX.value.get_scales_min(chunk_idx, d_sc_word, m_word, m_sc_word)
        final_d = d * sc.to(DTYPE)
        final_dm = dmin * m.to(DTYPE)

        # 3. QL (Lower 4 bits)
        # qs offsets start at 48 (16 byte header + 32 byte qh)
        qs_byte_offset = (chunk_idx // 2) * 32
        qs_ptrs = base_ptrs[:, None] + 48 + qs_byte_offset + offsets_32[None, :]
        qs_bytes = tl.load(qs_ptrs, mask=mask_2d, other=0)

        # Use ALU Mux to avoid branching on the chunk index
        is_even = (chunk_idx % 2) == 0
        ql = tl.where(is_even, qs_bytes & 0x0F, (qs_bytes >> 4) & 0x0F)

        # 4. QH (High 1 bit)
        # qh offsets start at 16
        qh_ptrs = base_ptrs[:, None] + 16 + offsets_32[None, :]
        qh_bytes = tl.load(qh_ptrs, mask=mask_2d, other=0)
        qh_bit = (qh_bytes >> chunk_idx) & 0x01

        # Combine and dequantize
        q = ql | (qh_bit << 4)
        return final_d[:, None] * q.to(DTYPE) - final_dm[:, None]

    @maybestaticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr, out_tensor_ptr, CTX: tl.constexpr, DTYPE: tl.constexpr
    ) -> None:
        offsets_32 = tl.arange(0, 32)
        offsets_scale = offsets_32 + 4 + CTX.value.k_scale_size

        # Pointers and initial loads
        d = tl.load(block_start_ptr.to(tl.pointer_type(tl.float16))).to(DTYPE)
        dmin = tl.load((block_start_ptr + 2).to(tl.pointer_type(tl.float16))).to(DTYPE)

        scales_ptr_u32 = (block_start_ptr + 4).to(tl.pointer_type(tl.uint32))
        d_sc_word = tl.load(scales_ptr_u32 + 0)
        m_word = tl.load(scales_ptr_u32 + 1)
        m_sc_word = tl.load(scales_ptr_u32 + 2)

        qh_start_ptr = block_start_ptr + offsets_scale
        qs_start_ptr = qh_start_ptr + CTX.value.block_size // 8

        qh_bytes_all = tl.load(qh_start_ptr)

        # Process in 8 chunks of 32 values
        for chunk_idx in tl.static_range(8):
            # # 1. Unpack scale and min for this chunk
            sc, m = CTX.value.get_scales_min(chunk_idx, d_sc_word, m_word, m_sc_word)

            final_d = d * sc.to(DTYPE)
            final_dm = dmin * m.to(DTYPE)

            # 2. Unpack ql (lower 4 bits) for this chunk
            qs_byte_offset = (chunk_idx // 2) * 32
            qs_bytes = tl.load(qs_start_ptr + qs_byte_offset)
            use_low_nibbles = chunk_idx % 2 == 0
            ql = tl.where(use_low_nibbles, qs_bytes & 0x0F, qs_bytes >> 4)

            # 3. Unpack qh (higher 1 bit) for this chunk
            qh = (qh_bytes_all >> chunk_idx) & 0x01

            # 4. Combine, dequantize, and store
            q = ql | (qh << 4)
            dequant_32 = final_d * q.to(DTYPE) - final_dm

            output_ptr = out_tensor_ptr + chunk_idx * 32 + offsets_32
            output_ptr.store(dequant_32)


@dataclass(frozen=True)
class KernelImpl_Q6_K(KernelImpl_K_Quant):
    vectorized: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(True))
    chunk_size: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(32))
    num_chunks: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(8))

    @maybestaticmethod
    @triton.jit
    def dequantize_chunk_to_registers(
        base_ptrs,
        mask_1d,
        chunk_idx,
        CTX: tl.constexpr,
        DTYPE: tl.constexpr,
    ):
        mask_2d = mask_1d[:, None]
        offsets_32 = tl.arange(0, 32)

        # 1. Super-scale
        d_ptrs = (base_ptrs + 208).to(tl.pointer_type(tl.float16))
        d_super_scale = tl.load(d_ptrs, mask=mask_1d, other=0.0).to(DTYPE)

        # 2. QL (Lower 4 bits)
        ql_offset = (chunk_idx % 2) * 32 + (chunk_idx // 4) * 64
        ql_ptrs = base_ptrs[:, None] + ql_offset + offsets_32[None, :]
        ql_bytes = tl.load(ql_ptrs, mask=mask_2d, other=0)

        is_even = ((chunk_idx // 2) % 2) == 0
        ql_vec = tl.where(is_even, ql_bytes & 0x0F, (ql_bytes >> 4) & 0x0F)

        # 3. QH (High 2 bits)
        qh_offset = 128 + (chunk_idx // 4) * 32
        qh_ptrs = base_ptrs[:, None] + qh_offset + offsets_32[None, :]
        qh_bytes = tl.load(qh_ptrs, mask=mask_2d, other=0)

        bit_shift = (chunk_idx % 4) * 2
        qh_vec = (qh_bytes >> bit_shift) & 0x03

        # Combine
        q_vec = (ql_vec | (qh_vec << 4)).to(tl.int8) - 32

        # 4. Scales (int8)
        scale_idx = chunk_idx * 2 + (offsets_32 // 16)
        scale_ptrs = base_ptrs[:, None] + 192 + scale_idx[None, :]
        scale_ptrs_i8 = scale_ptrs.to(tl.pointer_type(tl.int8))
        scales = tl.load(scale_ptrs_i8, mask=mask_2d, other=0).to(DTYPE)

        # Compute and return!
        return q_vec.to(DTYPE) * (d_super_scale[:, None] * scales)

    @maybestaticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr, out_tensor_ptr, CTX: tl.constexpr, DTYPE: tl.constexpr
    ) -> None:
        offsets_32 = tl.arange(0, 32)
        mask_16 = offsets_32 < 16

        d_ptr = block_start_ptr + 208
        scales_ptr = block_start_ptr + 192
        d_super_scale = tl.load(d_ptr.to(tl.pointer_type(tl.float16))).to(DTYPE)

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
            q_vec_32 = ((ql_vec_32 | (qh_vec_32 << 4)) - 32).to(DTYPE)

            # 4. Load and apply correct scales
            scale_0_ptr = scales_ptr + chunk_idx * 2
            scales_0_1 = (
                tl.where(
                    mask_16,
                    tl.load(scale_0_ptr),
                    tl.load(scale_0_ptr + 1),
                )
                .to(tl.int8, bitcast=True)
                .to(DTYPE)
            )
            scales_32 = d_super_scale * scales_0_1
            dequant_32 = q_vec_32 * scales_32

            # 5. Store result
            output_ptr = out_tensor_ptr + chunk_idx * 32
            tl.store(output_ptr + offsets_32, dequant_32)


### Legacy quants


@dataclass(frozen=True)
class KernelImpl_Legacy(KernelImpl):
    @maybestaticmethod
    @triton.jit
    def store_output(out_tensor_ptr, dequant_low, dequant_high) -> None:
        offsets_16 = tl.arange(0, 16)

        out_ptrs_low = out_tensor_ptr + offsets_16
        out_ptrs_high = out_tensor_ptr + 16 + offsets_16

        # Store the 32 dequantized results.
        out_ptrs_low.store(dequant_low)
        out_ptrs_high.store(dequant_high)


@dataclass(frozen=True)
class KernelImpl_Q4_0(KernelImpl_Legacy):
    vectorized: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(True))
    chunk_size: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(32))
    num_chunks: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(1))

    @maybestaticmethod
    @triton.jit
    def dequantize_chunk_to_registers(
        base_ptrs,
        mask_1d,
        chunk_idx,
        CTX: tl.constexpr,
        DTYPE: tl.constexpr,
    ):
        d_ptrs = base_ptrs.to(tl.pointer_type(tl.float16))
        d = tl.load(d_ptrs, mask=mask_1d, other=0.0).to(DTYPE)

        # Modulo trick: loads [BLOCK_N, 32] layout natively by wrapping the 16 bytes!
        offsets_32 = tl.arange(0, 32)
        qs_byte_idx = offsets_32 % 16
        qs_ptrs = base_ptrs[:, None] + 2 + qs_byte_idx[None, :]
        qs_bytes = tl.load(qs_ptrs, mask=mask_1d[:, None], other=0)

        # Use an ALU mux to split high/low nibbles in the 32-element register
        is_high = offsets_32[None, :] >= 16
        qs_nibbles = tl.where(is_high, qs_bytes >> 4, qs_bytes)
        q_vals = (qs_nibbles & 0x0F).to(tl.int8, bitcast=True) - 8

        return q_vals.to(DTYPE) * d[:, None]

    @maybestaticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr, out_tensor_ptr, CTX: tl.constexpr, DTYPE: tl.constexpr
    ) -> None:
        # Vector of offsets for the 16 bytes of quantized data
        offsets_16 = tl.arange(0, 16)

        # 1. Load the float16 scale 'd'. It's the first 2 bytes of the block.
        d = tl.load(block_start_ptr.to(tl.pointer_type(tl.float16))).to(DTYPE)

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
        dequant_low = d * q_low.to(DTYPE)
        dequant_high = d * q_high.to(DTYPE)

        CTX.value.store_output(out_tensor_ptr, dequant_low, dequant_high)


@dataclass(frozen=True)
class KernelImpl_Q4_1(KernelImpl_Legacy):
    vectorized: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(True))
    chunk_size: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(32))
    num_chunks: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(1))

    @maybestaticmethod
    @triton.jit
    def dequantize_chunk_to_registers(
        base_ptrs, mask_1d, chunk_idx, CTX: tl.constexpr, DTYPE: tl.constexpr
    ):
        # 1. Load scale 'd' and min 'm'
        d_ptrs = base_ptrs.to(tl.pointer_type(tl.float16))
        d = tl.load(d_ptrs, mask=mask_1d, other=0.0).to(DTYPE)

        m_ptrs = (base_ptrs + 2).to(tl.pointer_type(tl.float16))
        m = tl.load(m_ptrs, mask=mask_1d, other=0.0).to(DTYPE)

        # 2. Modulo wrapping for 16-byte memory layout
        offsets_32 = tl.arange(0, 32)
        qs_byte_idx = offsets_32 % 16
        qs_ptrs = base_ptrs[:, None] + 4 + qs_byte_idx[None, :]
        qs_bytes = tl.load(qs_ptrs, mask=mask_1d[:, None], other=0)

        # 3. Unpack into 32 values
        is_high = offsets_32[None, :] >= 16
        qs_nibbles = tl.where(is_high, qs_bytes >> 4, qs_bytes)
        q_vals = (qs_nibbles & 0x0F).to(DTYPE)

        # 4. Dequantize
        return d[:, None] * q_vals + m[:, None]

    @maybestaticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr, out_tensor_ptr, CTX: tl.constexpr, DTYPE: tl.constexpr
    ) -> None:
        # Vector of offsets for the 16 bytes of quantized data
        offsets_16 = tl.arange(0, 16)

        # 1. Load scale 'd' (first 2 bytes) and min 'm' (next 2 bytes).
        d_ptr = block_start_ptr
        m_ptr = block_start_ptr + 2

        d = tl.load(d_ptr.to(tl.pointer_type(tl.float16))).to(DTYPE)
        m = tl.load(m_ptr.to(tl.pointer_type(tl.float16))).to(DTYPE)

        # 2. Load the 16 bytes of quantized data ('qs').
        qs_ptr = block_start_ptr + 4
        qs_bytes_16 = tl.load(qs_ptr + offsets_16)

        # 3. Unpack the 16 bytes into 32 4-bit values (0-15).
        qs_low = (qs_bytes_16 & 0x0F).to(DTYPE)
        qs_high = (qs_bytes_16 >> 4).to(DTYPE)

        # 4. Dequantize: (d * qs) + m
        dequant_low = d * qs_low + m
        dequant_high = d * qs_high + m

        CTX.value.store_output(out_tensor_ptr, dequant_low, dequant_high)


@dataclass(frozen=True)
class KernelImpl_Q5_0(KernelImpl_Legacy):
    vectorized: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(True))
    chunk_size: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(32))
    num_chunks: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(1))

    @maybestaticmethod
    @triton.jit
    def dequantize_chunk_to_registers(
        base_ptrs, mask_1d, chunk_idx, CTX: tl.constexpr, DTYPE: tl.constexpr
    ):
        # 1. Load scale 'd'
        d_ptrs = base_ptrs.to(tl.pointer_type(tl.float16))
        d = tl.load(d_ptrs, mask=mask_1d, other=0.0).to(DTYPE)

        offsets_32 = tl.arange(0, 32)

        # We map each of the 32 elements to the exact 1-byte memory address that holds its high bit.
        qh_byte_idx = offsets_32 // 8
        qh_bit_idx = offsets_32 % 8

        qh_ptrs = base_ptrs[:, None] + 2 + qh_byte_idx[None, :]
        qh_bytes = tl.load(qh_ptrs, mask=mask_1d[:, None], other=0)

        # Extract the specific bit for this element
        qh_bit = (qh_bytes >> qh_bit_idx[None, :]) & 1

        # 2. Load 16 bytes of low-bits (qs)
        qs_byte_idx = offsets_32 % 16
        qs_ptrs = base_ptrs[:, None] + 6 + qs_byte_idx[None, :]
        qs_bytes = tl.load(qs_ptrs, mask=mask_1d[:, None], other=0)

        # 3. Extract Low bits
        is_high = offsets_32[None, :] >= 16
        ql = tl.where(is_high, qs_bytes >> 4, qs_bytes) & 0x0F

        # 4. Combine and apply scale
        q_vals = (ql | (qh_bit << 4)).to(tl.int8) - 16

        return d[:, None] * q_vals.to(DTYPE)

    @maybestaticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr, out_tensor_ptr, CTX: tl.constexpr, DTYPE: tl.constexpr
    ) -> None:
        offsets_16 = tl.arange(0, 16)

        d_ptr = block_start_ptr
        qh_ptr = block_start_ptr + 2
        qs_ptr = block_start_ptr + 6

        d = tl.load(d_ptr.to(tl.pointer_type(tl.float16))).to(DTYPE)
        qs_bytes_16 = tl.load(qs_ptr + offsets_16)

        # --- Low 16 elements (indices 0 to 15) ---
        qh_byte_low = offsets_16 // 8
        qh_bit_low = offsets_16 % 8
        qh_bytes_loaded_low = tl.load(qh_ptr + qh_byte_low)

        ql_low = qs_bytes_16 & 0x0F
        qh_low = (qh_bytes_loaded_low >> qh_bit_low) & 1
        q_low = (ql_low | (qh_low << 4)).to(tl.int8) - 16
        dequant_low = d * q_low.to(DTYPE)

        # --- High 16 elements (indices 16 to 31) ---
        offsets_16_high = offsets_16 + 16
        qh_byte_high = offsets_16_high // 8
        qh_bit_high = offsets_16_high % 8
        qh_bytes_loaded_high = tl.load(qh_ptr + qh_byte_high)

        ql_high = qs_bytes_16 >> 4
        qh_high = (qh_bytes_loaded_high >> qh_bit_high) & 1
        q_high = (ql_high | (qh_high << 4)).to(tl.int8) - 16
        dequant_high = d * q_high.to(DTYPE)

        CTX.value.store_output(out_tensor_ptr, dequant_low, dequant_high)


@dataclass(frozen=True)
class KernelImpl_Q5_1(KernelImpl_Legacy):
    vectorized: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(True))
    chunk_size: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(32))
    num_chunks: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(1))

    @maybestaticmethod
    @triton.jit
    def dequantize_chunk_to_registers(
        base_ptrs, mask_1d, chunk_idx, CTX: tl.constexpr, DTYPE: tl.constexpr
    ):
        # 1. Load 'd', 'm' and high-bit word 'qh'
        d_ptrs = base_ptrs.to(tl.pointer_type(tl.float16))
        d = tl.load(d_ptrs, mask=mask_1d, other=0.0).to(DTYPE)

        m_ptrs = (base_ptrs + 2).to(tl.pointer_type(tl.float16))
        m = tl.load(m_ptrs, mask=mask_1d, other=0.0).to(DTYPE)

        qh_ptrs = (base_ptrs + 4).to(tl.pointer_type(tl.uint32))
        qh_word = tl.load(qh_ptrs, mask=mask_1d, other=0)

        # 2. Load 16 bytes of low-bits
        offsets_32 = tl.arange(0, 32)
        qs_byte_idx = offsets_32 % 16
        qs_ptrs = base_ptrs[:, None] + 8 + qs_byte_idx[None, :]
        qs_bytes = tl.load(qs_ptrs, mask=mask_1d[:, None], other=0)

        # 3. Extract Lows and Highs
        is_high = offsets_32[None, :] >= 16
        ql = tl.where(is_high, qs_bytes >> 4, qs_bytes) & 0x0F
        qh_bit = (qh_word[:, None] >> offsets_32[None, :]) & 1

        # 4. Combine and apply scale/bias
        q_vals = (ql | (qh_bit << 4)).to(DTYPE)

        return d[:, None] * q_vals + m[:, None]

    @maybestaticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr, out_tensor_ptr, CTX: tl.constexpr, DTYPE: tl.constexpr
    ) -> None:
        offsets_16 = tl.arange(0, 16)

        # Data layout: 2 bytes 'd', 2 bytes 'm', 4 bytes 'qh', 16 bytes 'qs'
        d_ptr = block_start_ptr
        m_ptr = block_start_ptr + 2
        qh_ptr = block_start_ptr + 4
        qs_ptr = block_start_ptr + 8

        # 1. Load the scales 'd', 'm' and the high-bit mask 'qh'.
        d = tl.load(d_ptr.to(tl.pointer_type(tl.float16))).to(DTYPE)
        m = tl.load(m_ptr.to(tl.pointer_type(tl.float16))).to(DTYPE)
        # This is a safe aligned load because TYPE_SIZE (24) and qh offset (4) are multiples of 4.
        qh_word = tl.load(qh_ptr.to(tl.pointer_type(tl.uint32)))

        # 2. Load the 16 bytes of low-bits 'qs'.
        qs_bytes_16 = tl.load(qs_ptr + offsets_16)

        # --- Process the first 16 values ---
        ql_low = qs_bytes_16 & 0x0F
        qh_low = (qh_word >> offsets_16) & 1
        q_low = (ql_low | (qh_low << 4)).to(DTYPE)
        dequant_low = d * q_low + m

        # --- Process the second 16 values ---
        ql_high = qs_bytes_16 >> 4
        qh_high = (qh_word >> (offsets_16 + 16)) & 1
        q_high = (ql_high | (qh_high << 4)).to(DTYPE)
        dequant_high = d * q_high + m

        CTX.value.store_output(out_tensor_ptr, dequant_low, dequant_high)


@dataclass(frozen=True)
class KernelImpl_Q8_0(KernelImpl_Legacy):
    vectorized: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(True))
    chunk_size: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(32))
    num_chunks: tl.constexpr = dcfield(default_factory=lambda: tl.constexpr(1))

    @maybestaticmethod
    @triton.jit
    def dequantize_chunk_to_registers(
        base_ptrs, mask_1d, chunk_idx, CTX: tl.constexpr, DTYPE: tl.constexpr
    ):
        d_ptrs = base_ptrs.to(tl.pointer_type(tl.float16))
        d = tl.load(d_ptrs, mask=mask_1d, other=0.0).to(DTYPE)

        offsets_32 = tl.arange(0, 32)
        x_ptrs = base_ptrs[:, None] + 2 + offsets_32[None, :]
        x = tl.load(
            x_ptrs.to(tl.pointer_type(tl.int8)), mask=mask_1d[:, None], other=0
        ).to(DTYPE)

        return d[:, None] * x

    @maybestaticmethod
    @triton.jit
    def dequantize_block_kernel(
        block_start_ptr, out_tensor_ptr, CTX: tl.constexpr, DTYPE: tl.constexpr
    ) -> None:
        offsets_32 = tl.arange(0, 32)
        d_ptr = block_start_ptr.to(tl.pointer_type(tl.float16), bitcast=True) + 0
        x_ptr = (
            block_start_ptr.to(tl.pointer_type(tl.int8), bitcast=True) + 2 + offsets_32
        )
        output_ptr = out_tensor_ptr + offsets_32
        d = tl.load(d_ptr).to(DTYPE)
        x = tl.load(x_ptr).to(DTYPE)
        output_ptr.store(d * x)


_type_kernel_class_map: dict[GGMLQuantizationType, type[KernelImpl]] = {
    # Legancy quants
    GGMLQuantizationType.Q4_0: KernelImpl_Q4_0,
    GGMLQuantizationType.Q4_1: KernelImpl_Q4_1,
    GGMLQuantizationType.Q5_0: KernelImpl_Q5_0,
    GGMLQuantizationType.Q5_1: KernelImpl_Q5_1,
    GGMLQuantizationType.Q8_0: KernelImpl_Q8_0,
    # K-quants
    GGMLQuantizationType.Q2_K: KernelImpl_Q2_K,
    GGMLQuantizationType.Q3_K: KernelImpl_Q3_K,
    GGMLQuantizationType.Q4_K: KernelImpl_Q4_K,
    GGMLQuantizationType.Q5_K: KernelImpl_Q5_K,
    GGMLQuantizationType.Q6_K: KernelImpl_Q6_K,
}


def build_dequantize_functions(
    **kwargs,
) -> dict[GGMLQuantizationType, KernelDefinition]:
    return {
        qtype: KernelDefinition(qtype, kernel_class=kclass, **kwargs)
        for qtype, kclass in _type_kernel_class_map.items()
    }


dequantize_functions: dict[GGMLQuantizationType, KernelDefinition] = (
    build_dequantize_functions()
)

dequantize_functions_legacy: dict[GGMLQuantizationType, KernelDefinition] = (
    build_dequantize_functions(use_vectorized=False)
)

__all__ = ("dequantize_functions", "dequantize_functions_legacy")
