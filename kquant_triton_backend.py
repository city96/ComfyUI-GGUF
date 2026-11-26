"""Triton K-quant decoding followed by the reference PyTorch Linear."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


SUPPORTED = {"Q3_K", "Q4_K", "Q5_K", "Q6_K"}
_BLOCK_SIZE = 256


def _supports(*, qtype, device, input_dtype, weight_shape, weight_dtype):
    return (
        triton is not None
        and device.type == "cuda"
        and qtype in SUPPORTED
        and input_dtype in {torch.float16, torch.bfloat16, torch.float32}
        and len(weight_shape) == 2
        and weight_shape[1] % _BLOCK_SIZE == 0
        and weight_dtype == torch.uint8
    )


def _select_launch_config(*, qtype, input_dtype, input_shape, weight_shape, device):
    if (
        device.type != "cuda"
        or input_dtype not in {torch.float16, torch.bfloat16}
        or len(input_shape) < 2
        or len(weight_shape) != 2
    ):
        return None
    if torch.cuda.get_device_capability(device) != (8, 6):
        return None

    m_size = 1
    for dim in input_shape[:-1]:
        m_size *= dim
    max_m_size = 8192 if qtype == "Q3_K" else 1024
    if m_size <= 0 or m_size > max_m_size:
        return None

    if qtype == "Q3_K":
        return 256, 4
    if (
        qtype == "Q5_K" and tuple(weight_shape) in {(16384, 4096), (4096, 4096)}
        or qtype == "Q6_K" and tuple(weight_shape) == (4096, 16384)
    ):
        return 256, 4
    return None


def _should_use(*, qtype, device, input_dtype, input_shape, weight_shape,
                weight_dtype, dequant_dtype="target"):
    return (
        dequant_dtype in {None, "target", torch.float16, torch.bfloat16, torch.float32}
        and _select_launch_config(
            qtype=qtype,
            device=device,
            input_dtype=input_dtype,
            input_shape=input_shape,
            weight_shape=weight_shape,
        ) is not None
    )


if triton is not None:

    @triton.jit
    def _kquant_decode_kernel(
        q_ptr,
        w_ptr,
        size,
        QTYPE: tl.constexpr,
        DECODE_DTYPE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < size
        local = i % 256
        block = i // 256
        if QTYPE == 3:
            # Q3_K block: hmask[32], qs[64], scales[12], d[fp16].
            base = q_ptr + block * 110
            scale_group = local // 16
            d = tl.load(
                (base + 108).to(tl.pointer_type(tl.float16)), mask=valid, other=0
            ).to(DECODE_DTYPE).to(tl.float32)
            low_scale_byte = tl.load(
                base + 96 + scale_group % 8, mask=valid, other=0
            )
            high_scale_byte = tl.load(
                base + 104 + scale_group % 4, mask=valid, other=0
            )
            low_scale = (low_scale_byte >> ((scale_group // 8) * 4)) & 15
            high_scale = (high_scale_byte >> ((scale_group // 4) * 2)) & 3
            scale = (low_scale.to(tl.int32) | (high_scale.to(tl.int32) << 4)) - 32
            scaled = (d * scale.to(tl.float32)).to(DECODE_DTYPE).to(tl.float32)

            qs_byte = tl.load(
                base + 32 + (local // 128) * 32 + local % 32,
                mask=valid, other=0,
            )
            ql = (qs_byte >> ((local // 32 % 4) * 2)) & 3
            hmask_byte = tl.load(base + local % 32, mask=valid, other=0)
            qh = ((hmask_byte >> (local // 32)) & 1) ^ 1
            q = ql.to(tl.int32) - (qh.to(tl.int32) << 2)
            w = (scaled * q.to(tl.float32)).to(DECODE_DTYPE)
        elif QTYPE == 2:
            base = q_ptr + block * 210
            d = tl.load(
                (base + 208).to(tl.pointer_type(tl.float16)), mask=valid, other=0
            ).to(DECODE_DTYPE).to(tl.float32)
            group32 = local // 32
            within = local % 32
            ql_group = group32 % 4
            ql = tl.load(
                base + (local // 128) * 64 + (ql_group % 2) * 32 + within,
                mask=valid, other=0,
            )
            qh = tl.load(
                base + 128 + (local // 128) * 32 + within,
                mask=valid, other=0,
            )
            q = ((ql >> ((ql_group // 2) * 4)) & 15) | (
                ((qh >> (ql_group * 2)) & 3) << 4
            )
            scale = tl.load(
                base + 192 + local // 16, mask=valid, other=0
            ).to(tl.int8)
            scaled = (d * scale.to(tl.float32)).to(DECODE_DTYPE).to(tl.float32)
            w = (scaled * (q.to(tl.float32) - 32)).to(DECODE_DTYPE)
        else:
            if QTYPE == 0:
                base = q_ptr + block * 144
                offset = 16
            else:
                base = q_ptr + block * 176
                offset = 48
            d = tl.load(
                base.to(tl.pointer_type(tl.float16)), mask=valid, other=0
            ).to(DECODE_DTYPE).to(tl.float32)
            dm = tl.load(
                (base + 2).to(tl.pointer_type(tl.float16)), mask=valid, other=0
            ).to(DECODE_DTYPE).to(tl.float32)
            group = local // 32
            within = local % 32
            byte = tl.load(
                base + offset + (group // 2) * 32 + within, mask=valid, other=0
            )
            q = (byte >> ((group % 2) * 4)) & 15
            if QTYPE == 1:
                high = tl.load(base + 16 + within, mask=valid, other=0)
                q = q | (((high >> group) & 1) << 4)
            si = group % 4
            slo = tl.load(base + 4 + si, mask=valid, other=0)
            shi = tl.load(base + 12 + si, mask=valid, other=0)
            mlo = tl.load(base + 8 + si, mask=valid, other=0)
            scale = tl.where(
                group < 4, slo & 63, (shi & 15) | ((slo >> 2) & 48)
            )
            minimum = tl.where(
                group < 4, mlo & 63, (shi >> 4) | ((mlo >> 2) & 48)
            )
            scaled = (d * scale.to(tl.float32)).to(DECODE_DTYPE).to(tl.float32)
            mins = (dm * minimum.to(tl.float32)).to(DECODE_DTYPE).to(tl.float32)
            product = (scaled * q.to(tl.float32)).to(DECODE_DTYPE).to(tl.float32)
            w = (product - mins).to(DECODE_DTYPE)
        tl.store(w_ptr + i, w, mask=valid)


def linear(*, input, weight, qtype, weight_shape, bias=None, dequant_dtype="target"):
    if not _supports(
        qtype=qtype,
        device=input.device,
        input_dtype=input.dtype,
        weight_shape=weight_shape,
        weight_dtype=weight.dtype,
    ):
        raise RuntimeError("Triton K-quant backend does not support this route")
    config = _select_launch_config(
        qtype=qtype,
        input_dtype=input.dtype,
        input_shape=tuple(input.shape),
        weight_shape=weight_shape,
        device=input.device,
    )
    if config is None:
        raise RuntimeError("Triton K-quant backend has no tuned config for this route")
    block, num_warps = config
    decode_dtype = input.dtype if dequant_dtype == "target" else dequant_dtype
    decode_dtype = torch.float16 if decode_dtype is None else decode_dtype
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    if decode_dtype not in dtype_map:
        raise ValueError(f"Unsupported K-quant decode dtype: {decode_dtype}")

    weight_fp = torch.empty(weight_shape, device=input.device, dtype=input.dtype)
    _kquant_decode_kernel[(triton.cdiv(weight_fp.numel(), block),)](
        weight,
        weight_fp,
        weight_fp.numel(),
        QTYPE={"Q4_K": 0, "Q5_K": 1, "Q6_K": 2, "Q3_K": 3}[qtype],
        DECODE_DTYPE=dtype_map[decode_dtype],
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return torch.nn.functional.linear(input, weight_fp, bias)


class _Backend:
    supports = staticmethod(_supports)
    should_use = staticmethod(_should_use)
    supports_dequant_dtype = True
    linear = staticmethod(linear)


backend = _Backend()
