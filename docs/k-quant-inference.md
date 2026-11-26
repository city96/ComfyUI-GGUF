# `_K` Quantization During Inference

## Conclusion

`_K` quants reduce GGUF storage and can reduce the resident compressed-weight
footprint. Eligible Q3_K, Q5_K, and Q6_K Linear operations use the experimental
bundled Triton backend by default on supported CUDA routes. Set
`COMFYUI_GGUF_KQUANT_BACKEND=none` before starting ComfyUI to opt out and use
portable dequantization; unsupported routes always use the portable path.
The backend receives original GGUF blocks and accelerates unpacking and
dequantization before the same PyTorch `Linear` as the portable path. It is not
low-bit arithmetic and still materializes a floating-point weight. The Krea-2
measurement below is encouraging for one local GPU/workflow, but no universal
speedup should be assumed.

The initial Issue 25 Krea-2 Turbo Q3_K_M check used the requested 1024x1024,
9-step `er_sde`/`simple` workflow, but the then-current bundled backend did not
support Q3_K. Its first-run times were 105.10 s (backend off) and 103.36 s
(`bundled`), with warm medians of 65.21 s and 66.26 s; the same-seed images
were pixel-identical. Because Q3_K never reached an accelerated route, those
numbers were not a kernel A/B result.

After Q3_K support was added, the same 1024x1024, 9-step `er_sde`/`simple`
Krea-2 Turbo workflow was rerun on an RTX 3080 Laptop GPU (SM 8.6). After two
warm-ups, three measured pairs alternated backend-off and `bundled` runs; the
model, VAE, text-encoding, and latent nodes were cached for these warm prompt
measurements. Times were:

| Backend | Pair 1 | Pair 2 | Pair 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Off | 70.09 s | 86.43 s | 95.97 s | 86.43 s |
| `bundled` | 56.84 s | 56.90 s | 66.57 s | 56.90 s |

The bundled median was 34.2% lower (about 1.52x by inverse latency) in this
run. Every measured bundled run recorded 2,358 Q3_K backend calls and zero
failures. All same-seed outputs were bit-identical (maximum and mean absolute
pixel differences were both 0). The GPU showed 31% utilization before the run,
and the backend-off times varied substantially, so treat this as encouraging,
best-effort evidence for this laptop and workflow—not a controlled universal
speed claim. The backend still dequantizes to floating point before PyTorch
Linear; it is not native low-bit arithmetic.

A [report on issue #26](https://github.com/molbal/ComfyUI-GGUF/issues/26#issuecomment-6047409099)
found the bundled Triton backend about 5x slower for LTX 2.3 Q5_K_M i2v and
about 2.2x slower for t2v on Windows with an RTX 5090 (SM 12.0) and
`triton-windows 3.6.0.post26`; outputs also differed from the portable path.
Keep it disabled on that setup until a compatible backend is validated.

Earlier investigation on an RTX 3080 Laptop (SM 8.6, PyTorch 2.14.0+cu130,
Triton 3.1.0) found that the fused kernel's default staging could produce
intermittently incorrect Q6_K outputs on one-hot probes of a real Qwen3 weight.
Setting `num_stages=1` removed the observed variation and matched the portable
dequantized weights exactly in those probes. Rounding decoded intermediates
to the activation dtype matched an explicit `target` decode reference, but
did not preserve the loader's default FP16 decoding with BF16 activations.
Even with identical decoded weights, the fused matrix multiply and bias
handling differed from PyTorch's `Linear`. Compute
Sanitizer racecheck reported no hazards for the one-stage Q4_K/Q5_K/Q6_K test
cases.

Larger fused matmul tiles reduced the original slowdown, but did not solve
output equivalence. One reduced i2v run (256x256 latent, 9 frames, one denoise
step per stage) took 152.88 s with the tuned fused route versus 153.87 s with
the backend off; mean frame PSNR was 39.9 dB, not identical output.

The next round replaced the fused matmul with a single Triton decoding pass
and reference PyTorch `Linear`. This preserves PyTorch's accumulation, bias,
and reduced-precision settings rather than trying to reproduce cuBLAS in a
custom matmul. Decoder intermediates round to the requested decode dtype,
including FP16 for the default and Dynamic VRAM paths, before casting to the
activation dtype. FP32 fusion is disabled to preserve the reference's
separate multiplication and subtraction.

An alternating warm sweep used real LTX weights, CUDA events, five warmups,
and 21 measurements per implementation per case. It covered three layers,
FP16/BF16 activations, default/activation decode precision, and seven row
counts from 1 to 1024: 84 cases. Every candidate output was bit-identical to
the corresponding portable `Linear`; every candidate case was faster than
portable, with the smallest measured speedup 1.52x. Representative BF16
activation/BF16 decode results are below. These are layer timings, not
full-workflow latency:

| GGUF layer shape `(N, K)` | Rows `M` | Portable (ms) | Previous tuned fused (ms) | Triton decode + PyTorch Linear (ms) |
| --- | ---: | ---: | ---: | ---: |
| Q5_K `(4096, 4096)` attention | 128 | 2.71 | 1.11 | 0.50 |
| Q5_K `(4096, 4096)` attention | 256 | 2.38 | 1.16 | 0.59 |
| Q5_K `(16384, 4096)` FFN up | 128 | 6.95 | 4.00 | 1.63 |
| Q5_K `(16384, 4096)` FFN up | 256 | 7.23 | 4.43 | 2.12 |
| Q6_K `(4096, 16384)` FFN down | 128 | 7.33 | 2.76 | 1.95 |
| Q6_K `(4096, 16384)` FFN down | 512 | 11.30 | 7.89 | 4.83 |

Current Q5_K/Q6_K dispatch permits only those three matrix shapes, FP16/BF16
activations, CUDA SM 8.6, and 1 through 1024 activation rows. Q3_K dispatch
accepts 2-D Linear weights on CUDA SM 8.6 with FP16/BF16 activations and 1
through 8192 activation rows; its generic route is intended for workflows such
as Krea-2 and remains subject to the same decode-and-fallback behavior. Q4_K
is covered by decoder tests but not routed in production. Other GPU
capabilities, including the reporter's SM 12.0 RTX 5090, use portable fallback
even when `bundled` is selected. The Q3_K decoder and static/Dynamic VRAM paths
pass focused CUDA tests; the Krea-2 result above is best-effort evidence for
that specific local workflow and hardware, not proof of a universal or
city96-relative end-to-end speedup.

Six reduced i2v executions of the reporter's KJ-loader graph used the same
seed, example input, 256x256 initial latent, 9 output frames at 512x512, and
one denoise step per stage. One portable and one candidate warmup preceded
two alternating measured runs of each. Both sampled video latents and every
decoded image element were bit-identical across all six executions, including
the portable-versus-portable control; all captured tensors were finite.
The candidate served 832 backend calls per execution without a failure
fallback. Audio decoding and video encoding were omitted so compression
could not hide or create output differences.

| Warm reduced-workflow timing | Portable | Candidate |
| --- | ---: | ---: |
| ComfyUI prompt execution median | 103.29 s | 98.30 s |
| Instrumented sampling/upscale/video-decode pipeline median | 48.93 s | 42.34 s |

This is a measured 4.8% prompt-latency reduction in this reduced, default-cache
workflow, with only two measured runs per route. These are not verified
cached-prompt timings: a later comparison found that ComfyUI's default
RAM-pressure cache can evict nodes and repeat text encoding and model loading.
It is not a full-resolution
or RTX 5090 result and does not resolve the separate city96-relative
text-encoder regression. At the time of that earlier comparison, the backend
was disabled by default. Nineteen focused backend tests passed, including
actual static and Dynamic VRAM
operations, and CUDA memcheck found no errors in the new decoder tests.
An isolated decoder racecheck also reported zero hazards while Q4_K, Q5_K,
and Q6_K results matched CPU references across FP16/BF16/FP32 decode precision
and FP16/BF16 output precision.

A direct comparison then used city96 `6ea2651` (the reporter's revision and
upstream HEAD at measurement time), fork `e5e7ce3` with the backend disabled,
and the same fork with `bundled` enabled. The models, local ComfyUI
installation, KJ loader settings, reduced graph, seed, and input were
unchanged. An initial default-cache sweep confirmed bit-identical outputs,
but unequal cached-node sets made its timing comparison unsuitable for
isolating the execution paths.

The controlled repeat used `--cache-classic` for all three implementations
and verified the same 23 cached nodes in every measured execution, including
text encoding and model loading. Six isolated processes ran in the order
city96 / fork-portable / candidate / candidate / fork-portable / city96,
with one warmup and two measured runs per process: four measured runs per
implementation. Cold loads and warmups are excluded from the following table.
Audio decoding and video encoding remain excluded.

| Cached reduced-workflow timing | city96 `6ea2651` | Fork portable `e5e7ce3` | Candidate `e5e7ce3` |
| --- | ---: | ---: | ---: |
| ComfyUI prompt median | 46.47 s | 43.29 s | 41.48 s |
| Prompt range across four measured runs | 46.20-46.82 s | 42.02-45.28 s | 38.95-43.01 s |
| Sampling/upscale/video-decode pipeline median | 46.31 s | 43.13 s | 41.30 s |

The candidate's prompt median was 10.7% lower than city96 and 4.2% lower
than this fork's portable path in this cache-controlled reduced workflow.
Both sampled video latents and every decoded frame element were bit-identical
to city96 across all 18 executions, including warmups and repeat controls.
All tensors were finite. Candidate runs served 832 backend calls each,
without a failure fallback. The portable/candidate timing ranges overlap,
so this remains a small-sample local result rather than a universal speed
claim. It does not establish RTX 5090 performance, full-resolution LTX
performance, or text-encoder performance; text encoding was cached.

`_K` remains reasonable for text encoders when the compressed file or
CPU/offload footprint is the primary constraint and the one-time text encoding
latency is acceptable. It should not be presented as an inference-speed
optimization in this node.

## What the Formats Save

`_K` types encode 256 weights per super-block with additional per-sub-block
scales/minima. That improves quality at a given storage budget, but it makes
unpacking more complex. The GGML block definitions give these payload sizes:

| Tensor type | Bytes per 256 weights | Payload bits/weight | Storage comparison |
| --- | ---: | ---: | --- |
| `Q2_K` | 84 | 2.625 | Much smaller than FP16 |
| `Q3_K` | 110 | 3.438 | Much smaller than FP16 |
| `Q4_K` | 144 | 4.500 | Same payload density as `Q4_0` |
| `Q5_K` | 176 | 5.500 | Same payload density as `Q5_0` |
| `Q6_K` | 210 | 6.563 | Smaller than `Q8_0` |
| `Q8_0` | 272 | 8.500 | Higher-quality conventional GGML baseline |
| FP16 | 512 | 16.000 | Uncompressed compute-weight baseline |

The `_S`, `_M`, and `_L` suffixes in distribution filenames commonly describe
how a model mixes quantization choices across tensors; they are not a separate
single tensor encoding that this loader can execute differently.

## Creating K-Quant GGUFs

The CLI, local conversion dashboard, and **Targeted Quantization (GGUF)** node
offer uniform `Q4_K`, `Q5_K`, and `Q6_K`, plus mixed `Q4_K_S`, `Q4_K_M`,
`Q5_K_S`, and `Q5_K_M` selections:

```powershell
python tools\convert.py --src model.safetensors --quant-type Q4_K_M
```

Uniform choices apply one K qtype to every eligible matrix. Mixed choices use
a stable tensor-name plan and promote sensitive attention, FFN-down,
embedding, and output categories while honoring each architecture's protected
tensor rules. K rows must be divisible by 256; incompatible eligible tensors
are rejected with their key and shape. Creation uses the converter's bundled
deterministic GGML-compatible encoder, so an installed `gguf` package with
read-only K support is sufficient.
These files do not imply native low-bit execution. Eligible Linear layers may
use the optional backend described below; every other route dequantizes first.

## Runtime Paths

The portable fallback path is explicit:

1. `GGMLLayer.cast_bias_weight()` in [`ops.py`](../ops.py) calls
   `get_weight()` for every weighted operation.
2. `get_weight()` calls `dequantize_tensor()`.
3. `dequant.py` decodes `_K` blocks through PyTorch tensor operations: unpacking
   bit fields, expanding scales/minima, and materializing floating-point
   weights.
4. `torch.nn.functional.linear()` or the matching PyTorch operation then runs
   on the expanded weight.

Dynamic VRAM uses `GGMLLayout.dequantize()` in
[`quant_ops.py`](../quant_ops.py), which follows the same materialization model
for unsupported types and operations. The compressed data may remain
mmap-backed until needed, but the operation still needs a full floating-point
temporary.

For eligible 2-D Q3_K, Q5_K, and Q6_K Linear weights, both the standard
`GGMLOps.Linear` path and Dynamic VRAM's `GGMLLayout` can offer compressed
GGUF bytes to the backend when it is enabled. Weight patches/LoRAs continue to
use the portable path so ComfyUI's patch semantics remain unchanged.
Unsupported devices, dtypes, shapes, operations, unavailable backends, and
rejected kernel calls also fall back to the existing dequantized operation.

### Backend selection and custom backend contract

The bundled backend module is loaded by default when
`COMFYUI_GGUF_KQUANT_BACKEND` is unset or empty; its Triton kernels run only
on supported routes, with all other cases using portable dequantization. The
dispatcher checks backend availability before repacking any weight storage,
so declined and disabled routes perform no per-forward copies. To
opt out entirely, set the variable before starting ComfyUI; in PowerShell:

```powershell
$env:COMFYUI_GGUF_KQUANT_BACKEND = "none"
python main.py
```

`off`, `disabled`, and `false` are equivalent opt-out values. To use a custom
backend instead, set the variable to its importable Python module name. Triton
and CUDA are optional; when unavailable, or when a route is unsupported, the
bundled backend falls back to the portable reference path. A backend module may
expose itself (or `backend`) with:

```python
def supports(*, qtype, device, input_dtype, weight_shape, weight_dtype) -> bool:
    ...

def should_use(*, qtype, device, input_dtype, input_shape, weight_shape, weight_dtype) -> bool:
    ...

def linear(*, input, weight, qtype, weight_shape, bias):
    ...
```

`qtype` may be `"Q3_K"`, `"Q4_K"`, `"Q5_K"`, or `"Q6_K"`.
`weight` is contiguous, device-resident GGUF block storage; `weight_shape` is
the logical `(N, K)` matrix. `linear` must return the same shape and device as
`torch.nn.functional.linear(input, dequantized_weight, bias)`, using the input
dtype. Alternatively, the module can call `register_kquant_backend()` from
[`kquant_backend.py`](../kquant_backend.py) while it imports.

Precision-aware backends set `supports_dequant_dtype = True`; the dispatcher
then also passes `dequant_dtype` to `linear()` and optional `should_use()`.
It is a resolved `torch.dtype`, separate from the input/output dtype:
`None` from a loader resolves to FP16 and `target` resolves to the activation
dtype. Decode intermediates must use this precision before casting the
completed weight. Backends without this capability keep their original
signature, but are declined when decode and activation precision differ.
Dynamic VRAM additionally declines acceleration when the stored logical
weight dtype differs from the activation dtype, preserving the portable
operation's dtype checks.

The integration verifies the block-storage size and output contract. A
backend's optional `should_use()` can decline particular shapes without
blacklisting the route; this supports performance-based fallback. A backend
that fails initialization or a kernel call is disabled for that
quant/device/dtype combination, and subsequent calls use the portable fallback.
Backend packages remain responsible for compiled-kernel correctness, device
capability checks, stream safety, and numerical validation against GGML
dequantization.

In contrast, `Q8_CR` is converted to ComfyUI's
`TensorWiseINT8Layout`; [`get_gguf_q8_ops()`](../ops.py) retains INT8 weights
and uses ComfyUI's native INT8 Linear route. That avoids the generic GGML
dequantize-then-FP16/BF16-matmul sequence for eligible Linear layers.

## Practical Implications

| Scenario | `_K` result in this node |
| --- | --- |
| GGUF disk size / mmap-backed source weights | Lower, according to its payload bits per weight. |
| System RAM or VRAM while a layer is not resident | Often lower, especially with offload or Dynamic VRAM. |
| Peak working memory for an executing layer | Still requires a floating temporary; the largest layer is accounted for in `ops.py`. |
| Diffusion denoising latency | Usually unfavorable: every sampling step revisits many layers and repeats unpacking. |
| Text-encoder latency | May be acceptable because encoding occurs once per prompt, but measure it. |
| Output quality at a size budget | Often better than legacy quants of comparable payload size, but architecture- and model-dependent. |
| Accelerated K-quant Linear | Experimental; bundled backend is selected by default for eligible routes and can be opted out of. Triton decoding still materializes a floating weight and uses reference PyTorch Linear; unsupported routes use the portable fallback. |

The actual outcome also depends on GPU, CPU, PCIe bandwidth, batch/sequence
size, ComfyUI offload policy, and whether the model is compute- or
transfer-bound. There is no defensible universal tokens/s or seconds/step
multiplier without a benchmark on the target workflow.

## Issue #26 status

Measurement on one RTX 3090 / ComfyUI 0.41 box reproduced the legacy text
encoder gap on identical hardware and revision: city96 `6ea2651` encodes in
3.05 s (warm median of 7) while the fork took 12.3–12.6 s (4.1x). Profiler
comparison showed identical matmul/dequant kernels and call counts; the fork
alone spent ~12.6 s in device↔host copies per encode.

Root cause: `GGUFModelPatcher` unconditionally forced `force_patch_weights`
on every `partially_load`/`load`, so each `load_models_gpu` call (ComfyUI
invokes it on every text encode) took a full model unload+reload cycle. The
flag exists so GGUF quantized patches route through `patch_weight_to_device`
(the default low-VRAM adapter path is ignored by the GGML forward), but it is
only needed when reinstallation is required. Forcing is now limited to loads
with quantized-touching patches that are not currently installed
(`patch_state.requires_patch_reinstall`, covered by
`tests/test_patch_state.py`); resident models with unchanged patches take
ComfyUI's no-movement fast path. The fake VRAM-accounting state dict
additionally reports packed storage bytes (`ggml_packed_nbytes()` in
`dequant.py`, covered by `tests/test_gguf_state_size.py`), and the K-quant
dispatcher gates on backend availability before repacking weight storage
(`tests/test_kquant_probe_order.py`).

Still open, pending reporter hardware: RTX 5090 (SM 12.0) has no accelerated
K-quant route and uses the portable path; and one t2v comparison showed the
portable path ~39.5 dB from city96 while the gated kernel matched exactly,
which needs a controlled `--cache-classic` retest after this fix since
placement alone cannot change numerics.

## Recommended Choices

- **Diffusion models on NVIDIA:** Prefer `Q8_CR` for eligible transformer/DiT
  Linear weights unless a compatible K-quant backend is installed and measured
  on the target workflow.
- **Portable diffusion GGUF:** Prefer the documented standard formats
  (`Q8_0`, `Q5_0`, `Q4_0`) and choose file size versus output quality. Do not
  select `_K` expecting faster samples.
- **Text encoders under memory pressure:** `_K` can be useful if its measured
  prompt-encoding latency is acceptable. `Q4_K_M`/`Q5_K_M` are quality/storage
  candidates, not speed recommendations.
- **Loader nodes:** Prefer the Dynamic VRAM loaders when the ComfyUI build
  supports them. The legacy loaders dequantize packed weights on every
  forward; on large text encoders this is several times slower
  ([issue #26](https://github.com/molbal/ComfyUI-GGUF/issues/26)).

## Benchmark Protocol

Compare only one variable at a time: use the same source model, ComfyUI
revision, workflow, seed, prompt, resolution, sampling steps, scheduler,
offload mode, and device placement.

1. Warm up the workflow once to exclude compilation and initial allocation.
2. Run at least five measured generations for each quantization.
3. Record median wall-clock seconds per denoise step, total generation time,
   text-encoder time, peak allocated/reserved VRAM, and process RAM.
4. Repeat once with full model residency and once with the intended offload or
   Dynamic VRAM policy; transfer-bound behavior can reverse a result.
5. Compare outputs at the same seed for visible degradation before accepting a
   smaller format.
6. Include loader logs with tensor type counts and document the hardware,
   PyTorch, ComfyUI, and node revision with the result.

## Sources

- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md):
  GGUF is mmap-compatible and stores model metadata and tensors.
- [GGML quantization reference implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c):
  defines the legacy and `_K` super-block encodings and their dequantization.
- Local implementation: [`dequant.py`](../dequant.py),
  [`ops.py`](../ops.py), and [`quant_ops.py`](../quant_ops.py).
