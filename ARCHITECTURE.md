# ComfyUI-GGUF Architecture & Technical Details

This document covers the engineering details behind ComfyUI-GGUF, including the custom `Q8_CR` native INT8 layout, target-size quantization fallback algorithms, dynamic patch behavior, and fallback caches for non-native adapter patches.

## Native Weight-Only Quantization (Q8_CR)

The converter supports one custom global quantization mode tailored for DiT/transformer UNets called `Q8_CR`. This is an INT8 weight-only format designed to reduce GGUF model storage and VRAM pressure while preserving the fast native INT8 Linear operations available on supported NVIDIA GPUs. It prevents severe VRAM bottlenecks when ComfyUI needs to offload weights to CPU memory.

During conversion, the quantization pipeline performs the following sequence:
1. Selects eligible 2-D Linear weights while intentionally excluding one-dimensional tensors, small tensors, architecture-designated sensitive tensors, and Conv2d weights to maintain FP32/FP16 precision where required.
2. Applies the compatible ConvRot/Hadamard rotation to each eligible weight matrix.
3. Quantizes the rotated weights to INT8 using an FP32 scale for every output row.
4. Stores the INT8 payload, row scales, and ConvRot metadata directly in the compiled GGUF file.

`Q8_CR` conversion accepts `--quantization-device auto`, `cpu`, or `cuda`. The `auto` flag prioritizes CUDA. If a matrix cannot fit in free VRAM, the converter logs a CPU fallback for that specific matrix without altering the overall output format. 

During load time, the GGUF loader reads the specialized metadata and passes the raw INT8 weights and row scales to ComfyUI's `TensorWiseINT8Layout`. On CUDA systems, ComfyUI natively executes the INT8/ConvRot Linear path directly without expanding the weight matrix to FP16.

### Q8_CR Platform Support

Q8_CR execution operates through ComfyUI's `comfy_kitchen` layout backend:
* NVIDIA CUDA triggers ComfyUI's optimized native INT8 backend automatically.
* Linux environments utilize the eager backend when CUDA is unavailable.
* Non-CUDA systems rely on the `comfy_kitchen` eager backend fallback.
* CPU Q8_CR loading and inference are fully supported but execute slower than hardware-accelerated CUDA passes.

### Maintainer Recommendation for NVIDIA RTX 30-Series

For Krea 2 and Ideogram 4 models on RTX 30-series architecture, `Q8_CR` offers significant benefits:
* Fast native INT8 operations routed through ComfyUI's ConvRot backend.
* Convenient CPU offload and memory-mapped model storage via the GGUF container.
* High image fidelity expected from 8-bit quantization while shielding sensitive tensors.
* Reduced VRAM pressure during complex multi-model spatial workflows.

## ConvRot INT4 Formats (Q4_CR_W4A4 and Q4_CR_L)

Both INT4 formats store the same kitchen-native ConvRot payload: packed signed
INT4 weights shaped `[N, K // 2]` in `I8` GGUF storage, one `F16` scale per
output row, and `int4_cr` metadata with `convrot_groupsize=256` and
`quant_group_size=64`. The metadata backing distinguishes the execution mode:
`w4a4` for `Q4_CR_W4A4`, or `w4a8` plus `linear_dtype="int8"` for `Q4_CR_L`.
`Q4_CR` remains an alias for `Q4_CR_W4A4`; the W4A8 variant is an explicit
converter selection and does not change the default.

`Q4_CR_L` builds the same `TensorCoreConvRotW4A4Layout` with
`linear_dtype="int8"`. On the supported CUDA backend this quantizes rotated
activations row-wise to INT8 and executes INT4-weight × INT8-activation GEMV
for small row counts or chunked INT8 GEMM for larger batches. This is a
`comfy_kitchen` CUDA route; there is no ComfyUI-GGUF Triton kernel for this
row-wise layout. The separate grouped-codebook `AsymW4A8Int8Layout` Triton
kernel uses different weight storage and is not used by `Q4_CR_L`. The
`comfy_kitchen` eager ConvRot implementation currently quantizes activations to
INT4 even when given the `int8` selector, so non-CUDA execution deliberately
uses dense dequantized PyTorch weights instead of silently running W4A4.

The custom GGUF loader validates the backing/linear-dtype metadata and selects
a corresponding operator mode. Mixed `Q8_CR` layers in a Q4 file are
materialized safely by the Q4 operator, and loader mode selection prioritizes
the Q4 backing when both custom formats are present. Unsupported or malformed
backings are rejected rather than interpreted as another format.

## Experimental Q1_CR_XL (W1A16)

`Q1_CR_XL` stores a 2D Linear matrix as little-endian packed sign bits
`[N, K // 8]` in GGUF `I8`, plus FP16 binary scales `[N, K // 64]`. Before
packing, each 256-column block is multiplied by the regular normalized
Hadamard matrix; each 64-weight group's scale is the mean absolute rotated
weight, the least-squares optimal scale for fixed `{-1, +1}` values. The
format metadata records the original dimensions, `backing="w1a16"`, and
`linear_dtype="bf16"`.

The experimental runtime rotates activations online, uses a Triton kernel to
decode the sign bits and group scales into a temporary BF16 weight matrix, then
calls PyTorch/cuBLAS Linear. This is not native binary TensorCore inference:
the temporary dense matrix raises peak memory during a layer call and the
prototype fused Triton GEMM was not fast enough to enable by default. Standard
LoRA/LoKr patches use a low-rank output correction; unsupported patch types use
the existing fused dense-weight fallback. Qwen Image 2.1 keeps conditioning,
normalization, input/output, and attention weights at higher precision
(attention uses `Q8_CR`) and applies `Q1_CR_XL` to eligible image MLP matrices.
This format is experimental; validate output quality, peak VRAM, and latency
on the target GPU before using it for production conversions.

## Experimental Accelerated GGML K-Quant Linear

Standard `Q3_K`, `Q4_K`, `Q5_K`, and `Q6_K` tensors retain their packed GGUF
bytes and eligible 2-D Linear operations use the experimental bundled
Triton/CUDA backend by default. Triton decodes packed values into a
floating-point weight, then calls the same `torch.nn.functional.linear()` as
the portable path. Intermediate rounding follows the loader's decode
precision, not necessarily the activation dtype: the default and Dynamic VRAM
paths decode in FP16 before casting, while explicit BF16, FP16, FP32, and
`target` settings retain their existing semantics. The backend is enabled when
`COMFYUI_GGUF_KQUANT_BACKEND` is unset or empty; set it to `none`, `off`,
`disabled`, or `false` before starting ComfyUI to opt out. A custom module name
selects an alternate backend.

The benchmarked selection uses FP16 or BF16 activations on CUDA SM 8.6.
Q3_K dispatch supports 2-D Linear weights and up to 8192 activation rows;
Q5_K/Q6_K dispatch remains restricted to the measured LTX shapes and at most
1024 rows. Q4_K is decoder-tested but not routed in production. Unsupported
devices, activation dtypes, shapes, operations, active weight patches/LoRAs,
and backend failures use portable dequantization. The Krea-2 Q3_K benchmark
showed lower latency on this RTX 3080 Laptop, but does not establish universal
performance or correctness on the reporter's RTX 5090. `Q2_K`, `Q8_K`, IQ
formats, and custom CR formats remain outside this route.

### K-Quant Creation

`tools/convert.py` exposes uniform `Q4_K`, `Q5_K`, and `Q6_K` selections and
the `Q4_K_S`, `Q4_K_M`, `Q5_K_S`, and `Q5_K_M` mixed presets. Mixed plans are
computed from sorted tensor names, so streamed and in-memory conversion assign
the same types. Architecture `keys_hiprec` and `keys_noquant` rules take
precedence; one-dimensional, small, and convolution tensors also retain their
safe precision. The mixed policies promote embeddings/output projections and
sensitive attention-V, attention-output, fused-QKV, and FFN-down matrices to
Q5_K or Q6_K according to the selected preset.

K encodings use 256-value blocks. The converter validates the effective row
shape before writing and rejects an incompatible tensor rather than silently
changing its qtype. Payload creation uses the converter's deterministic local
GGML-compatible encoder, so installed `gguf` builds only need container and
reader support even when their Python K encoder is read-only.

## Target-Size Quantization Algorithm

Developers can utilize `tools/convert.py --max-size-mb <MiB>` to mandate the best supported mixed quantization below a strict output size ceiling. 

The fallback logic operates as follows:
1. Core 2-D Linear weights default to native INT8 ConvRot (`Q8_CR`), preserving all protected tensors in FP32.
2. Matrices closest to the model's core center drop to `Q5_0`.
3. If further reduction is required, those core matrices drop to `Q4_0`.
4. If the target size is still unmet after all core matrices reach `Q4_0`, standard 1-D tensors are reduced to BF16 (protected architecture tensors strictly remain FP32).

`Q4_0` acts as the hard floor for core quantization. If a specified target falls below this theoretical minimum, the converter throws an error reporting the minimum achievable size. 

## LoRAs and Fused GGUF Exports

**Load LoRA (GGUF)** integrates standard GGUF adapters directly into ComfyUI's patch mechanism. It natively parses `general.type=adapter` and `adapter.type=lora` alongside `.lora_a`/`.lora_b` tensors in F32, F16, BF16, or Q8_0.

Imported GGUF LoRAs retain normal dynamic-patch behavior to ensure maximum compatibility. Because of this, an active LoRA prevents `Q8_CR` Linear layers from utilizing their native INT8 fast path. 

For `Q4_CR_W4A4` and `Q4_CR_L`, compatible standard LoRA and LoKr patches keep
the packed INT4 base and add a low-rank output correction in the active compute
dtype. They do not fuse or cache a full patched weight. The static model patcher
installs patch functions on the quantized Linear rather than applying
`calculate_weight()` to packed nibble bytes; the Dynamic VRAM patcher promotes
its `LowVramPatch`, prepares eligible adapter factors, and uses the same native
bypass. This preserves LoRA execution without corrupting packed weights.

Other patch forms that cannot use either compatible LoRA/LoKr route fall back to a compute-dtype cache:
1. It dequantizes the target and applies the patch in the compute dtype.
2. It caches the resulting patched floating-point weight while the model remains loaded.
3. It evicts the derived caches when the model or patch layout changes.

If a patched INT4 layer exhausts CUDA memory during execution, the layer retries in
system memory: the packed base is dequantized on the CPU, the adapter is applied
there, and only the completed output is copied back to the execution device. This
reduces the transient CUDA peak but cannot avoid the final output allocation needed
by the following GPU layer.

The fallback cache remains floating-point because re-quantizing a patched matrix to INT4 can erase small deltas. Unsupported patch forms and CUDA OOM retries retain the existing floating-point fallback behavior.

### Performance Diagnostics

Performance logging is off unless `COMFYUI_GGUF_PERF_LOG` is set to `1`, a
truthy value, or a log path. It synchronizes CUDA before and after every
quantized Linear to produce per-layer wall-clock timings. This intentionally
serializes work and is unsuitable for normal inference or comparative
throughput benchmarks.

For fixed adapter combinations, developers should merge adapters statically during export. Running `tools/convert.py --lora path/to/adapter.safetensors` fuses the parameters prior to quantization. Using the **Targeted Quantization (GGUF)** node's `streamed` input flag reads, fuses, quantizes, and stages one tensor block at a time to aggressively minimize peak RAM consumption.
