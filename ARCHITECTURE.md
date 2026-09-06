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

For `Q4_CR_W4A4`, compatible standard LoRA and LoKr patches keep the packed
INT4 base and add an exact BF16 low-rank output correction on every platform.
They do not fuse or cache a full patched weight.

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
