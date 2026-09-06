---
license: other
license_name: krea-2-community-license
license_link: https://huggingface.co/krea/Krea-2-Turbo/blob/main/LICENSE
pipeline_tag: text-to-image
tags:
  - image-generation
  - diffusion
  - flow-matching
  - dit
  - krea2
  - quantization
  - comfyui
library_name: gguf
base_model:
  - krea/Krea-2-Raw
  - krea/Krea-2-Turbo
---

# Krea 2 GGUF

Quantized GGUF diffusion transformer weights for [Krea 2](https://huggingface.co/krea/Krea-2-Turbo),
converted from the original BF16 releases for use with ComfyUI GGUF loader nodes.

This repository provides GGUF files for two checkpoints of the Krea 2 model family:

- `krea2_raw_bf16-*.gguf` — converted from [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw), the base release checkpoint.
- `krea2_turbo_bf16-*.gguf` — converted from [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo), the post-trained checkpoint with additional fine-tuning and distillation.

Krea 2 is a 12-billion parameter Diffusion Transformer with a novel architecture featuring layerwise and refiner text-fusion blocks. It is not based on Flux or any prior open-weight architecture.

These files are not a complete standalone Krea 2 package. Your workflow still needs the text encoder and VAE components.

## ComfyUI Support

Use these models with the ComfyUI nodes from [molbal/ComfyUI-GGUF](https://github.com/molbal/ComfyUI-GGUF).
Install that custom node repository into your ComfyUI `custom_nodes` folder, then restart ComfyUI.

> **Important:** This repository requires [molbal/ComfyUI-GGUF](https://github.com/molbal/ComfyUI-GGUF),
> which is a fork of [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) with added support
> for the Krea 2 architecture. The original city96 plugin does **not** support these files. (as of 2026-06-24)

Place the downloaded `.gguf` files in one of ComfyUI's diffusion model folders:

```
ComfyUI/models/diffusion_models/
ComfyUI/models/unet/
```

Load the file with `Unet Loader (GGUF)` in a Krea 2 workflow. Krea 2 uses a single transformer
(unlike Ideogram 4, there is no separate unconditional transformer component).

## Files

| Quant | Raw (base) | Turbo | Size |
|-------|-----------|-------------------|------|
| Q4_0 | krea2_raw_bf16-Q4_0.gguf | krea2_turbo_bf16-Q4_0.gguf | 7.74 GB |
| Q4_1 | krea2_raw_bf16-Q4_1.gguf | krea2_turbo_bf16-Q4_1.gguf | 8.47 GB |
| Q5_0 | krea2_raw_bf16-Q5_0.gguf | krea2_turbo_bf16-Q5_0.gguf | 9.20 GB |
| Q5_1 | krea2_raw_bf16-Q5_1.gguf | krea2_turbo_bf16-Q5_1.gguf | 9.93 GB |
| Q8_0 | krea2_raw_bf16-Q8_0.gguf | krea2_turbo_bf16-Q8_0.gguf | 13.56 GB |

Choose either the Raw or Turbo variant depending on your workflow; they are not paired with each other.

## Which Checkpoint to Use

| Checkpoint | Steps | CFG | Notes |
|------------|-------|-----|-------|
| **Turbo** | 4–8 | 0.0 | Distilled; CFG-free. Fast, good for most use cases. |
| **Raw** | 20–30 | 3.0–7.0 | Full CFG; more controllable, higher inference cost. |

The Turbo checkpoint has been post-trained with distillation and runs well at 8 steps with `CFG=1`. The Raw checkpoint behaves like a standard flow-matching DiT and benefits from more steps and positive CFG.

## When GGUFs Are Worth the Tradeoff

The BF16 source weights for Krea 2 are 26.6 GB each — far beyond what most consumer GPUs can hold entirely in VRAM. GGUFs make sense when:

- **Limited VRAM:** Q4_0 at 7.74 GB fits entirely in an 8 GB GPU; Q5_1 at 9.93 GB targets 10–12 GB cards. Running BF16 on these GPUs would require heavy CPU offloading and become impractically slow.
- **CPU offload workflows:** If you are already offloading model layers to RAM, GGUF reduces the RAM footprint proportionally alongside VRAM, which is often the actual bottleneck.
- **Acceptable quality loss at Q5+:** At Q5_0 and above the visual output of Krea 2 is very close to BF16. Q4 levels show mild softening on fine detail but remain usable for most creative tasks.

GGUFs are generally **not** worth it if you have a 24 GB+ GPU and want maximum fidelity — load the FP8 or BF16 source directly in that case.

## Download

Download the file you want from the Files tab, or use the Hugging Face CLI. For example:

```
huggingface-cli download molbal/krea2-gguf krea2_turbo_bf16-Q5_1.gguf --local-dir ComfyUI/models/diffusion_models
```

## Compatibility Notes

These are non-K GGUF quantizations intended for PyTorch dequantization in ComfyUI. K-quants are not included because this ComfyUI loading path does not use fused quantized linear kernels.

Krea 2 GGUF support requires ComfyUI to have the `krea2` architecture registered in its model detection system. If your ComfyUI installation does not recognise the checkpoint, update ComfyUI core and [molbal/ComfyUI-GGUF](https://github.com/molbal/ComfyUI-GGUF) to their latest versions.

## License

These files are derived from [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) and [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) and follow the [Krea 2 Community License](https://huggingface.co/krea/Krea-2-Turbo/blob/main/LICENSE).
