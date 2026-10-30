# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

ComfyUI users who need to convert local diffusion-model checkpoints into GGUF files and want to monitor several conversions without repeatedly composing terminal commands.

## Product Purpose

ComfyUI-GGUF loads GGUF-encoded diffusion, text, and vision models into ComfyUI and converts supported local checkpoints to GGUF. The conversion dashboard makes the existing converter easier to operate while keeping files on the user's computer.

## Operating Context

Users run the dashboard locally from the ComfyUI-GGUF checkout, provide filesystem paths to large checkpoint files, select a quantization strategy, and wait for a memory- and GPU-intensive conversion to finish.

## Capabilities and Constraints

The dashboard invokes `tools/convert.py` with the active Python interpreter. It must stay dependency-free, bind only to localhost, accept filesystem paths instead of uploading model files, and run conversions serially to avoid resource contention.

## Evidence on Hand

The repository provides `tools/convert.py`, supported quantization types, target-size conversion, device selection, streamed safetensors conversion, and documented CLI examples in `README.md`. No product-specific visual assets are available.

## Product Principles

- Preserve the converter's behavior and error messages rather than reimplementing conversion.
- Keep large model files local and visible as paths.
- Make queued, running, failed, and completed work unambiguous.
- Prefer a small, dependable operator tool over a complex deployment.
