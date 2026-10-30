# Unet Loader (GGUF)

Loads a diffusion model stored in GGUF format and returns a ComfyUI `MODEL`.

Place GGUF diffusion models in `ComfyUI/models/unet` or
`ComfyUI/models/diffusion_models`, then select the file with **unet_name**.

## Q8_CR models

`Q8_CR` GGUFs use ComfyUI's native INT8 ConvRot path. Eligible Linear weights
remain INT8 during inference; 1-D, small, and designated high-precision tensors
are retained in FP32, while convolution weights remain FP16.

CUDA is optional. CUDA uses the optimized ComfyUI backend when available;
non-CUDA environments use the eager backend and are slower.

## Usage

Connect **MODEL** to a sampler such as `KSampler`. Use the normal model-family
workflow inputs, including the matching text encoder and VAE.