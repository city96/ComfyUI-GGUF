# Unet Loader (GGUF/Advanced)

Loads a GGUF diffusion model with the same behavior as **Unet Loader (GGUF)**,
with additional controls for dequantization and LoRA patch handling.

## Parameters

- **unet_name**: GGUF diffusion model in `models/unet` or
  `models/diffusion_models`.
- **dequant_dtype**: Dtype used when a conventional GGUF quant needs
  dequantization. Use **default** unless troubleshooting a model.
- **patch_dtype**: Dtype used when applying weight patches such as LoRAs. Use
  **default** unless a specific workflow requires another dtype.
- **patch_on_device**: Applies patches on the model load device. Leave this off
  for the usual lower-VRAM behavior.

## Q8_CR

For `Q8_CR` files, the loader selects ComfyUI's native INT8 ConvRot path.
The advanced dtype controls do not convert native INT8 weights to floating
point.

## Usage

Connect **MODEL** to the same downstream nodes as the standard GGUF UNet loader.
Start with all advanced options set to **default**.
