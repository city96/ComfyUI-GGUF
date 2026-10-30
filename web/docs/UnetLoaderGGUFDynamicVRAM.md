# Unet Loader (Dynamic VRAM)

Loads a GGUF diffusion model through ComfyUI's DynamicVRAM path.

DynamicVRAM keeps quantized GGUF weights in host memory and loads them on demand. It requires a ComfyUI build with DynamicVRAM enabled; use the regular GGUF loader when DynamicVRAM is unavailable.

For Q8_CR models, supported linear layers retain ComfyUI's native INT8 ConvRot layout.
