# QuadrupleCLIPLoader (Dynamic VRAM)

Loads four text encoders through ComfyUI's DynamicVRAM path.

Use this node when DynamicVRAM is enabled and your workflow requires four CLIP/text-encoder inputs.

## Outputs

- **CLIP**: The combined text encoders.
- **model_name**: All four selected file names, comma-separated, as a `STRING`.
