# DualCLIPLoader (Dynamic VRAM)

Loads two text encoders through ComfyUI's DynamicVRAM path.

Use this node when DynamicVRAM is enabled and your workflow requires two CLIP/text-encoder inputs.

## Outputs

- **CLIP**: The combined text encoders.
- **model_name**: Both selected file names, comma-separated, as a `STRING`.
