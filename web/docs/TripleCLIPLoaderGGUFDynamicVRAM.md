# TripleCLIPLoader (Dynamic VRAM)

Loads three text encoders through ComfyUI's DynamicVRAM path.

Use this node when DynamicVRAM is enabled and your workflow requires three CLIP/text-encoder inputs.

## Outputs

- **CLIP**: The combined text encoders.
- **model_name**: All three selected file names, comma-separated, as a `STRING`.
