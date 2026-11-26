# CLIPLoader (Dynamic VRAM)

Loads one text encoder through ComfyUI's DynamicVRAM path.

Use this node when DynamicVRAM is enabled and the text encoder benefits from demand loading. It accepts GGUF and regular text-encoder files.

## Outputs

- **CLIP**: The loaded text encoder.
- **model_name**: The selected file name, as a `STRING`.
