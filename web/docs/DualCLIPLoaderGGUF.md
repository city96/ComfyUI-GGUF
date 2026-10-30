# Dual CLIP Loader (GGUF)

Loads two text encoder files and returns one combined ComfyUI `CLIP` object.

## Parameters

- **clip_name1**: First encoder file.
- **clip_name2**: Second encoder file.
- **type**: The model family required by the downstream diffusion model.

Files can be GGUF or compatible regular text-encoder checkpoints. Select the
same pair and model type normally used by the equivalent built-in Dual CLIP
loader.

## Usage

Connect the output to the workflow's text-encoding nodes. Both encoders must
match the selected model family.
