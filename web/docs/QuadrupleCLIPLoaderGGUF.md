# Quadruple CLIP Loader (GGUF)

Loads four text encoder files and returns one combined ComfyUI `CLIP` object.

## Parameters

- **clip_name1**
- **clip_name2**
- **clip_name3**
- **clip_name4**

Each input accepts a GGUF text encoder or a compatible regular checkpoint.
The default model type is `stable_diffusion`.

## Usage

Use this node only for workflows whose model family requires four text
encoders. Connect **CLIP** to the workflow's normal text-encoding node.
