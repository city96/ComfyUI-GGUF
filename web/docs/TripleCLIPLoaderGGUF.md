# Triple CLIP Loader (GGUF)

Loads three text encoder files and returns one combined ComfyUI `CLIP` object.

## Parameters

- **clip_name1**
- **clip_name2**
- **clip_name3**

Each input accepts a GGUF text encoder or a compatible regular checkpoint.
This node uses the `sd3` model type by default.

## Usage

Use this node where an SD3-style workflow requires three text encoders. Connect
the resulting **CLIP** output to the normal text-encoding node.
