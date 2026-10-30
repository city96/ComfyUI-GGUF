# CLIP Loader (GGUF)

Loads one text encoder from a GGUF model file and returns a ComfyUI `CLIP`.

## Parameters

- **clip_name**: A GGUF text encoder from `models/clip` or
  `models/text_encoders`. Regular compatible checkpoint files are also listed.
- **type**: Select the model family expected by the workflow, such as
  `stable_diffusion`, `sd3`, or `krea2`.

## Usage

Connect **CLIP** to `CLIP Text Encode` nodes. The selected **type** must match
the diffusion model; for example, Krea 2 workflows require the `krea2` type.

GGUF text encoders are loaded with the package's GGML operations and may use
less VRAM than full-precision checkpoints.
