---
name: add-model-architecture
description: Add safe, tested support for a new ComfyUI model architecture and its GGUF conversion path.
---

# Add a Model Architecture

Use this skill when adding or investigating a new diffusion, text, or vision
model architecture for ComfyUI-GGUF.

## Goal

Deliver end-to-end support: a source checkpoint is recognized, converted with
correct precision choices, accepted by this node's loader, and recognized by
the target ComfyUI installation. Do not label a converter-only match as
supported.

## Discovery

1. Identify the model role: diffusion model, text encoder, vision encoder, or
   multimodal projector.
2. Obtain an authoritative checkpoint key listing with shapes and dtypes. Use
   a safetensors header when possible; do not download model weights merely to
   infer names.
3. Verify the current ComfyUI source can detect the model and instantiate its
   runtime class. Record the minimum compatible ComfyUI revision if it is new.
4. Compare the key layout against every existing `Model*` class in
   `tools/convert.py`. Reuse an existing architecture only when its detection,
   precision rules, and runtime behavior all apply.

## Conversion Support

1. Add a focused `Model<Architecture>` subclass in `tools/convert.py`.
2. Set `arch` to the ComfyUI/GGUF architecture identifier expected at load
   time.
3. Define `keys_detect` with multiple stable, distinctive keys. Use alternate
   key sets only for known checkpoint export variants.
4. Add `keys_banned` when a similarly named incompatible checkpoint format
   exists, such as a Diffusers export with incompatible fused projections.
5. Classify tensors before quantization:
   - `keys_hiprec`: must remain FP32 due to numerical sensitivity, buffers, or
     ComfyUI runtime requirements.
   - `keys_noquant`: retain source FP16/BF16 because native low-bit execution
     is unsafe or slower.
   - `keys_ignore`: omit conversion-only state that is not model weight data.
6. Add the class to `arch_list`. Confirm `handle_tensors` preserves original
   shapes and that all quantized dimensions satisfy the selected GGML block
   size.

## Loading Support

1. Add the architecture to `IMG_ARCH_LIST`, `TXT_ARCH_LIST`, or
   `VIS_TYPE_LIST` in `loader.py`, as appropriate.
2. Add a key mapper, tokenizer loader, or detection marker only when a real
   naming or ComfyUI-detection mismatch requires it. Keep such transformations
   deterministic and covered by a test.
3. Check standard and Dynamic VRAM loading. Dynamic loading preserves GGML
   storage through `quant_ops.py`; static loading uses `ops.py`.
4. Do not enable `_K` diffusion quantization as a performance optimization:
   this repository currently expands standard GGML quants before PyTorch
   compute. Prefer `Q8_CR` for supported native INT8 Linear inference.

## Tests and Documentation

1. Add a synthetic test to `tests/test_targeted_quantization.py` that verifies
   architecture detection from the distinctive keys.
2. Convert a minimal state dict and assert `general.architecture`, selected
   GGML tensor types, and required FP32/FP16 exceptions.
3. Run `python -m unittest tests.test_targeted_quantization`.
4. Validate one real checkpoint in ComfyUI with a fixed workflow and inspect
   loader logs for tensor types and unexpected-key failures.
5. Update `README.md` with supported model variants, minimum ComfyUI version,
   conversion command, and quantization limitations.

## Completion Checklist

- [ ] ComfyUI support is present and its minimum version is documented.
- [ ] Detection uses distinctive keys and rejects incompatible formats.
- [ ] Sensitive tensors have explicit precision treatment.
- [ ] Static and Dynamic VRAM loaders accept the generated GGUF.
- [ ] Synthetic conversion tests pass.
- [ ] A real model loads and produces output in ComfyUI.
- [ ] User-facing documentation makes no unmeasured performance claim.
