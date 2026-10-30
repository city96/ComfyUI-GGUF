# Repository Guide for Agents

## Purpose

ComfyUI-GGUF loads GGUF-encoded diffusion, text, and vision models into
ComfyUI. It also converts supported checkpoint layouts to GGUF. Preserve
ComfyUI compatibility and model output quality over file-size reductions.

## Repository Map

| Area | Responsibility |
| --- | --- |
| `loader.py` | Reads GGUF metadata and tensors, maps text/vision checkpoints, and selects loader behavior. |
| `ops.py` | Defines runtime tensor wrappers and on-the-fly dequantization or native INT8 execution. |
| `dequant.py` | PyTorch implementations of GGML block dequantizers. |
| `quant_ops.py` | Dynamic-VRAM `GGMLLayout` integration. |
| `tools/convert.py` | Detects checkpoint architectures and writes GGUF files. |
| `nodes.py` | ComfyUI node definitions and loading/conversion entry points. |
| `tests/test_targeted_quantization.py` | Unit and integration coverage for conversion and loader detection. |

## Working Rules

- Treat `general.architecture`, tensor names, tensor shapes, dtypes, and GGML
  quantization types as compatibility contracts. Reject unsupported inputs with
  clear errors rather than guessing.
- Add a model architecture in `tools/convert.py` only after confirming its
  checkpoint key layout and that the installed ComfyUI can detect and run it.
  A converter-only match is not usable model support.
- Protect non-Linear, numerically sensitive, and architecture-specific tensors
  with `keys_hiprec` or `keys_noquant`. Do not quantize Conv2d weights merely
  because they are two-dimensional after reshaping.
- Standard GGML quants are dequantized before PyTorch compute in this project.
  Do not describe them as native low-bit inference. `Q8_CR` is the supported
  native INT8 Linear path.
- Keep static and Dynamic VRAM behavior aligned. A new quantization type must
  be supported by both `dequant.py` and `quant_ops.py`, or be rejected.
- Keep changes focused. Do not alter user-owned working-tree changes, generated
  files, or model assets.

## Validation

Run the focused suite from the repository root when dependencies are available:

```powershell
python -m unittest tests.test_targeted_quantization
```

For a new architecture, add a minimal synthetic checkpoint test that verifies
detection, intended protected-tensor precision, and GGUF metadata. Validate a
real checkpoint in ComfyUI before advertising support.

## Documentation

Update `README.md` when user-visible model support, conversion options, or
quantization behavior changes. Keep performance statements qualified by the
actual runtime path and hardware; do not publish unmeasured speed claims.
