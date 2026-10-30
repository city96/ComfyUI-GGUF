
# ComfyUI-GGUF

GGUF Quantization support for native ComfyUI models including the custom Q8_CR. 

> [!NOTE]  
> This is a fork of the original nodes, updated to support loading Ideogram 4 GGUFs and Krea 2 GGUFs. 
> To use this maintained fork, clone `https://github.com/molbal/ComfyUI-GGUF`.

While quantization was previously unfeasible for regular UNET models (conv2d), transformer/DiT models such as flux are less affected by quantization. This allows running them in lower bits per weight variable bitrate quants on GPUs with less VRAM.

More details on how to use it, pre-converted models, and sample workflows are here: [Documentation](https://molbal.github.io/gguf/ecosystem/using-the-custom-nodes.html)

For technical details on the custom `Q8_CR`  and `Q4_CR` formats , memory-mapped loading, please see [ARCHITECTURE.md](ARCHITECTURE.md).

## Installation

> [!IMPORTANT]  
> Make sure your ComfyUI is on v0.27.0 or later.

To install the custom node normally, git clone this repository into your custom nodes folder (`ComfyUI/custom_nodes`) and restart ComfyUI.

```bash
git clone https://github.com/molbal/ComfyUI-GGUF
```
  
## Usage  
  
Simply use the GGUF Unet loader found under the `bootleg` category. Place the .gguf model files in your `ComfyUI/models/unet` folder.  

Pre-quantized models (🍴 icon on ones added by this fork):  
  
- [flux1-dev GGUF](https://huggingface.co/city96/FLUX.1-dev-gguf)  
- [flux1-schnell GGUF](https://huggingface.co/city96/FLUX.1-schnell-gguf)  
- [stable-diffusion-3.5-large GGUF](https://huggingface.co/city96/stable-diffusion-3.5-large-gguf)  
- [stable-diffusion-3.5-large-turbo GGUF](https://huggingface.co/city96/stable-diffusion-3.5-large-turbo-gguf)  
- [Krea 2 (Both Turbo and Raw)](https://huggingface.co/molbal/krea2-gguf) 🍴  
- [Ideogram 4](https://huggingface.co/molbal/ideogram-4-gguf) 🍴  
- [MiniMax H3](https://huggingface.co/molbal/MiniMax-H3-GGUF) 🍴  
- [MiniMax Music3](https://huggingface.co/molbal/Minimax-Music3-GGUF) 🍴  
- [LTX 2.5](https://huggingface.co/molbal/LTX-2.5-GGUF) 🍴  
- [Qwen Image 2.1](https://huggingface.co/molbal/Qwen-Image-2.1-GGUF) 🍴  
- [Flux3 Klein](https://huggingface.co/molbal/Flux-3-Klein-GGUF) 🍴  
- [Moss Image](https://huggingface.co/molbal/MOSS-Image-GGUF) 🍴  
- [Fibo 2.0](https://huggingface.co/molbal/fibo-2.0-GGUF) 🍴  


Support for quantizing T5 has also been added, these can be used using the various `*CLIPLoader (GGUF)` nodes which can be used inplace of the regular ones. For the CLIP model, use whatever model you were using before for CLIP. The loader can handle both types of files - `gguf` and regular `safetensors`/`bin`.  
  
- [t5_v1.1-xxl GGUF](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf)  
- [Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) 🍴  
- [Qwen3-VL-32B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-VL-32B-Instruct-GGUF) 🍴  
- [Qwen3-VL-32B-Instruct-MiniMax-H3 pruned GGUFs](https://huggingface.co/nif0/Qwen3-VL-32B-Instruct-MiniMax-H3-GGUF) 🍴  
- [Qwen3.5 GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) text encoders (0.8B, 2B, 4B, 9B, and 27B) with a ComfyUI build containing Qwen3.5 TE support. Place the matching `mmproj-*.gguf` beside the text encoder for image conditioning; text-only workflows do not need it. 🍴  
- [Gemma 4 GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF) text encoders (E2B, E4B, 12B, and 31B) with ComfyUI v0.30.0 or later. 🍴


## Converting Models

Fibo 2.0 Dev and Edit Turbo BF16 checkpoints are recognized using their
repacked ComfyUI tensor layout. For example, convert either checkpoint with:

```powershell
python tools\convert.py --src "$env:USERPROFILE\Models\fibo2_dev_bf16.safetensors" --streamed --quant-type Q4_0
python tools\convert.py --src "$env:USERPROFILE\Models\fibo2_edit_turbo_bf16.safetensors" --streamed --quant-type Q4_0
```


`--streamed` converts one safetensors tensor at a time, reducing peak host RAM
for this large checkpoint. The GGUF contains only the diffusion transformer;
load the matching Qwen3.5 text encoder and FLUX.2 VAE separately, and use the
**Unet Loader (GGUF)**. MOSS-Image loading requires ComfyUI core MOSS support,
introduced in upstream commit
[`e087d8f6`](https://github.com/Comfy-Org/ComfyUI/commit/e087d8f6cb7c6cdb11b59dc7c9bf17c79f8d877d).
Standard GGML quants are dequantized for PyTorch compute; only the repository's
supported `Q8_CR` route uses native INT8 Linear execution.

This node pack includes a GGUF converter. It has 3 possible interfaces that you can use: 
- a python file you can call directly
- a web interface
- a custom node

Each option is documented here: [Quantizing models](https://molbal.github.io/gguf/ecosystem/quantizing-models.html)

Official Qwen3-VL 4B and 8B BF16 checkpoints convert only to a text-only
`Q8_CR` text-encoder GGUF; its `comfy.gguf.qwen3vl.variant` field records
which size it came from.

## Supported Conversion Formats

| Format | Storage / execution       | Recommended use                                             |
|--------|---------------------------|-------------------------------------------------------------|
| F16    | FP16 GGUF                 | Maximum compatibility with half-precision storage.          |
| BF16   | BF16 GGUF                 | Preserve BF16 source models where the target supports BF16. |
| Q8_0   | Standard GGML 8-bit       | Excellent general-quality 8-bit GGUF.                       |
| Q5_1   | Standard GGML 5-bit       | Lower storage with a quality-oriented 5-bit format.         |
| Q5_0   | Standard GGML 5-bit       | Lower storage alternative to `Q5_1`.                        |
| Q4_1   | Standard GGML 4-bit       | Smaller files when VRAM or RAM is constrained.              |
| Q4_0   | Standard GGML 4-bit       | Smallest supported format for constrained setups.           |
| Q6_K   | Uniform GGML K 6-bit      | High-quality K-quant storage for eligible Linear weights.    |
| Q5_K   | Uniform GGML K 5-bit      | Uniform 5-bit K-quant storage.                               |
| Q5_K_S / Q5_K_M | Mixed GGML K     | Deterministic 5/6-bit distributions; `_M` protects more sensitive tensors. |
| Q4_K   | Uniform GGML K 4-bit      | Uniform 4-bit K-quant storage.                               |
| Q4_K_S / Q4_K_M | Mixed GGML K     | Deterministic 4/5/6-bit distributions; `_M` protects more sensitive tensors. |
| Q8_CR  | Per-row INT8 ConvRot      | Maintainer recommendation for NVIDIA RTX 30-series systems. |
| Q4_CR | Per-row packed INT4, INT4 activations | Existing W4A4 format; `Q4_CR` remains its alias. |
| Q4_CR_L | Per-row packed INT4, INT8 activations | Opt-in W4A8 execution using the same packed weight storage. |

## K-quant acceleration

K-quant creation uses the GGML-compatible encoder bundled with
this converter and requires every selected row to be divisible by 256.
Sensitive, convolution, small, and one-dimensional tensors retain
higher precision. 

For eligible K-quant Linear operations, the experimental bundled Triton/CUDA
backend is enabled by default; unsupported routes continue to use portable
PyTorch dequantization. It accelerates decoding but still materializes a
floating-point weight and uses the same PyTorch Linear as the portable path.
Q3_K is supported on CUDA SM 8.6 with FP16/BF16 activations and up to 8192 rows;
Q5_K/Q6_K remain limited to measured LTX matrix shapes and at most 1024 rows.
Set `COMFYUI_GGUF_KQUANT_BACKEND=none` before starting ComfyUI to opt out.

See [K-quant inference notes](docs/k-quant-inference.md) for details.

## Legavy vs Dynamic VRAM

Prefer the **Dynamic VRAM** loader nodes when your ComfyUI build supports
Dynamic VRAM: they avoid per-forward dequantization overhead on large text
encoders and retain native INT8 paths where available.
