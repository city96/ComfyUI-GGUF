
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
  
  
> [!IMPORTANT] > Please note, that this fork does not support _K quants on diffusion models, only on text encoders. They may or may not load, but inference speed may be very slow. There may be other forks, or other custom nodes with better support for these quantization types.  
  
Initial support for quantizing T5 has also been added recently, these can be used using the various `*CLIPLoader (gguf)` nodes which can be used inplace of the regular ones. For the CLIP model, use whatever model you were using before for CLIP. The loader can handle both types of files - `gguf` and regular `safetensors`/`bin`.  
  
- [t5_v1.1-xxl GGUF](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf)  
- [Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) 🍴  
- [Qwen3-VL-32B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-VL-32B-Instruct-GGUF) 🍴  
- [Qwen3-VL-32B-Instruct-MiniMax-H3 pruned GGUFs](https://huggingface.co/nif0/Qwen3-VL-32B-Instruct-MiniMax-H3-GGUF) 🍴  
- [Qwen3.5 GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) text encoders (0.8B, 2B, 4B, 9B, and 27B) with a ComfyUI build containing Qwen3.5 TE support. Place the matching `mmproj-*.gguf` beside the text encoder for image conditioning; text-only workflows do not need it. 🍴  
- [Gemma 4 GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF) text encoders (E2B, E4B, 12B, and 31B) with ComfyUI v0.30.0 or later. 🍴

## Converting Models (Krea 2, Ideogram 4, MiniMax H3, MiniMax Music 3)

This node pack includes a GGUF converter. It has 3 possible interfaces that you can use: 
- a python file you can call directly
- a web interface
- a custom node

Each option is documented here: [Quantizing models](https://molbal.github.io/gguf/ecosystem/quantizing-models.html)

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
| Q8_CR  | Per-row INT8 ConvRot      | Maintainer recommendation for NVIDIA RTX 30-series systems. |
| Q4_CR  | Experimental INT4 ConvRot | Maintainer recommendation for NVIDIA RTX 30-series systems. |