# ComfyUI-GGUF
GGUF Quantization support for native ComfyUI models

This is currently very much WIP. These custom nodes provide support for model files stored in the GGUF format popularized by [llama.cpp](https://github.com/ggerganov/llama.cpp).

While quantization wasn't feasible for regular UNET models (conv2d), transformer/DiT models such as flux seem less affected by quantization. This allows running it in much lower bits per weight variable bitrate quants on low-end GPUs.

![Comfy_Flux1_dev_Q4_0_GGUF_1024](https://github.com/user-attachments/assets/23150750-bcb6-49ef-a78f-9c814528a640)

## Installation

To install the custom node, clone it to your custom_nodes ComfyUI folder using the following command:

```
git clone https://github.com/city96/ComfyUI-GGUF custom_nodes/ComfyUI-GGUF
```

To install the required `gguf` python package on the standalone ComfyUI, use the following:
```
.\python_embeded\python.exe -s -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-GGUF\requirements.txt
```

## Usage

Simply use the GGUF Unet loader found under the `bootleg` category. You can find [pre-quantized models for flux1-dev here](https://huggingface.co/city96/FLUX.1-dev-gguf) - place these in your `ComfyUI/models/unet` folder.
