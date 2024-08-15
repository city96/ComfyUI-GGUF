This needs the llama.cpp version of gguf-py to work at the moment, not the pip one as that one does not have the python quantization code yet.

```
git clone https://github.com/ggerganov/llama.cpp
pip install llama.cpp/gguf-py
```

To quantize:
```
python convert.py --src ~/ComfyUI/models/unet/flux1-dev.safetensors --dst ~/ComfyUI/models/unet/flux1-dev-Q4_0.gguf --qtype Q4_0
```

Working quant types: Q4_0, Q5_0, Q8_0, F16

> [!WARNING]  
> Do not use the diffusers UNET for flux, it won't work, use the default checkpoint that comes with the model or convert it.

> [!IMPORTANT]  
> The model format is very much WIP. I don't recommend uploading the model files created with this method anywhere until proper metadata is added, although the key/quantization format is unlikely to change.
