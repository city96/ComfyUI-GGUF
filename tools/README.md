This needs the llama.cpp version of gguf-py to work at the moment, not the pip one as that one does not have the python quantization code yet.

```
git clone https://github.com/ggerganov/llama.cpp
pip install llama.cpp/gguf-py
```


To convert your initial source model to FP16 (or BF16), run the following command:
```
python convert.py --src E:\models\unet\flux1-dev.safetensors
```


To quantize the model, first apply the provided patch to the llama.cpp repo you've just cloned. If you get a "corrupt patch" error, you may have to [change the line endings in the patch file](https://github.com/city96/ComfyUI-GGUF/issues/90#issuecomment-2323011648).
```
cd llama.cpp
git checkout tags/b3600
git apply ..\lcpp.patch
```


Then, compile the llama-quantize binary. This example uses cmake, on linux you can just use make.
```
mkdir build
cd build
cmake ..
cmake --build . --config Debug -j10 --target llama-quantize
cd ..
cd ..
```


Now you can use the newly build binary to quantize your model to the desired format:
```
llama.cpp\build\bin\Debug\llama-quantize.exe E:\models\unet\flux1-dev-BF16.gguf E:\models\unet\flux1-dev-Q4_K_S.gguf Q4_K_S
```


You can extract the patch again with `git diff src\llama.cpp > lcpp.patch` if you wish to change something and contribute back.


> [!WARNING]  
> Do not use the diffusers UNET for flux, it won't work, use the default/reference checkpoint format. This is due to q/k/v being merged into one qkv key. You can convert it by loading it in ComfyUI and saving it using the built-in "ModelSave" node.


> [!WARNING]  
> Do not quantize SDXL / SD1 / other Conv2D heavy models. There's little to no benefit with these models. If you do, make sure to **extract the UNET model first**.
>This should be obvious, but also don't use the resulting llama-quantize binary with LLMs.
