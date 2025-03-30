## STEP 1 (Patch files with Unix (LF) line endings

Solution to fix lines endings of the patch files from Windows (CRLF) to Unix (LF)

```
python fix_lines_ending.py
```
## STEP 2 (Clone llama.cpp version of gguf-py)

Git clone llama.cpp into the current folder. You may also install gguf-py from the llama.cpp repo directly, though the one specified in `requirements.txt` should also work on recent versions.

```
git clone https://github.com/ggerganov/llama.cpp
pip install llama.cpp/gguf-py
```

## STEP 3 (Convert to FP16 or BF16)

To convert your initial source model to FP16 (or BF16), run the following command:
```
python convert.py --src E:\models\unet\flux1-dev.safetensors
```
## STEP 4 (Patch llama.cpp)

- To quantize the model, first apply the provided patch to the llama.cpp repo you've just cloned.
```
cd llama.cpp
git checkout tags/b3962
git apply ..\lcpp.patch
```


## STEP 5 (Compile llama-quantize binary)

Then, compile the llama-quantize binary. This example uses cmake, on linux you can just use make.
```
mkdir build
cd build
cmake ..
cmake --build . --config Debug -j10 --target llama-quantize
cd ..
cd ..
```

## STEP 6 (Quantization)
Now you can use the newly build binary to quantize your model to the desired format:
```
llama.cpp\build\bin\Debug\llama-quantize.exe E:\models\unet\flux1-dev-BF16.gguf E:\models\unet\flux1-dev-Q4_K_S.gguf Q4_K_S
```


You can extract the patch again with `git diff src\llama.cpp > lcpp.patch` if you wish to change something and contribute back.

> [!WARNING] 
>For hunyuan video/wan 2.1, you will have to uncomment the block in convert.py that deals with 5D tensors. This will save a **non functional** model to disk first, that you can quantize. After quantization, run `fix_5d_tensor.py` to add back the missing key that was saved by the conversion code. You will have to edit this file to set the correct paths/architecture. This may change in the future.


> [!WARNING]  
> Do not use the diffusers UNET for flux, it won't work, use the default/reference checkpoint format. This is due to q/k/v being merged into one qkv key. You can convert it by loading it in ComfyUI and saving it using the built-in "ModelSave" node.


> [!WARNING]  
> Do not quantize SDXL / SD1 / other Conv2D heavy models. There's little to no benefit with these models. If you do, make sure to **extract the UNET model first**.
>This should be obvious, but also don't use the resulting llama-quantize binary with LLMs.
