# Converting FLUX Models into Quantized GGUF Models

This guide provides step-by-step instructions for converting your FLUX image models into quantized GGUF models using an automated Python script. This process simplifies the conversion and quantization, allowing you to work efficiently without extensive manual steps.

---

## **Table of Contents**

- [Prerequisites](#prerequisites)
- [Using the Automated Script](#using-the-automated-script)
  - [Step 1: Prepare Your Environment](#step-1-prepare-your-environment)
  - [Step 2: Download the Script](#step-2-download-the-script)
  - [Step 3: Run the Script](#step-3-run-the-script)
    - [Basic Usage](#basic-usage)
    - [Specifying Quantization Formats](#specifying-quantization-formats)
    - [Specifying Working Directory](#specifying-working-directory)
- [Manual Conversion Steps (Deprecated)](#manual-conversion-steps-deprecated)
- [Important Notes and Warnings](#important-notes-and-warnings)
- [Troubleshooting](#troubleshooting)
- [Contributing Back](#contributing-back)
- [License](#license)

---

## **Prerequisites**

Before you begin, ensure you have the following installed on your Windows 10 machine:

1. **Python 3.10 or Later**
   - Download from [Python Downloads](https://www.python.org/downloads/windows/).
   - **Important**: Check the box that says **"Add Python 3.x to PATH"** during installation.

2. **Git for Windows**
   - Download from [Git for Windows](https://git-scm.com/download/win).

3. **CMake**
   - Download from [CMake Downloads](https://cmake.org/download/).
   - Choose the **Windows x64 Installer**.
   - **Add CMake to the system PATH** during installation.

4. **Visual Studio Build Tools 2022**
   - Download from [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/).
   - Under **"Tools for Visual Studio 2022"**, select **"Build Tools for Visual Studio 2022"**.
   - In the installer, select **"Desktop development with C++"**.

---

## **Using the Automated Script**

We've created an automated Python script that simplifies the conversion and quantization process for FLUX models. Follow the steps below to use the script.

### **Step 1: Prepare Your Environment**

1. **Place Your Model File**

   - Ensure your `.safetensors` FLUX model file is accessible.
   - Note the full path to your model file (e.g., `D:\models\flux1-dev.safetensors`).

### **Step 2: Download the Script**

1. **Obtain the `convert_flux_model.py` Script**

   - Download the script from the repository's `tools` directory.
   - Alternatively, save the following script as `convert_flux_model.py`:

     ```python
     import os
     import sys
     import subprocess
     import argparse
     import shutil
     import urllib.request
     from pathlib import Path
     from typing import List

     # List of all quantization formats
     QUANTIZATION_FORMATS = [
         'Q2_K',
         'Q3_K',
         'Q4_0',
         'Q4_1',
         'Q4_K',
         'Q4_K_S',
         'Q5_0',
         'Q5_1',
         'Q5_K',
         'Q6_K',
         'Q8_0',
         'Q8_1',
     ]

     def run_command(command: List[str], cwd: Path = None, env: dict = None):
         """Run a system command."""
         print(f"\nRunning command: {' '.join(map(str, command))}")
         result = subprocess.run(command, cwd=str(cwd) if cwd else None, env=env)
         if result.returncode != 0:
             print(f"Command failed with return code {result.returncode}")
             sys.exit(result.returncode)

     def download_file(url: str, destination: Path):
         """Download a file from a URL."""
         print(f"Downloading {url} to {destination}")
         urllib.request.urlretrieve(url, destination)

     def main():
         parser = argparse.ArgumentParser(description='Automate FLUX Model Conversion and Quantization')
         parser.add_argument('--model', required=True, help='Path to your .safetensors model file')
         parser.add_argument('--quantize', nargs='+', choices=QUANTIZATION_FORMATS + ['ALL'], default=['ALL'], help='Quantization formats to apply (default: ALL)')
         parser.add_argument('--dir', default='flux_conversion', help='Working directory (default: flux_conversion)')
         args = parser.parse_args()

         model_file = Path(args.model).resolve()
         model_name = model_file.stem
         working_dir = Path(args.dir).resolve()
         os.makedirs(working_dir, exist_ok=True)

         # Set up virtual environment
         venv_dir = working_dir / 'venv'
         if not venv_dir.exists():
             print("\nCreating virtual environment...")
             run_command([sys.executable, '-m', 'venv', str(venv_dir)])

         # Activate virtual environment
         if os.name == 'nt':
             python_executable = venv_dir / 'Scripts' / 'python.exe'
             pip_executable = venv_dir / 'Scripts' / 'pip.exe'
         else:
             python_executable = venv_dir / 'bin' / 'python'
             pip_executable = venv_dir / 'bin' / 'pip'

         # Install required Python packages
         print("\nInstalling required Python packages...")
         run_command([str(pip_executable), 'install', '--upgrade', 'pip'])
         run_command([str(pip_executable), 'install', 'torch', 'safetensors', 'tqdm'])

         # Clone llama.cpp repository
         llama_cpp_dir = working_dir / 'llama.cpp'
         if not llama_cpp_dir.exists():
             print("\nCloning llama.cpp repository...")
             run_command(['git', 'clone', 'https://github.com/ggerganov/llama.cpp'], cwd=working_dir)
         else:
             print("\nllama.cpp repository already exists.")

         # Install gguf-py from llama.cpp
         print("\nInstalling gguf-py from llama.cpp...")
         run_command([str(pip_executable), 'install', './llama.cpp/gguf-py'], cwd=working_dir)

         # Checkout specific tag and apply patch
         print("\nChecking out specific tag and applying patch...")
         run_command(['git', 'fetch', '--tags'], cwd=llama_cpp_dir)
         run_command(['git', 'checkout', 'tags/b3600'], cwd=llama_cpp_dir)

         # Download convert.py and lcpp.patch into tools directory
         tools_dir = llama_cpp_dir / 'tools'
         tools_dir.mkdir(exist_ok=True)
         convert_py_url = 'https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/tools/convert.py'
         lcpp_patch_url = 'https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/tools/lcpp.patch'

         convert_py_path = tools_dir / 'convert.py'
         lcpp_patch_path = tools_dir / 'lcpp.patch'

         if not convert_py_path.exists():
             download_file(convert_py_url, convert_py_path)

         if not lcpp_patch_path.exists():
             download_file(lcpp_patch_url, lcpp_patch_path)

         # Ensure line endings are correct (LF) for the patch file
         print("\nEnsuring correct line endings for lcpp.patch...")
         with open(lcpp_patch_path, 'rb') as file:
             content = file.read()
         content = content.replace(b'\r\n', b'\n')
         with open(lcpp_patch_path, 'wb') as file:
             file.write(content)

         # Apply the patch
         print("\nApplying patch...")
         run_command(['git', 'apply', 'tools/lcpp.patch'], cwd=llama_cpp_dir)

         # Build llama-quantize
         print("\nBuilding llama-quantize...")
         build_dir = llama_cpp_dir / 'build'
         build_dir.mkdir(exist_ok=True)
         run_command(['cmake', '..'], cwd=build_dir)
         run_command(['cmake', '--build', '.', '--config', 'Release', '--target', 'llama-quantize'], cwd=build_dir)

         # Run convert.py to create BF16 GGUF model
         print("\nConverting model to BF16 GGUF format...")
         output_bf16_path = working_dir / f'{model_name}-BF16.gguf'
         run_command([str(python_executable), 'tools/convert.py', '--src', str(model_file), '--dst', str(output_bf16_path)], cwd=llama_cpp_dir)

         # Determine quantization formats
         if 'ALL' in args.quantize:
             quant_formats = QUANTIZATION_FORMATS
         else:
             quant_formats = args.quantize

         # Build path to llama-quantize executable
         if os.name == 'nt':
             llama_quantize_exe = build_dir / 'bin' / 'Release' / 'llama-quantize.exe'
         else:
             llama_quantize_exe = build_dir / 'bin' / 'llama-quantize'

         # Ensure the llama-quantize executable exists
         if not llama_quantize_exe.exists():
             print(f"Error: llama-quantize executable not found at {llama_quantize_exe}")
             sys.exit(1)

         # Quantize the model for each selected format
         for quant_format in quant_formats:
             print(f"\nQuantizing to {quant_format} format...")
             output_quant_path = working_dir / f'{model_name}-{quant_format}.gguf'
             run_command([str(llama_quantize_exe), str(output_bf16_path), str(output_quant_path), quant_format])

         print("\nConversion and quantization complete.")
         print(f"Quantized models are located in: {working_dir}")

     if __name__ == '__main__':
         main()
     ```

### **Step 3: Run the Script**

Open Command Prompt and navigate to the directory where you've saved `convert_flux_model.py`.

#### **Basic Usage**

```cmd
python convert_flux_model.py --model "path_to_your_model_file.safetensors"
```

- **Example**:

  ```cmd
  python convert_flux_model.py --model "D:\models\flux1-dev.safetensors"
  ```

This command will:

- Set up a virtual environment.
- Install necessary Python packages.
- Clone the `llama.cpp` repository.
- Install `gguf-py` from the cloned repository.
- Download `convert.py` and `lcpp.patch`.
- Apply the patch and build `llama-quantize`.
- Convert your model to BF16 GGUF format.
- Quantize your model into **all available quantization formats**.

#### **Specifying Quantization Formats**

To specify particular quantization formats, use the `--quantize` option:

```cmd
python convert_flux_model.py --model "path_to_your_model_file.safetensors" --quantize Q4_K_S Q6_K Q8_0
```

- **Available Quantization Formats**:

  - `Q2_K`
  - `Q3_K`
  - `Q4_0`
  - `Q4_1`
  - `Q4_K`
  - `Q4_K_S`
  - `Q5_0`
  - `Q5_1`
  - `Q5_K`
  - `Q6_K`
  - `Q8_0`
  - `Q8_1`

#### **Specifying Working Directory**

To specify a different working directory, use the `--dir` option:

```cmd
python convert_flux_model.py --model "path_to_your_model_file.safetensors" --dir "path_to_working_directory"
```

---

## **Manual Conversion Steps (Deprecated)**

> **Note**: The following manual steps are deprecated in favor of the automated script. However, they are provided here for reference.

### **1. Clone the llama.cpp Repository**

```cmd
git clone https://github.com/ggerganov/llama.cpp
```

### **2. Install gguf-py**

```cmd
pip install llama.cpp/gguf-py
```

### **3. Convert Your Model to FP16/BF16**

```cmd
python convert.py --src "E:\models\unet\flux1-dev.safetensors"
```

### **4. Apply the Patch**

```cmd
cd llama.cpp
git checkout tags/b3600
git apply ..\lcpp.patch
```

> **Warning**: If you get a "corrupt patch" error, you may have to change the line endings in the patch file to Unix (LF).

### **5. Build the llama-quantize Binary**

```cmd
mkdir build
cd build
cmake ..
cmake --build . --config Release --target llama-quantize
cd ..
cd ..
```

### **6. Quantize Your Model**

```cmd
llama.cpp\build\bin\Release\llama-quantize.exe "E:\models\unet\flux1-dev-BF16.gguf" "E:\models\unet\flux1-dev-Q4_K_S.gguf" Q4_K_S
```

---

## **Important Notes and Warnings**

- **Model Compatibility**: Do not use the diffusers UNET for FLUX; it won't work due to the merging of `q/k/v` into a single `qkv` key. Use the default/reference checkpoint format.

- **Avoid Quantizing Certain Models**: Do not quantize SDXL, SD1, or other Conv2D-heavy models. There's little to no benefit with these models. If you do, ensure you **extract the UNET model first**.

- **Binary Usage**: Do not use the resulting `llama-quantize` binary with Large Language Models (LLMs).

---

## **Troubleshooting**

- **Corrupt Patch Error**: If you encounter a "corrupt patch" error when applying `lcpp.patch`, change the line endings to Unix (LF).

  - **Using VS Code**:
    - Open the file in VS Code.
    - Click on the line-ending indicator in the bottom-right corner.
    - Select `LF (Unix)`.
    - Save the file.

- **Build Errors**: Ensure that all prerequisites are installed correctly, especially the Visual Studio Build Tools for C++.

- **ModuleNotFoundError**: If you encounter a `ModuleNotFoundError` for `torch` or other packages, ensure they are installed in your virtual environment.

---

## **Contributing Back**

If you wish to change something and contribute back:

1. **Extract the Patch**:

   ```cmd
   git diff src\llama.cpp > lcpp.patch
   ```

2. **Submit a Pull Request**: Follow the standard GitHub process to fork the repository, make changes, and submit a pull request.

---

## **License**

This guide and the provided scripts are released under the Apache-2.0 License.

---

**Disclaimer**: This guide is provided as-is. Ensure you understand each step before proceeding. If you encounter issues, consider seeking assistance from communities or forums related to the software tools used.
