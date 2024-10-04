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
