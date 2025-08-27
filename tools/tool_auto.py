# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import re
import sys
import time
import torch
import logging
import argparse
import subprocess
import huggingface_hub as hf

logging.getLogger().setLevel(logging.DEBUG)

qtypes =[
    # "F16", "BF16",
    "Q8_0", "Q6_K",
    "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0",
    "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0",
    "Q3_K_M", "Q3_K_S", "Q2_K"
]

dtype_dict = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", "_invalid"),
    "F8_E5M2": getattr(torch, "float8_e5m2", "_invalid"),
}

# this is pretty jank but I want to be able to run it on a blank instance w/o setup
terraform_dict = {
    "repo": "city96/ComfyUI-GGUF",
    "target": "auto_convert",
    "lcpp_repo": "ggerganov/llama.cpp",
    "lcpp_target": "tags/b3962",
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source model file or huggingface repo name")
    parser.add_argument("--quants", nargs="+", choices=["all", "base", *qtypes], default=["Q8_0"])
    parser.add_argument("--output-dir", default=None, help="Location for output files, defaults to current dir or ComfyUI model dir.")
    parser.add_argument("--temp-dir", default=None, help="Location for temp files, defaults to [output_dir]/tmp")
    parser.add_argument("--force-update", action="store_true", help="Force update & rebuild entire quantization stack.")
    parser.add_argument("--resume", action="store_true", help="Skip over existing files. Will NOT check for broken/interrupted files.")

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = get_output_dir()
    if args.temp_dir is None:
        args.temp_dir = os.path.join(args.output_dir, "tmp")

    if os.path.isdir(args.temp_dir) and len(os.listdir(args.temp_dir)) > 0:
        raise OSError("Output temp folder not empty!")

    if "all" in args.quants:
        args.quants = ["base", *qtypes]

    return args

def run_cmd(*args, log_error=False):
    logging.debug(f"cmd: {args}")
    try:
        log = subprocess.run(args, capture_output=True, text=True)
    except Exception as e:
        logging.warning(f"{args[0]}, {e}")
        return -1
    if log.returncode != 0 and log_error:
        logging.warning(f"{args[0]}: {log.stdout} {log.stderr}")
    else:
        logging.debug(f"{args[0]}: {repr(log.stdout)} {repr(log.stderr.strip())} RET:{log.returncode}")
    return log.returncode

def setup_utils(force_update=False):
    # get ComfyUI-GGUF if missing, then compile patched llama.cpp if required
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(root)

    if os.path.split(root)[1] != "tools":
        cg_dir = os.path.join(root, "ComfyUI-GGUF")
        if not os.path.isdir(cg_dir):
            logging.warning(f"Running outside tools folder! Cloning to {cg_dir}")
            run_cmd("git", "clone", f"https://github.com/{terraform_dict['repo']}", cg_dir)
            need_update = True
        else:
            need_update = False

        if force_update or need_update:
            if terraform_dict['target']:
                logging.info(f"Attemtping to check out ComfyUI-GGUF branch {terraform_dict['target']}")
                run_cmd("git", "-C", cg_dir, "checkout", terraform_dict['target'])

            logging.info("Attemtping to git pull ComfyUI-GGUF to latest")
            run_cmd("git", "-C", cg_dir, "pull")

        tools_dir = os.path.join(root, "ComfyUI-GGUF", "tools")
        sys.path.append(tools_dir) # to make import(s) work
    else:
        # TODO: Git pull here too?
        logging.warning(f"Assuming latest ComfyUI-GGUF. Please git pull & check out branch {terraform_dict['target']} manually!")
        tools_dir = root

    if not os.path.isdir(tools_dir):
        raise OSError(f"Can't find tools subfoder in ComfyUI-GGUF at {tools_dir}")

    convert_path = os.path.join(tools_dir, "convert.py")
    if not os.path.isfile(convert_path):
        raise OSError(f"Cannot find convert.py at location: {convert_path}")

    lcpp_path = os.path.join(root, "llama.cpp.auto") # avoid messing with regular dir
    if not os.path.isdir(lcpp_path):
        logging.info(f"Attemtping to clone llama.cpp repo to {lcpp_path}")
        run_cmd("git", "clone", f"https://github.com/{terraform_dict['lcpp_repo']}", lcpp_path)
        need_update = True
    else:
        need_update = False

    if force_update or need_update:
        # TODO: check reflog and/or git reset before checkout?
        logging.info(f"Attemtping to check out llama.cpp target {terraform_dict['lcpp_target']}")
        run_cmd("git", "-C", lcpp_path, "checkout", terraform_dict['lcpp_target'])

        # TODO: git reset before patch?
        patch_path = os.path.join(tools_dir, "lcpp.patch")
        # patch (probably) has wrong file endings:
        logging.info("Converting patch file endings")
        with open(patch_path, "rb") as file:
            content = file.read().replace(b"\r\n", b"\n")
        with open(patch_path, "wb") as file:
            file.write(content)

        if run_cmd("git", "-C", lcpp_path, "apply", "--check", "-R", patch_path) != 0:
            logging.info("Attemtping to apply patch to llama.cpp repo")
            run_cmd("git", "-C", lcpp_path, "apply", patch_path)
        else:
            logging.info("Patch already applied")

    # using cmake here as llama.cpp switched to it completely for new versions
    if os.name == "nt":
        bin_path = os.path.join(lcpp_path, "build", "bin", "debug", "llama-quantize.exe")
    else:
        bin_path = os.path.join(lcpp_path, "build", "bin", "llama-quantize")

    if not os.path.isfile(bin_path) or force_update or need_update:
        if run_cmd("cmake", "--version") != 0:
            raise RuntimeError("Can't find cmake! Make sure you have a working build environment set up")

        build_path = os.path.join(lcpp_path, "build")
        os.makedirs(build_path, exist_ok=True)
        logging.info("Attempting to build llama.cpp binary from source")
        run_cmd("cmake", "-B", build_path, lcpp_path)
        run_cmd("cmake", "--build", build_path, "--config", "Debug", "-j4", "--target", "llama-quantize")
        if not os.path.isfile(bin_path):
            raise RuntimeError("Build failed! Rerun with --debug to see error log.")
    else:
        logging.info("Binary already present")

    return bin_path

def get_output_dir():
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(root)
    split = os.path.split(root)
    while split[1]:
        if split[1] == "ComfyUI":
            if os.path.isdir(os.path.join(*split, "models", "unet")): # new
                root = os.path.join(*split, "models", "unet", "gguf")
                logging.info(f"Found ComfyUI, using model folder: {root}")
                return root

            if os.path.isdir(os.path.join(*split, "models", "diffusion_models")): # old
                root = os.path.join(*split, "models", "diffusion_models", "gguf")
                logging.info(f"Found ComfyUI, using model folder: {root}")
                return root

            logging.info("Found ComfyUI, but can't find model folder")
            break

        split = os.path.split(split[0])

    root = os.path.join(root, "models")
    logging.info(f"Defaulting to [script dir]/models: {root}")
    return root

def get_hf_fake_sd(repo, path, device=torch.device("meta")):
    sd = {}
    meta = hf.parse_safetensors_file_metadata(repo, path)
    for key, raw in meta.tensors.items():
        shape = tuple(raw.shape)
        dtype = dtype_dict.get(raw.dtype, torch.float32)
        sd[key] = torch.zeros(shape, dtype=dtype, device=device)
    return sd

def get_hf_file_arch(repo, path):
    pattern = r'(\d+)-of-(\d+)'
    match = re.search(pattern, path)

    if match:
        # we need to load it as multipart
        if int(match.group(1)) != 1:
            return None
        sd = {}
        for k in range(int(match.group(2))):
            shard_path = path.replace(match.group(1), f"{k+1:0{len(match.group(1))}}")
            sd.update(get_hf_fake_sd(repo, shard_path))
    else:
        sd = get_hf_fake_sd(repo, path)

    # this should raise an error on failure
    sd = strip_prefix(sd)
    model_arch = detect_arch(sd)

    # this is for SDXL and SD1.5, I want to overhaul this logic to match sd.cpp eventually
    assert not model_arch.shape_fix, "Model uses shape fix (SDXL/SD1) - unsupported for now."
    return model_arch.arch

def get_hf_valid_files(repo):
    # TODO: probably tweak this?
    MIN_SIZE_GB = 1
    VALID_SRC_EXTS = [".safetensors", ] # ".pt", ".ckpt", ]
    meta = hf.model_info(repo, files_metadata=True)

    valid = {}
    for file in meta.siblings:
        path = file.rfilename
        fname = os.path.basename(path)
        name, ext = os.path.splitext(fname)

        if ext.lower() not in VALID_SRC_EXTS:
            logging.debug(f"Invalid ext: {path} {ext}")
            continue

        if file.size / (1024 ** 3) < MIN_SIZE_GB:
            logging.debug(f"File too small: {path} {file.size}")
            continue

        try:
            arch = get_hf_file_arch(repo, path)
        except Exception as e:
            logging.warning(f"Arch detect fail: {e} ({path})")
        else:
            if arch is not None:
                valid[path] = arch
                logging.info(f"Found '{arch}' model at path {path}")
    return valid

def make_base_quant(src, output_dir, temp_dir, final=True, resume=True):
    name, ext = os.path.splitext(os.path.basename(src))
    if ext == ".gguf":
        logging.info("Input file already in gguf, assuming base quant")
        return None, src, None

    name = name.lower() # uncomment to preserve case in all quants
    dst_tmp = os.path.join(temp_dir, f"{name}-{{ftype}}.gguf") # ftype is filled in by convert.py

    tmp_path, model_arch, fix_path = convert_file(src, dst_tmp, interact=False, overwrite=False)
    dst_path = os.path.join(output_dir, os.path.basename(tmp_path))
    if os.path.isfile(dst_path):
        if resume:
            logging.warning("Resuming with interrupted base quant, may be incorrect!")
            return dst_path, tmp_path, fix_path
        raise OSError(f"Output already exists! Clear folder? {dst_path}")

    if fix_path is not None and os.path.isfile(fix_path):
        quant_source = tmp_path
        if final:
            apply_5d_fix(tmp_path, dst_path, fix=fix_path, overwrite=False)
        else:
            dst_path = None
    else:
        fix_path = None
        if final:
            os.rename(tmp_path, dst_path)
            quant_source = dst_path
        else:
            dst_path = None
            quant_source = tmp_path

    return dst_path, quant_source, fix_path

def make_quant(src, output_dir, temp_dir, qtype, quantize_binary, fix_path=None, resume=True):
    name, ext = os.path.splitext(os.path.basename(src))
    assert ext.lower() == ".gguf", "Invalid input file"

    src_qtext = [x for x in ["-F32.gguf", "-F16.gguf", "-BF16.gguf"] if x in src]
    if len(src_qtext) == 1:
        tmp_path = os.path.join(
            temp_dir,
            os.path.basename(src).replace(src_qtext[0], f"-{qtype.upper()}.gguf")
        )
    else:
        tmp_path = os.path.join(
            temp_dir,
            f"{name}-{qtype.upper()}.gguf"
        )
    tmp_path = os.path.abspath(tmp_path)
    dst_path = os.path.join(output_dir, os.path.basename(tmp_path))
    if os.path.isfile(dst_path):
        if resume:
            return dst_path
        raise OSError("Output already exists! Clear folder?")

    r = run_cmd(quantize_binary, src, tmp_path, qtype, log_error=True)
    time.sleep(2) # leave time for file sync?
    if r != 0:
        raise RuntimeError(f"Quantization failed with error code {r}")

    if fix_path is not None:
        apply_5d_fix(tmp_path, dst_path, fix=fix_path, overwrite=False)
        if os.path.isfile(dst_path) and os.path.isfile(tmp_path):
            os.remove(tmp_path)
    else:
        os.rename(tmp_path, dst_path)

    return dst_path

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    quantize_binary = setup_utils(args.force_update)

    try:
        from convert import detect_arch, strip_prefix, convert_file
        from fix_5d_tensors import apply_5d_fix
    except [ImportError, ModuleNotFoundError] as e:
        raise ImportError(f"Can't import required utils: {e}")

    if not os.path.isfile(args.src):
        # huggingface repo. TODO: file choice
        if len(args.src.split("/")) != "1":
            raise OSError(f"Invalid huggingface repo or model path {args.src}")
        raise NotImplementedError("HF not yet supported")
        # download then set to temp file
        # hf_repo = "Lightricks/LTX-Video" # "fal/AuraFlow-v0.3"
        # get_hf_valid_files(hf_repo)
        # args.src = ...

    out_files = []

    base_quant, quant_source, fix_path = make_base_quant(
        args.src,
        args.output_dir,
        args.temp_dir,
        final=("base" in args.quants),
        resume=args.resume,
    )
    if "base" in args.quants:
        args.quants = [x for x in args.quants if x not in ["base"]]
    if base_quant is not None:
        out_files.append(base_quant)

    for qtype in args.quants:
        out_files.append(make_quant(
            quant_source,
            args.output_dir,
            args.temp_dir,
            qtype,
            quantize_binary,
            fix_path,
            resume=args.resume,
        ))

    if fix_path is not None and os.path.isfile(fix_path):
        os.remove(fix_path)

    if base_quant != quant_source:
        # make sure our quant source is in the temp folder before removing it
        cc = os.path.commonpath([os.path.normpath(quant_source), os.path.normpath(args.temp_dir)])
        if cc == os.path.normpath(args.temp_dir):
            os.remove(quant_source)

    out_file_str = '\n'.join(out_files)
    logging.info(f"Output file(s): {out_file_str}")
