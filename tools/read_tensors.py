#!/usr/bin/python3
"""
read_tensors.py - list tensors in a GGUF file, optionally comparing two files.

Single-file mode:
    python read_tensors.py model.gguf [--quantized-only]

    Lists every tensor with its quant type and shape. By default this
    INCLUDES F32 tensors (unlike the original version), since those are
    exactly the ones that can silently suffer a dimension collapse when
    quantizing (see below). Pass --quantized-only to restore the old
    behavior of skipping F32 tensors. Tensors with a suspicious shape
    (see is_suspicious_shape) are flagged inline.

Compare mode (recommended when you have both files):
    python read_tensors.py before.gguf after.gguf

    Compares two GGUF files tensor-by-tensor (e.g. the pre-quantize
    BF16/F16 GGUF from convert.py vs. the output of llama-quantize).
    Flags any tensor whose number of dimensions changed between the two
    files -- this is the exact, unambiguous signature of a GGML
    ggml_n_dims() collapse (a tensor with a trailing `ne` dim of 1 gets
    silently truncated to fewer dims when llama.cpp reloads it into a
    ggml_tensor and writes it back out) -- e.g. SD3's pos_embed,
    AuraFlow's positional_encoding/register_tokens, Wan's .modulation
    tensors, or Krea2's txtfusion.projector.weight. This mode doesn't
    rely on any heuristic, so prefer it whenever you have both files.

    Any tensor flagged here is a candidate for a gguf_set_tensor_ndim()
    fix in lcpp.patch, inside the llama_model_quantize_internal()
    tensor-writing loop.

Batch/log-friendly compare mode:
    python read_tensors.py --verify before.gguf after.gguf

    Same check as compare mode, but prints exactly one line -- "OK" or
    "FAIL" plus a compact issue list -- instead of the full per-tensor
    table. No --quantized-only equivalent needed here since the ndim
    check inherently only concerns itself with what actually differs.
    Skips the interactive "press enter to close" pause, and sets the
    process exit code (0 = OK, 1 = FAIL) so it plays well in scripts.
    Intended for looping over several quant types and collecting one
    line per run into a shared log file, e.g.:

        for QUANT in Q4_0 Q5_0 Q8_0 Q2_K Q5_K_M Q6_K; do
            ./llama-quantize model-BF16.gguf model-${QUANT}.gguf ${QUANT}
            python read_tensors.py --verify model-BF16.gguf model-${QUANT}.gguf \\
                >> quant_test_results.log
        done
"""
import os
import sys
import gguf


def shape_str(tensor):
    return "x".join(str(d) for d in tensor.shape)


def is_suspicious_shape(shape):
    """
    Flags shapes with a dimension of size 1 anywhere except shape[0].

    gguf.GGUFReader reports tensor.shape in GGML `ne` order, i.e.
    reversed relative to the original torch/numpy shape. For example
    txtfusion.projector.weight was (1, 12) in the safetensors file, but
    shows up here as (12, 1) -- the "1" moved to the end. llama.cpp's
    C++ quantizer determines a tensor's dimensionality via ggml_n_dims(),
    which scans the `ne` array from the highest index downward and drops
    trailing 1s -- so a 1 anywhere in shape[1:] is at risk of being
    silently collapsed once the tensor passes through llama-quantize,
    even though the GGUF written by convert.py still has the correct
    shape. A 1 at shape[0] is safe, since the scan stops there.

    Only usable as an early warning on a single file (typically the
    *-BF16.gguf straight out of convert.py); prefer compare mode when
    you have both the pre- and post-quantize files, since that checks
    the actual outcome instead of guessing.
    """
    dims = list(shape)
    if len(dims) < 2:
        return False
    return any(d == 1 for d in dims[1:])


def print_single(path, quantized_only):
    reader = gguf.GGUFReader(path)
    n_flagged = 0
    for tensor in reader.tensors:
        if quantized_only and tensor.tensor_type == gguf.GGMLQuantizationType.F32:
            continue
        suspicious = is_suspicious_shape(tensor.shape)
        flag = ""
        if suspicious:
            flag = "  <-- dim other than shape[0] is 1, check for ndim collapse risk"
            n_flagged += 1
        print(f"{str(tensor.tensor_type):26} {shape_str(tensor):18}: {tensor.name}{flag}")
    if n_flagged:
        print(f"\n{n_flagged} tensor(s) flagged as at risk of a ggml_n_dims() collapse.")


def print_compare(path_a, path_b, verify_only=False):
    tensors_a = {t.name: t for t in gguf.GGUFReader(path_a).tensors}
    tensors_b = {t.name: t for t in gguf.GGUFReader(path_b).tensors}

    names = sorted(set(tensors_a) | set(tensors_b))
    ndim_mismatches = []
    missing = []

    for name in names:
        a = tensors_a.get(name)
        b = tensors_b.get(name)

        if a is None:
            missing.append(f"{name} (only in B)")
            if not verify_only:
                print(f"only in B ({path_b}): {name}")
            continue
        if b is None:
            missing.append(f"{name} (only in A)")
            if not verify_only:
                print(f"only in A ({path_a}): {name}")
            continue

        flag = ""
        if len(a.shape) != len(b.shape):
            flag = "  <-- DIM COUNT CHANGED (ggml_n_dims collapse!)"
            ndim_mismatches.append(f"{name} (dim {len(a.shape)}->{len(b.shape)})")

        if not verify_only:
            print(
                f"{name:55} A: {str(a.tensor_type):10} {shape_str(a):14}  "
                f"B: {str(b.tensor_type):10} {shape_str(b):14}{flag}"
            )

    issues = ndim_mismatches + missing

    if verify_only:
        # One compact line per run -- meant to be piped/redirected into a
        # shared log file across a batch of quantization types, e.g.:
        #   python read_tensors.py --verify base.gguf quant.gguf >> quant_test.log
        status = "OK  " if not issues else "FAIL"
        summary = f"{status} {os.path.basename(path_b):40} ({len(names)} tensors, {len(issues)} issue(s))"
        if issues:
            summary += ": " + "; ".join(issues)
        print(summary)
        return not issues

    print()
    if ndim_mismatches:
        print(f"!!! {len(ndim_mismatches)} tensor(s) changed dimensionality between the two files:")
        for name in ndim_mismatches:
            print(f"  - {name}")
        print("These are prime suspects for needing gguf_set_tensor_ndim() in lcpp.patch.")
    else:
        print("No dimensionality changes detected between the two files.")
    return not issues


if __name__ == "__main__":
    raw_args = sys.argv[1:]
    quantized_only = "--quantized-only" in raw_args
    verify_only = "--verify" in raw_args
    paths = [a for a in raw_args if not a.startswith("--")]

    try:
        assert len(paths) in (1, 2), "Usage: read_tensors.py <file.gguf> [file_to_compare.gguf] [--quantized-only] [--verify]"
        assert not (verify_only and len(paths) != 2), "--verify requires two files to compare"
        for p in paths:
            assert os.path.isfile(p), f"Invalid path: {p}"
    except Exception as e:
        input(f"failed: {e}")
        sys.exit(1)
    else:
        if len(paths) == 1:
            print(f"input: {paths[0]}")
            print_single(paths[0], quantized_only)
            input()
        else:
            if verify_only:
                ok = print_compare(paths[0], paths[1], verify_only=True)
                sys.exit(0 if ok else 1)
            print(f"comparing:\n  A: {paths[0]}\n  B: {paths[1]}\n")
            print_compare(paths[0], paths[1])
            input()
