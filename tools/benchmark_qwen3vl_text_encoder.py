"""Benchmark Qwen3-VL 4B BF16 and text-only Q8_CR text encoding in ComfyUI.

Run from the ComfyUI-GGUF repository with its ComfyUI virtual environment:
    python tools\\benchmark_qwen3vl_text_encoder.py --source <BF16.safetensors> --quantized <Q8_CR.gguf>

The benchmark initializes ComfyUI Dynamic VRAM in-process. It times model load,
first-use encoding, and warm text-only encodes separately. The BF16 source is the
full multimodal checkpoint; the Q8_CR GGUF intentionally omits its vision tower.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


_LONG_PROMPT = (
    "Write a practical guide for a small community preparing for a week-long winter storm. "
    "Explain how households can organize food, drinking water, medications, flashlights, batteries, and safe heating. "
    "Include a plan for checking on older neighbors, sharing verified local updates, keeping phones charged, and deciding when travel is too dangerous. "
    "Distinguish routine preparations from urgent safety actions, and avoid inventing emergency-service instructions that vary by location. "
    "End with a short checklist that a volunteer coordinator could read aloud at a neighborhood meeting."
)
PROMPTS = (
    "A red fox standing in a snowy forest at dawn.",
    "Summarize how a bicycle's gears help a rider climb a steep hill, including how the chain moves between sprockets and why lower gears make uphill pedaling easier.",
    _LONG_PROMPT,
    " ".join((_LONG_PROMPT,) * 4),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Original Comfy-Org BF16 safetensors checkpoint.")
    parser.add_argument("--quantized", required=True, help="Text-only Q8_CR GGUF produced by this package.")
    parser.add_argument("--comfy-root", type=Path, help="ComfyUI installation root; inferred from the current checkout when omitted.")
    parser.add_argument("--runs", type=int, default=10, help="Measured encodes per prompt and model (default: 10).")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up encodes per model before timing (default: 2).")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON results report.")
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be positive and --warmup cannot be negative.")
    args.source = Path(args.source).expanduser().resolve()
    args.quantized = Path(args.quantized).expanduser().resolve()
    if not args.source.is_file():
        parser.error(f"BF16 source file does not exist: {args.source}")
    if not args.quantized.is_file():
        parser.error(f"Q8_CR GGUF does not exist: {args.quantized}")
    if args.comfy_root is None:
        args.comfy_root = Path(__file__).resolve().parents[3]
    else:
        args.comfy_root = args.comfy_root.expanduser().resolve()
    return args


def _initialize_comfy(comfy_root: Path):
    if not (comfy_root / "comfy" / "sd.py").is_file():
        raise FileNotFoundError(f"ComfyUI root is invalid: {comfy_root}")
    os.chdir(comfy_root)
    sys.path.insert(0, str(comfy_root))

    import comfy_aimdo.control

    comfy_aimdo.control.init()
    import comfy_aimdo.host_buffer as host_buffer
    if host_buffer.lib is None:
        importlib.reload(host_buffer)

    import comfy.memory_management
    import comfy.model_management
    import comfy.model_patcher
    import comfy.sd
    from comfy.cli_args import args as comfy_args

    initialized = comfy_aimdo.control.init_devices(
        (device.index, int(comfy_args.vram_headroom * 1024**3))
        for device in comfy.model_management.get_all_torch_devices()
    )
    if not initialized:
        raise RuntimeError(
            "ComfyUI Dynamic VRAM could not initialize comfy-aimdo. "
            "Run this benchmark in the ComfyUI virtual environment with a supported CUDA device."
        )

    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True

    repo_root = Path(__file__).resolve().parents[1]
    package_name = "comfyui_gguf_benchmark"
    package_spec = importlib.util.spec_from_file_location(
        package_name,
        repo_root / "__init__.py",
        submodule_search_locations=[str(repo_root)],
    )
    if package_spec is None or package_spec.loader is None:
        raise RuntimeError("Unable to load the ComfyUI-GGUF package for benchmarking.")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[package_name] = package
    package_spec.loader.exec_module(package)

    from comfyui_gguf_benchmark.nodes import _load_dynamic_gguf_clip
    from comfyui_gguf_benchmark.ops import q8_execution_stats

    return comfy, comfy.model_management, _load_dynamic_gguf_clip, q8_execution_stats


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _free_models(model_management, device: torch.device) -> None:
    model_management.unload_all_models()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _memory_mib(value: int) -> float:
    return round(value / (1024**2), 2)


def _time_encode(clip, tokens, device: torch.device):
    _synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        result = clip.encode_from_tokens(tokens)
    _synchronize(device)
    return result, (time.perf_counter() - started) * 1000


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction + 0.5)))
    return round(ordered[index], 3)


def _token_sequences(token_batch) -> list[list[tuple[int, float]]]:
    sequences = []
    for sample in next(iter(token_batch.values())):
        sequence = []
        for token in sample:
            if isinstance(token, (tuple, list)):
                token_id = int(token[0])
                weight = float(token[1]) if len(token) > 1 else 1.0
            else:
                token_id, weight = int(token), 1.0
            sequence.append((token_id, weight))
        sequences.append(sequence)
    return sequences


def _benchmark_loaded_clip(clip, device: torch.device, runs: int, warmup: int):
    encoded_tokens = [clip.tokenize(prompt) for prompt in PROMPTS]
    token_sequences = {
        prompt: _token_sequences(tokens)
        for prompt, tokens in zip(PROMPTS, encoded_tokens)
    }
    token_counts = {
        prompt: sum(len(sample) for sample in token_sequences[prompt])
        for prompt in PROMPTS
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    first_result, first_ms = _time_encode(clip, encoded_tokens[0], device)
    outputs: dict[str, torch.Tensor] = {PROMPTS[0]: first_result.detach().float().cpu()}
    for tokens in encoded_tokens:
        for _ in range(warmup):
            _time_encode(clip, tokens, device)

    latencies: dict[str, list[float]] = {prompt: [] for prompt in PROMPTS}
    for _ in range(runs):
        for prompt, tokens in zip(PROMPTS, encoded_tokens):
            result, elapsed = _time_encode(clip, tokens, device)
            latencies[prompt].append(elapsed)
            outputs[prompt] = result.detach().float().cpu()

    if device.type == "cuda":
        _synchronize(device)
        peak_encode_memory = _memory_mib(torch.cuda.max_memory_allocated(device))
        peak_encode_reserved = _memory_mib(torch.cuda.max_memory_reserved(device))
    else:
        peak_encode_memory = None
        peak_encode_reserved = None
    per_prompt = {
        prompt: {
            "token_count": token_counts[prompt],
            "median_ms": _percentile(samples, 0.5),
            "p90_ms": _percentile(samples, 0.9),
            "mean_ms": round(statistics.fmean(samples), 3),
        }
        for prompt, samples in latencies.items()
    }
    all_latencies = [elapsed for samples in latencies.values() for elapsed in samples]
    result = {
        "load_device": str(clip.patcher.load_device),
        "offload_device": str(clip.patcher.offload_device),
        "first_encode_ms": round(first_ms, 3),
        "warm_encode_median_ms": _percentile(all_latencies, 0.5),
        "warm_encode_p90_ms": _percentile(all_latencies, 0.9),
        "warm_encode_mean_ms": round(statistics.fmean(all_latencies), 3),
        "measured_encode_count": len(all_latencies),
        "peak_encode_vram_mib": peak_encode_memory,
        "peak_encode_reserved_vram_mib": peak_encode_reserved,
        "per_prompt": per_prompt,
        "token_sequences": token_sequences,
        "outputs": outputs,
    }
    return result


def _load_baseline(comfy, source_path: Path):
    import folder_paths

    return comfy.sd.load_clip(
        ckpt_paths=[str(source_path)],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
    )


def _load_q8_clip(comfy, loader, quantized_path: Path):
    return loader(
        [str(quantized_path)],
        comfy.sd.CLIPType.STABLE_DIFFUSION,
        enable_q8_native=True,
    )


def _compare_outputs(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    reference_token_sequences: dict[str, list[list[tuple[int, float]]]],
    candidate_token_sequences: dict[str, list[list[tuple[int, float]]]],
):
    metrics: dict[str, dict[str, Any]] = {}
    for prompt in PROMPTS:
        left_tensor = reference[prompt]
        right_tensor = candidate[prompt]
        if left_tensor.shape != right_tensor.shape:
            raise ValueError(
                f"Encoder outputs do not align for {prompt!r}: "
                f"{tuple(left_tensor.shape)} vs {tuple(right_tensor.shape)}"
            )
        left = left_tensor.flatten()
        right = right_tensor.flatten()
        diff = right - left
        max_index = int(diff.abs().argmax().item())
        coordinate = tuple(
            int(index.item())
            for index in torch.unravel_index(torch.tensor(max_index), left_tensor.shape)
        )
        tokenization_matches = (
            reference_token_sequences[prompt] == candidate_token_sequences[prompt]
        )
        reference_rms = torch.mean(left.square()).sqrt().item()
        rmse = torch.mean(diff.square()).sqrt().item()
        max_abs_error = diff.abs().max().item()
        reference_abs_max = left_tensor.abs().max().item()
        max_error_token_id = reference_token_sequences[prompt][coordinate[0]][coordinate[1]][0]
        metrics[prompt] = {
            "tokenization_matches": tokenization_matches,
            "tokenization_baseline_count": sum(map(len, reference_token_sequences[prompt])),
            "tokenization_q8_count": sum(map(len, candidate_token_sequences[prompt])),
            "shape": list(left_tensor.shape),
            "cosine_similarity": round(
                torch.nn.functional.cosine_similarity(left, right, dim=0).item(), 7
            ),
            "rmse": round(rmse, 7),
            "relative_l2_error": round(
                (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(left)).item(), 7
            ),
            "reference_rms": round(reference_rms, 7),
            "max_abs_error": round(max_abs_error, 7),
            "max_abs_error_percent_of_reference_abs_max": round(
                100 * max_abs_error / reference_abs_max, 4
            ) if reference_abs_max else None,
            "max_error_coordinate": list(coordinate),
            "max_error_token_id": max_error_token_id,
            "reference_at_max_error": round(left_tensor[coordinate].item(), 7),
            "q8_at_max_error": round(right_tensor[coordinate].item(), 7),
            "reference_abs_max": round(reference_abs_max, 7),
            "q8_abs_max": round(right_tensor.abs().max().item(), 7),
        }
    return metrics


def _comfy_revision(comfy_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(comfy_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    args = _parse_args()
    benchmark_argv = sys.argv
    sys.argv = [benchmark_argv[0]]
    try:
        comfy, model_management, load_dynamic_clip, q8_execution_stats = _initialize_comfy(args.comfy_root)
    finally:
        sys.argv = benchmark_argv
    device = model_management.get_torch_device()
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The Qwen3-VL Q8_CR benchmark requires an available CUDA device.")
    capability = torch.cuda.get_device_capability(device)
    baseline_outputs: dict[str, torch.Tensor] = {}
    results: dict[str, Any] = {
        "comfyui_root": str(args.comfy_root),
        "comfyui_revision": _comfy_revision(args.comfy_root),
        "device": torch.cuda.get_device_name(device),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "prompts": list(PROMPTS),
        "runs_per_prompt": args.runs,
        "warmup_count": args.warmup,
        "baseline": {
            "source": str(args.source),
            "precision": "BF16 source safetensors",
            "vision_weights_present": True,
            "vision_inference_timed": False,
        },
        "q8_cr": {
            "source": str(args.quantized),
            "format": "Q8_CR / int8_tensorwise",
            "text_only": True,
        },
    }

    print("Loading full BF16 Qwen3-VL baseline...")
    _free_models(model_management, device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    baseline_clip = _load_baseline(comfy, args.source)
    _synchronize(device)
    results["baseline"]["load_ms"] = round((time.perf_counter() - started) * 1000, 3)
    results["baseline"]["peak_load_vram_mib"] = _memory_mib(torch.cuda.max_memory_allocated(device))
    baseline = _benchmark_loaded_clip(baseline_clip, device, args.runs, args.warmup)
    baseline_outputs = baseline.pop("outputs")
    baseline_token_sequences = baseline.pop("token_sequences")
    results["baseline"].update(baseline)
    del baseline_clip
    _free_models(model_management, device)

    print("Loading text-only Q8_CR Dynamic VRAM encoder...")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    q8_clip = _load_q8_clip(comfy, load_dynamic_clip, args.quantized)
    _synchronize(device)
    results["q8_cr"]["load_ms"] = round((time.perf_counter() - started) * 1000, 3)
    results["q8_cr"]["peak_load_vram_mib"] = _memory_mib(torch.cuda.max_memory_allocated(device))
    q8_execution_stats(reset=True)
    q8 = _benchmark_loaded_clip(q8_clip, device, args.runs, args.warmup)
    q8_outputs = q8.pop("outputs")
    q8_token_sequences = q8.pop("token_sequences")
    results["q8_cr"].update(q8)
    execution_stats = q8_execution_stats(reset=True)
    results["q8_cr"]["execution_stats"] = execution_stats
    if execution_stats["native_calls"] and execution_stats["fallback_calls"]:
        route = "mixed native INT8 and dequantized higher-precision fallback"
    elif execution_stats["native_calls"]:
        route = "native TensorWiseINT8 Linear"
    elif execution_stats["fallback_calls"]:
        route = "dequantized higher-precision fallback"
    else:
        route = "no Q8 Linear calls observed"
    results["q8_cr"]["execution_route"] = route
    results["comparison"] = _compare_outputs(
        baseline_outputs,
        q8_outputs,
        baseline_token_sequences,
        q8_token_sequences,
    )
    del q8_clip
    _free_models(model_management, device)

    print(f"Q8_CR execution route: {route}")
    print(f"BF16 warm median: {results['baseline']['warm_encode_median_ms']:.3f} ms")
    print(f"Q8_CR warm median: {results['q8_cr']['warm_encode_median_ms']:.3f} ms")
    print("Output similarity:")
    for prompt, metrics in results["comparison"].items():
        print(
            f"  cosine={metrics['cosine_similarity']:.7f} "
            f"rmse={metrics['rmse']:.7f} max_abs={metrics['max_abs_error']:.7f}  {prompt}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(1)
