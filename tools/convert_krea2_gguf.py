"""
Batch GGUF converter for Krea-2 (Base and Turbo) models.

Produces Q4_0, Q4_1, Q5_0, Q5_1, and Q8_0 GGUF files from a single
BF16/F16 safetensors source file.  The converter auto-detects architecture
(ModelKrea2 or ModelIdeogram) from the tensor keys in the source file.

Usage
-----
# Convert all five quant levels from a BF16 safetensors:
python tools/convert_krea2_gguf.py --src krea2_base_bf16.safetensors

# Convert specific quant levels only:
python tools/convert_krea2_gguf.py --src krea2_turbo_bf16.safetensors --quant Q5_0 Q8_0

# Explicit output directory:
python tools/convert_krea2_gguf.py --src krea2_base_bf16.safetensors --outdir /path/to/output

Notes
-----
- 1-D tensors (biases, norms, scales) are always kept in F32 regardless of
  the requested quant type.
- Tensors with fewer than 1024 elements are kept in F32.
- No model weights are downloaded by this script.  You must supply the source
  safetensors file yourself.
- This script calls convert_file() from tools/convert.py; both files must be
  on the Python path (they are when run from the repo root or from tools/).

Approximate output sizes per quant level (Krea-2 ~24 B parameters):
  Q4_0 ~ 14 GB   Q4_1 ~ 15 GB   Q5_0 ~ 17 GB   Q5_1 ~ 18 GB   Q8_0 ~ 26 GB
"""

import os
import sys
import logging
import argparse
from tqdm import tqdm

# Allow running directly from tools/ or from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from convert import convert_file, QUANT_TYPE_MAP  # noqa: E402  (tools/convert.py)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_QUANTS = ["Q4_0", "Q8_CR", "Q8_0"]


def build_dst_path(src: str, quant: str, outdir: str | None) -> str:
    """Derive an output path like <stem>-Q4_0.gguf next to the source."""
    stem = os.path.splitext(os.path.basename(src))[0]
    filename = f"{stem}-{quant}.gguf"
    directory = outdir if outdir else os.path.dirname(os.path.abspath(src))
    return os.path.join(directory, filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-convert a Krea-2 safetensors file to multiple GGUF quant levels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Path to the source BF16/F16 safetensors (or ckpt) file.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directory to write GGUF files into. Defaults to the same directory as --src.",
    )
    parser.add_argument(
        "--quant",
        nargs="+",
        choices=list(QUANT_TYPE_MAP.keys()),
        default=DEFAULT_QUANTS,
        metavar="QUANT",
        help=(
            "One or more quantization levels to produce. "
            f"Choices: {list(QUANT_TYPE_MAP.keys())}. "
            f"Default: {DEFAULT_QUANTS}"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files without prompting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.src):
        logging.error(f"Source file not found: {args.src}")
        sys.exit(1)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    results: list[tuple[str, str, bool]] = []  # (quant, dst_path, success)

    for quant in tqdm(args.quant, desc="GGUF quant levels", unit="quant"):
        dst = build_dst_path(args.src, quant, args.outdir)
        logging.info(f"\n{'='*60}")
        logging.info(f"Converting {os.path.basename(args.src)} → {os.path.basename(dst)}")
        logging.info(f"Quant type : {quant}")
        logging.info(f"Output     : {dst}")
        logging.info(f"{'='*60}")

        try:
            out_path, model_arch = convert_file(
                args.src,
                dst_path=dst,
                interact=False,
                overwrite=args.overwrite,
                quant_type_name=quant,
            )
            logging.info(f"[OK] {quant} → {out_path}  (arch={model_arch.arch})")
            results.append((quant, out_path, True))
        except Exception as exc:
            logging.error(f"[FAILED] {quant}: {exc}")
            results.append((quant, dst, False))

    # Summary
    print(f"\n{'='*60}")
    print("Conversion summary:")
    for quant, path, ok in results:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {quant:6s}  {path}")
    print(f"{'='*60}")

    if not all(ok for _, _, ok in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
