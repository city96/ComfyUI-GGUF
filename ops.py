# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import contextlib
import functools
import logging
import operator
from typing import Callable, NamedTuple, Optional

import comfy.lora
import comfy.model_management as model_management
import comfy.ops
import torch

from . import dequant
from .dequant import DEFAULT_CONFIG, GGUFConfig, dequantize_tensor, is_quantized

try:
    import comfy.weight_adapter as wadapter
except (ImportError, ModuleNotFoundError):
    wadapter = None


def chained_hasattr(obj, chained_attr):
    probe = obj
    for attr in chained_attr.split("."):
        if hasattr(probe, attr):
            probe = getattr(probe, attr)
        else:
            return False
    return True


# A bakcward and forward compatible way to get `torch.compiler.disable`.
def get_torch_compiler_disable_decorator():
    def dummy_decorator(*args, **kwargs):
        def noop(x):
            return x

        return noop

    from packaging import version

    if not chained_hasattr(torch, "compiler.disable"):
        logging.info("ComfyUI-GGUF: Torch too old for torch.compile - bypassing")
        return dummy_decorator  # torch too old
    elif version.parse(torch.__version__) >= version.parse("2.8"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile")
        return dummy_decorator  # torch compile works
    if chained_hasattr(torch, "_dynamo.config.nontraceable_tensor_subclasses"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile (nightly)")
        return dummy_decorator  # torch compile works, nightly before 2.8 release
    else:
        logging.info(
            "ComfyUI-GGUF: Partial torch compile only, consider updating pytorch"
        )
        return torch.compiler.disable


torch_compiler_disable = get_torch_compiler_disable_decorator()


class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """

    def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches

    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            logging.warning(f"ignoring 'copy_' on tensor: {e}")

    def new_empty(self, size, *args, **kwargs):
        # Intel Arc fix, ref#50
        new_tensor = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(
            new_tensor,
            tensor_type=getattr(self, "tensor_type", None),
            tensor_shape=size,
            patches=getattr(self, "patches", []).copy(),
        )

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


class GGMLLayer(torch.nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """

    comfy_cast_weights = True
    largest_layer = False

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = (
            state_dict.get(f"{prefix}weight"),
            state_dict.get(f"{prefix}bias"),
        )
        # NOTE: using modified load for linear due to not initializing on creation, see GGMLOps todo
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(
            self, torch.nn.Linear
        ):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # Not strictly required, but fixes embedding shape mismatch. Threshold set in loader.py
        if isinstance(self, torch.nn.Embedding) and self.weight.shape[0] >= (64 * 1024):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k[prefix_len:] == "weight":
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                unexpected_keys.append(k)

        # For Linear layer with missing weight
        if self.weight is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix + "weight")

        # for vram estimation (TODO: less fragile logic?)
        if getattr(self.weight, "is_largest_weight", False):
            self.largest_layer = True

    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        # This is a fake state dict for vram estimation
        weight = torch.zeros_like(self.weight, device=torch.device("meta"))
        destination[prefix + "weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[prefix + "bias"] = bias

        # Take into account space required for dequantizing the largest tensor
        if self.largest_layer:
            dequant_dtype = self.gguf_config.dequant_dtype
            shape = getattr(self.weight, "tensor_shape", self.weight.shape)
            dtype = (
                dequant_dtype
                if dequant_dtype and dequant_dtype != "target"
                else torch.float16
            )
            temp = torch.empty(*shape, device=torch.device("meta"), dtype=dtype)
            destination[prefix + "temp.weight"] = temp

        return
        # This would return the dequantized state dict
        destination[prefix + "weight"] = self.get_weight(self.weight)
        if bias is not None:
            destination[prefix + "bias"] = self.get_weight(self.bias)

    def get_weight_patches(self, tensor, device):
        # consolidate and load patches to GPU in async
        patch_list = []
        key = None
        for patches, key in getattr(tensor, "patches", []):
            patch_list += move_patch_to_device(patches, device)

        return tuple(patch_list), key

    def get_weight(
        self, tensor, dtype, patches_tensor=None, apply_patches: bool = True
    ):
        if tensor is None:
            return

        device = tensor.device
        patch_list, key = (
            self.get_weight_patches(
                patches_tensor if patches_tensor is not None else tensor, device
            )
            if apply_patches
            else ((), None)
        )

        # dequantize tensor while patches load
        weight = dequantize_tensor(tensor, dtype, self.gguf_config)

        # prevent propagating custom tensor class
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)

        if not patch_list:
            return weight

        # apply patches
        patch_dtype = self.gguf_config.patch_dtype
        if patch_dtype is None:
            weight = comfy.lora.calculate_weight(patch_list, weight, key)
        else:
            # for testing, may degrade image quality
            if patch_dtype == "target":
                patch_dtype = dtype
            weight = comfy.lora.calculate_weight(patch_list, weight, key, patch_dtype)
        return weight

    @torch_compiler_disable()
    def cast_bias_weight(self, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        ostream = None
        try:
            qweight, qbias, ostream = comfy.ops.cast_bias_weight(
                self,
                input=None,
                device=device,
                dtype=self.weight.dtype,
                bias_dtype=None if self.bias is None else self.bias.dtype,
                offloadable=True,
            )
            if qbias is not None:
                bias = self.get_weight(qbias, dtype, patches_tensor=self.bias)
            weight = self.get_weight(qweight, dtype, patches_tensor=self.weight)
            return weight, bias
        finally:
            if ostream is not None:
                comfy.ops.uncast_bias_weight(self, qweight, qbias, ostream)

    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)
        else:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)

        # non-ggml forward might still propagate custom tensor class
        if isinstance(out, GGMLTensor):
            out = torch.Tensor(out)
        return out

    def forward_ggml_cast_weights(self, input):
        raise NotImplementedError


def compress_lora_pair(
    A_cat: torch.Tensor,
    B_cat: torch.Tensor,
    *,
    max_rank: int = 256,
    threshold: float = 1e-05,
):
    """
    A_cat: shape (in_features, R_total)
    B_cat: shape (R_total, out_features)
    """
    R_total = A_cat.shape[1]

    # 1. QR Decomposition of A
    # Q_A is (in_features, R_total), R_A is (R_total, R_total)
    Q_A, R_A = torch.linalg.qr(A_cat)

    # 2. QR Decomposition of B (transposed to work on columns)
    # Q_B_T is (out_features, R_total), R_B_T is (R_total, R_total)
    Q_B_T, R_B_T = torch.linalg.qr(B_cat.T)

    # Re-transpose to get L_B and Q_B
    L_B = R_B_T.T  # shape (R_total, R_total)
    Q_B = Q_B_T.T  # shape (R_total, out_features)

    # 3. Create the tiny core matrix!
    # This matrix is strictly (R_total x R_total). E.g., 640 x 640.
    core_matrix = R_A @ L_B

    # 4. Run SVD on the tiny core matrix
    U, S, Vh = torch.linalg.svd(core_matrix)

    # 5. Prune the overlapping/useless ranks
    # Keep ranks where the singular value is above the threshold, up to max_rank
    rank = min((S > threshold).sum().item(), max_rank)

    if rank == R_total:
        # No overlap found! Return the originals to save math.
        return A_cat, B_cat

    # Slice to the compressed rank
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vh_k = Vh[:rank, :]

    # 6. Reconstruct the new, pruned A and B matrices
    # A_new = Q_A @ U_k @ diag(S_k)
    A_new = Q_A @ (U_k * S_k[None, :])  # Broadcasting is faster than diag

    # B_new = Vh_k @ Q_B
    B_new = Vh_k @ Q_B

    return A_new, B_new


def hash_patch(patch) -> int:
    if isinstance(patch, (list, tuple)):
        return functools.reduce(operator.xor, (hash_patch(pi) for pi in patch))
    if isinstance(patch, wadapter.WeightAdapterBase):
        return hash_patch(patch.weights) ^ hash_patch(patch.loaded_keys)
    if isinstance(patch, torch.Tensor):
        return patch.hash_tensor().detach().cpu().item()
    if isinstance(patch, (set, frozenset)):
        return hash_patch(tuple(patch))
    if isinstance(patch, dict):
        return hash_patch(tuple(patch.items()))
    return hash(patch) if getattr(patch, "__hash__", None) is not None else id(patch)


class PatchCacheItem(NamedTuple):
    layer_id: int
    weight_shape: tuple[int, ...]
    patch_ids: tuple[int, ...]
    patch_hashes: tuple[int, ...]
    lora_A: torch.Tensor | None
    lora_B: torch.Tensor | None


# Key: id of layer module.
LORA_CACHE: dict[tuple[int, ...], PatchCacheItem] = {}


class GGMLOps(comfy.ops.manual_cast):
    """
    Dequantize weights on the fly before doing the compute
    """

    _MODULE_NAMES = ("Linear", "Conv2d", "Embedding", "LayerNorm", "GroupNorm")

    def __init__(self, *args, gguf_config: Optional[GGUFConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        linear_config = gguf_config or DEFAULT_CONFIG
        # Ignore patch_dtype and dequant_dtype for non-Linear layers.
        other_config = linear_config._replace(patch_dtype=None, dequant_dtype=None)
        self.gguf_config = linear_config
        for module_name in self._MODULE_NAMES:
            module = getattr(self.__class__, module_name)
            curr_config = linear_config if module_name == "Linear" else other_config
            setattr(
                self,
                module_name,
                type(module_name, (module,), {"gguf_config": curr_config}),
            )

    class Linear(GGMLLayer, comfy.ops.manual_cast.Linear):
        def __init__(
            self, in_features, out_features, bias=True, device=None, dtype=None
        ):
            torch.nn.Module.__init__(self)
            # TODO: better workaround for reserved memory spike on windows
            # Issue is with `torch.empty` still reserving the full memory for the layer
            # Windows doesn't over-commit memory so without this 24GB+ of pagefile is used
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def _get_lora(
            self,
            *,
            x: torch.Tensor,
            patches,
            weight_shape: tuple[int, ...],
            dequantize_weight: Callable[[torch.dtype], torch.Tensor],
            rank_threshold: float = 1e-05,
            max_rank: int = 256,
            decomp_iters: int = 2,
        ) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor | None]:
            flat_patches = []
            for p, pk in patches:
                flat_patches += p
            if not flat_patches:
                return None, None

            # print(f"\nFLAT PATCHES: {patches}")
            patch_hashes = None
            patch_ids = tuple(id(p) for p in flat_patches)
            patch_key = pk

            device, dtype = x.device, x.dtype
            cache_key = (id(self),)
            orig_cache_item = cache_item = LORA_CACHE.get(cache_key)
            if cache_item and (
                weight_shape != cache_item.weight_shape
                or (cache_item.lora_A is not None and cache_item.lora_A.dtype != dtype)
            ):
                cache_item = None
            need_hashes = not cache_item or cache_item.patch_ids != patch_ids
            if need_hashes:
                patch_hashes = tuple(hash_patch(p) for p in flat_patches)
            if cache_item and need_hashes:
                if cache_item and cache_item.patch_hashes == patch_hashes:
                    cache_item = cache_item._replace(patch_ids=patch_ids)
                    LORA_CACHE[cache_key] = cache_item
            if not cache_item and orig_cache_item:
                print(
                    f"\nLORA: INVALIDATING: key={patch_key}, id={cache_key}, weight={weight_shape}, expected weight={orig_cache_item.weight_shape}, patch_hashes {patch_hashes} != {orig_cache_item.patch_hashes}",
                )
                del LORA_CACHE[cache_key], orig_cache_item
            if (
                cache_item is not None
                and cache_item.lora_A is not None
                and cache_item.lora_B is not None
            ):
                lora_A, lora_B = cache_item.lora_A, cache_item.lora_B
                if lora_A.device != device:
                    lora_A = model_management.cast_to_device(
                        lora_A,
                        device,
                        dtype,
                        copy=True,
                    )
                if lora_B.device != device:
                    lora_B = model_management.cast_to_device(
                        lora_B,
                        device,
                        dtype,
                        copy=True,
                    )
                return (lora_A.to(dtype=dtype), lora_B.to(dtype=dtype)), None

            pure_lora = all(
                # No offset
                offset is None
                # No special function handler
                and fun is None
                # Just LoRAs
                and wadapter is not None
                and isinstance(lo, wadapter.LoRAAdapter)
                # With the expected number of weights
                and len(lo.weights) == 6
                # And no funny business like mid parts, reshaping or DoRA scales.
                and tuple(lo.weights[-3:]) == (None, None, None)
                for _strength, lo, _strength_model, offset, fun, *_rest in flat_patches
            )
            skip_decomp = cache_item is not None
            patch_dtype = self.gguf_config.patch_dtype
            eff_patch_dtype = (
                (x.dtype if patch_dtype == "target" else patch_dtype)
                if skip_decomp or pure_lora
                else torch.float32
            )
            patch_list = []
            for ppair in patches:
                patch_list += move_patch_to_device(
                    ppair[0], device=device, dtype=eff_patch_dtype
                )

            if skip_decomp or not pure_lora:
                dq_weight = dequantize_weight(eff_patch_dtype)
                # 1. Trick ComfyUI into giving us the dense delta
                patched_weight = comfy.lora.calculate_weight(
                    patch_list,
                    dq_weight.clone(),
                    patch_key,
                    eff_patch_dtype,
                )
            if skip_decomp:
                return None, patched_weight.to(dtype=x.dtype)
            if pure_lora:
                stack_a = []
                stack_b = []
                for strength, lo, *_rest in patch_list:
                    lb, la, alpha = lo.weights[:3]
                    la = la.to(dtype=torch.float32, copy=False)
                    lb = lb.to(dtype=torch.float32, copy=False)
                    rank = la.shape[0]
                    eff_strength = strength * getattr(lo, "multiplier", 1.0)
                    if alpha is not None:
                        eff_strength *= alpha / rank
                    if eff_strength == 0:
                        continue
                    if eff_strength != 1.0:
                        la = la * eff_strength
                    stack_a.append(la.T)
                    stack_b.append(lb.T)
                n_loras = len(stack_a)
                lora_A = torch.cat(stack_a, dim=1)
                lora_B = torch.cat(stack_b, dim=0)
                print(
                    f"\nSIMPLE PATH: Initial shapes: a={lora_A.shape}, b={lora_B.shape}"
                )
                del stack_a, stack_b, patch_list
                if n_loras > 1:
                    lora_A, lora_B = compress_lora_pair(
                        lora_A,
                        lora_B,
                        max_rank=max_rank,
                        threshold=rank_threshold,
                    )
                    print(
                        f"SIMPLE PATH: Compressed shapes: a={lora_A.shape}, b={lora_B.shape}"
                    )
                lora_A = lora_A.to(dtype=dtype).contiguous()
                lora_B = lora_B.to(dtype=dtype).contiguous()
                LORA_CACHE[cache_key] = PatchCacheItem(
                    layer_id=cache_key[0],
                    weight_shape=weight_shape,
                    patch_ids=patch_ids,
                    patch_hashes=patch_hashes,
                    lora_A=lora_A.to(device="cpu"),
                    lora_B=lora_B.to(device="cpu"),
                )
                return (lora_A, lora_B), None

            dense_delta = patched_weight - dq_weight
            del dq_weight

            # 2. FAST SVD: Only compute the top singular values!
            # Note: svd_lowrank returns V directly (shape: in_features, q) instead of V^T
            max_expected_rank = min(max_rank, *weight_shape)
            U, S, V = torch.svd_lowrank(
                dense_delta, q=max_expected_rank, niter=decomp_iters
            )

            rank = (S > rank_threshold).sum().item()

            if rank == max_expected_rank:
                print(
                    f"[{patch_key}] Warning: LoRA rank hit max limit ({max_expected_rank}). Some detail may be lost."
                )

            if rank == 0:
                print(f"LORA: patch len={len(patch_ids)}, rank=0. Skipping.")
                lora_A, lora_B = None, None
            else:
                # V is already (in_features, q), so we just slice it!
                lora_A = V[:, :rank].to(dtype).contiguous()

                # U is (out_features, q). Multiply by S and transpose to (rank, out_features)
                lora_B = U[:, :rank].mul_(S[:rank]).T.to(dtype).contiguous()
                print(
                    f"\nLORA: patch len={len(patch_list)}, input shape={x.shape}, weight shape={weight_shape}, rank={rank} ({max_expected_rank}), A shape={lora_A.shape}, B shape={lora_B.shape}"
                )

            LORA_CACHE[cache_key] = PatchCacheItem(
                layer_id=cache_key[0],
                weight_shape=weight_shape,
                patch_ids=patch_ids,
                patch_hashes=patch_hashes,
                lora_A=lora_A.to(device="cpu"),
                lora_B=lora_B.to(device="cpu"),
            )
            return None, patched_weight.to(dtype=x.dtype)

        def forward_lora_cache(self, input: torch.Tensor) -> torch.Tensor | None:
            if "loracache" not in self.gguf_config.optimize:
                return None
            dtype, device = input.dtype, input.device
            weight = getattr(self, "weight", None)
            patches = getattr(weight, "patches", ())
            if weight is None or dequant.is_torch_compatible(weight):
                return None
            qtype = getattr(weight, "tensor_type", None)
            qfun = self.gguf_config.dequantize_handlers.get(qtype)
            if not (hasattr(qfun, "block_size") and hasattr(qfun, "type_size")):
                return None
            oshape = tuple(getattr(weight, "tensor_shape", weight.shape))
            ostream = qweight = qbias = None
            try:
                qweight, qbias, ostream = comfy.ops.cast_bias_weight(
                    self,
                    input=None,
                    device=device,
                    dtype=self.weight.dtype,
                    bias_dtype=None if self.bias is None else self.bias.dtype,
                    offloadable=True,
                )

                def dequantize_weight(
                    dtype=torch.float32, qweight=qweight
                ) -> torch.Tensor:
                    out = qfun(
                        qweight,
                        dtype=dtype,
                        block_size=qfun.block_size,
                        type_size=qfun.type_size,
                    )
                    return out.reshape(oshape)

                if patches:
                    lora_result, patched_weight = self._get_lora(
                        x=input,
                        patches=patches,
                        weight_shape=oshape,
                        dequantize_weight=dequantize_weight,
                    )
                    lora_A, lora_B = (
                        lora_result if lora_result is not None else (None, None)
                    )
                else:
                    lora_A = lora_B = patched_weight = None
                if qbias is not None:
                    bias = self.get_weight(qbias, dtype, patches_tensor=self.bias)
                if patched_weight is None:
                    patched_weight = dequantize_weight(input.dtype)
            finally:
                if ostream is not None:
                    comfy.ops.uncast_bias_weight(self, qweight, qbias, ostream)
            del qweight, qbias, ostream

            already_patched = lora_A is None or lora_B is None
            if not already_patched:
                M = input.numel() // input.shape[-1]
                K, N = self.in_features, self.out_features
                use_activation = M <= ((N * K) / max(1, N + K))
                if not use_activation:
                    patched_weight.addmm_(lora_B.T, lora_A.T, alpha=1.0, beta=1.0)
                    already_patched = True
            result = torch.nn.functional.linear(input, patched_weight, bias)
            if already_patched or lora_A is None or lora_B is None:
                return result
            # Activation patching code path.
            input_2d = input.view(-1, K)
            result_2d = result.view(-1, N)

            result_2d.addmm_(
                # X @ A -> shape (M, rank)
                input_2d @ lora_A,
                lora_B,
                alpha=1.0,
                beta=1.0,
            )

            # 4. Reshape back to original dimensions
            return result_2d.view(*input.shape[:-1], N)

        def forward_ggml_cast_weights(self, input):
            comfy.ops.run_every_op()
            result = self.forward_lora_cache(input)
            if result is not None:
                return result
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(GGMLLayer, comfy.ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Embedding(GGMLLayer, comfy.ops.manual_cast.Embedding):
        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if (
                self.weight.dtype == torch.float16
                or self.weight.dtype == torch.bfloat16
            ):
                out_dtype = None
            weight, _bias = self.cast_bias_weight(
                self, device=input.device, dtype=out_dtype
            )
            return torch.nn.functional.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ).to(dtype=output_dtype)

    class LayerNorm(GGMLLayer, comfy.ops.manual_cast.LayerNorm):
        def forward_ggml_cast_weights(self, input):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, weight, bias, self.eps
            )

    class GroupNorm(GGMLLayer, comfy.ops.manual_cast.GroupNorm):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(
                input, self.num_groups, weight, bias, self.eps
            )


def move_patch_to_device(item, device, *, dtype=None):
    if device is None:
        return item
    if isinstance(item, torch.Tensor):
        return model_management.cast_to_device(item, device, dtype, copy=True)
    if isinstance(item, (tuple, list)):
        return item.__class__(
            move_patch_to_device(seqitem, device, dtype=dtype) for seqitem in item
        )
    if (
        wadapter is not None
        and isinstance(item, wadapter.WeightAdapterBase)
        and hasattr(item, "loaded_keys")
        and isinstance(getattr(item, "weights", None), (tuple, list))
    ):
        return item.__class__(
            item.loaded_keys,
            item.weights.__class__(
                wi
                if not isinstance(wi, torch.Tensor)
                else model_management.cast_to_device(wi, device, dtype, copy=True)
                for wi in item.weights
            ),
        )
    return item
