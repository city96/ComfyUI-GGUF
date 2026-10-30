# Helpers deciding when GGUF LoRA/adapter patches must be rebuilt through
# patch_weight_to_device instead of ComfyUI's default low-VRAM path.
#
# ComfyUI's default path wraps changed patches in LowVramPatch adapters bound
# to module weight_function lists, which the GGML dequantize-on-forward path
# ignores, silently dropping the adapter. Forcing patch_weight_to_device
# avoids that, but forcing unconditionally makes every load_models_gpu call
# (e.g. each text encode) pay a full unload+reload cycle (issue #26).
# Forcing is therefore limited to loads where reinstallation is required.
# This module is intentionally free of ComfyUI imports so it stays unit
# testable outside a ComfyUI environment.
from .dequant import is_quantized


def _resolve_module(model, dotted_key):
    target = model
    for part in dotted_key.split("."):
        target = getattr(target, part)
    return target


def patch_targets_quantized(patches, model):
    """Whether any active patch key touches a GGUF quantized weight path."""
    if not patches or model is None:
        # Uncertain (or nothing to do): callers treat "no patches" as False
        # via requires_patch_reinstall; a missing model preserves the forced
        # behavior below.
        return model is None and bool(patches)
    for key in patches:
        module_key, _, param = key.rpartition(".")
        if param not in ("weight", "bias") or not module_key:
            continue
        try:
            module = _resolve_module(model, module_key)
        except AttributeError:
            continue
        if is_quantized(getattr(module, "weight", None)):
            return True
        if is_quantized(getattr(module, "bias", None)):
            return True
        if bool(getattr(module, "_quantized", False)):
            return True
    return False


def requires_patch_reinstall(patcher):
    """Whether GGUF patch state must be rebuilt on the next load.

    True when quantized-touching patches exist and are not currently
    installed (first load, changed patch set, or a prior full unpatch
    cleared them). False preserves ComfyUI's residency fast path.
    """
    patches = getattr(patcher, "patches", None) or {}
    if not patches:
        return False
    model = getattr(patcher, "model", None)
    if model is None:
        return True
    if not patch_targets_quantized(patches, model):
        return False
    if getattr(model, "current_weight_patches_uuid", None) != getattr(
        patcher, "patches_uuid", None
    ):
        return True
    return not getattr(patcher, "_gguf_patches_installed", False)
