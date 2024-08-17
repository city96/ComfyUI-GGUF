# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import gguf
import torch
import types
import logging

import comfy.lora
from .ops import GGMLTensor

def load_lora_gguf(model, lora, strength):
    key_map = comfy.lora.model_lora_keys_unet(model.model)
    loaded = comfy.lora.load_lora(lora, key_map)

    new = model.clone()
    k = new.add_patches(loaded, strength)

    if not getattr(new, "custom_calc_weight", False):
        new.calculate_weight_orig = new.calculate_weight
        new.calculate_weight = types.MethodType(calculate_weight, new)
        new.custom_calc_weight = True

    for key in [x for x in loaded if x not in k]:
        logging.warning("NOT LOADED {}".format(key))

    return (new)

def calculate_weight(self, patches, weight, key):
    if isinstance(weight, GGMLTensor):
        qtype = weight.tensor_type
        # TODO: don't even store these in a custom format
        if qtype in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]:
            return self.calculate_weight_orig(patches, weight, key)
        else:
            weight.patches.append((self.calculate_weight_orig, patches, key))
            return weight
    else:
        return self.calculate_weight_orig(patches, weight, key)
