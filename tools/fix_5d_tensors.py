import gguf
import torch
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("prec")
parser.add_argument("--root")
args = parser.parse_args()

arch = "hyvid" # TODO: Really should autodetect this
prec = args.prec #"Q4_K_S" # edit manually for each step
src = fr"{args.root}/raw/hunyuan-video-t2v-720p-{prec}.gguf"
dst = fr"{args.root}/hunyuan-video-t2v-720p-{prec}.gguf"

sd5d = torch.load(f"./fix_5d_tensors_{arch}.pt")
print("5D:", sd5d.keys())

reader = gguf.GGUFReader(src)
writer = gguf.GGUFWriter(path=None, arch=arch)

writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
writer.add_file_type(getattr(gguf.LlamaFileType, f"MOSTLY_{prec}")) # TODO: also autodetect

added = []
def add_extra_key(writer, key, data):
    old_dtype = data.dtype
    data_qtype = gguf.GGMLQuantizationType.F32
    n_dims = len(data.shape)
    data_shape = data.shape
    data = gguf.quants.quantize(data, data_qtype)
    tqdm.write(f"Adding key {key} ({data_shape})")
    writer.add_tensor(key, data, raw_dtype=data_qtype)
    global added
    added.append(key)

# main loop to add missing
for tensor in tqdm(reader.tensors):
    writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    key5d = tensor.name.replace(".bias", ".weight")
    if key5d in sd5d.keys():
        add_extra_key(writer, key5d, sd5d[key5d])

# brute force for any missed
for key, data in sd5d.items():
    if key not in added:
        add_extra_key(writer, key, data)

writer.write_header_to_file(path=dst)
writer.write_kv_data_to_file()
writer.write_tensors_to_file(progress=True)
writer.close()
