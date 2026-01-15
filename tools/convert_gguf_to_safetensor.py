import sys
import gguf  # Ensure this is the gguf module from llama.cpp
from gguf.constants import GGUFValueType
from safetensors.numpy import save_file
import numpy as np

def reconstruct_field_value(field):
    # Reconstruct the value from the field
    types = field.types
    if not types:
        return None
    gtype = types[0]
    if gtype == GGUFValueType.ARRAY:
        # Handle arrays
        raw_itype = field.parts[0][0]  # np.uint32
        array_length = field.parts[1][0]  # np.uint64
        # Now read the elements
        array_values = []
        idx = 2
        for _ in range(array_length):
            element = field.parts[idx][0]
            array_values.append(int(element))
            idx += 1
        return array_values
    elif gtype in [
        GGUFValueType.UINT8, GGUFValueType.INT8,
        GGUFValueType.UINT16, GGUFValueType.INT16,
        GGUFValueType.UINT32, GGUFValueType.INT32,
        GGUFValueType.UINT64, GGUFValueType.INT64,
        GGUFValueType.FLOAT32, GGUFValueType.FLOAT64,
        GGUFValueType.BOOL,
    ]:
        # Scalar types
        value = field.parts[0][0]
        return value
    elif gtype == GGUFValueType.STRING:
        # String type
        value = field.parts[1].tobytes().decode('utf-8')
        return value
    else:
        # Other types can be added as needed
        return None

def bfloat16_to_float32(bf16_array):
    """Converts an array of bfloat16 stored as uint16 to float32."""
    # Create an empty float32 array
    float32_array = np.empty(bf16_array.shape, dtype=np.float32)
    # View the float32 array as uint32
    uint32_view = float32_array.view(np.uint32)
    # Assign the high 16 bits from the bf16 data and zero out the low 16 bits
    uint32_view[:] = bf16_array.astype(np.uint32) << 16
    return float32_array

def main():
    import numpy as np
    print(f"NumPy version: {np.__version__}")

    if len(sys.argv) != 3:
        print("Usage: python convert_gguf_to_safetensor.py <input.gguf> <output.safetensors>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Reading GGUF file: {input_file}")
    reader = gguf.GGUFReader(input_file)

    tensors = {}
    print(f"Number of tensors: {len(reader.tensors)}")

    for tensor in reader.tensors:
        name = tensor.name.decode('utf-8') if isinstance(tensor.name, bytes) else tensor.name
        print(f"Reading tensor: {name}")
        data = tensor.data

        # Check for original shape metadata
        orig_shape_key = f"comfy.gguf.orig_shape.{name}"
        if orig_shape_key in reader.fields:
            field = reader.get_field(orig_shape_key)
            orig_shape = reconstruct_field_value(field)
            print(f"Reshaping tensor '{name}' to original shape: {orig_shape}")
            data = data.reshape(orig_shape)
        else:
            print(f"No original shape found for tensor '{name}', using current shape.")

        # Convert data types if necessary
        if data.dtype.name == 'bfloat16':
            print(f"Converting tensor '{name}' from bfloat16 to float32")
            data = bfloat16_to_float32(data)

        tensors[name] = data

    print(f"Saving to safetensors file: {output_file}")
    save_file(tensors, output_file)
    print("Conversion completed successfully.")

if __name__ == "__main__":
    main()
