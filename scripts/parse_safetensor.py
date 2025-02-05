#
# Parse a `.safetensors` file header into a more minimal and easier to process format
#

import json
import struct


with open("../Qwen2.5-0.5B/model.safetensors", "rb") as fp:

    metadata = fp.read(8)
    header_size = struct.unpack('<Q', metadata)[0]


    fp.seek(8)
    model_data = json.loads(fp.read(header_size))

    print(f"Safetensor data starts at byte: {8 + header_size}")


    with open("../parsed_tensors.txt", "w") as fp2:
        for key in model_data.keys():
            if key != "__metadata__":
                fp2.write(f"{key} {model_data[key]['shape']} {model_data[key]['data_offsets']}\n")







    

