"""Copy the last 16 Qwen3.5 trunk blocks with their original quantized bytes."""

import re, sys
from gguf import GGUFReader, GGUFWriter, GGUFValueType

r = GGUFReader(sys.argv[1])
assert r.fields["general.architecture"].contents() == "qwen35"
assert [
    r.fields["qwen35." + k].contents()
    for k in ("block_count", "nextn_predict_layers", "full_attention_interval")
] == [33, 1, 4]
w = GGUFWriter(sys.argv[2], "qwen35")
for f in r.fields.values():
    if f.name.startswith("GGUF.") or f.name == "general.architecture":
        continue
    value = {"qwen35.block_count": 16, "qwen35.nextn_predict_layers": 0}.get(
        f.name, f.contents()
    )
    w.add_key_value(
        f.name,
        value,
        f.types[0],
        sub_type=f.types[-1] if f.types[0] == GGUFValueType.ARRAY else None,
    )
kept = []
for t in r.tensors:
    m = re.match(r"blk\.(\d+)\.(.*)", t.name)
    name = t.name
    if m:
        i = int(m[1])
        if not 16 <= i < 32:
            continue
        name = f"blk.{i - 16}.{m[2]}"
    w.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    kept.append(t)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for t in kept:
    w.write_tensor_data(t.data, tensor_endianess=r.endianess)
w.close()
