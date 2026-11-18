"""D10 Part B: read a GGUF header and tensor table only (no tensor data).

usage: python3 gguf_table.py FILE OUT.json
Reads sequentially from the start of the file up to the end of the tensor
info table and stops; prints how many bytes it read.
"""
import json
import struct
import sys

GT = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
      10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
      17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
      23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
      30: "BF16", 39: "MXFP4"}


class R:
    def __init__(self, f):
        self.f, self.n = f, 0

    def read(self, k):
        b = self.f.read(k)
        if len(b) != k:
            raise EOFError
        self.n += k
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def s(self):
        return self.read(self.u64()).decode("utf-8", "replace")


SCAL = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
        6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


def val(r, t):
    if t in SCAL:
        fmt, n = SCAL[t]
        return struct.unpack(fmt, r.read(n))[0]
    if t == 8:
        return r.s()
    if t == 9:
        et, n = r.u32(), r.u64()
        v = [val(r, et) for _ in range(n)]
        return v if n <= 64 else f"<array {n} of type {et}>"
    raise ValueError(t)


def main(path, out):
    with open(path, "rb") as f:
        r = R(f)
        assert r.read(4) == b"GGUF"
        ver, nt, nkv = r.u32(), r.u64(), r.u64()
        kv = {}
        for _ in range(nkv):
            k = r.s()
            kv[k] = val(r, r.u32())
        align = kv.get("general.alignment", 32)
        ts = []
        for _ in range(nt):
            name = r.s()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            t = r.u32()
            off = r.u64()
            ts.append(dict(name=name, dims=dims, type=GT.get(t, str(t)), off=off))
        hdr = r.n
        data0 = (hdr + align - 1) // align * align
        f.seek(0, 2)
        size = f.tell()
    ts.sort(key=lambda x: x["off"])
    for i, x in enumerate(ts):
        end = ts[i + 1]["off"] if i + 1 < len(ts) else size - data0
        x["bytes"] = end - x["off"]
        x["abs_off"] = data0 + x["off"]
    json.dump(dict(version=ver, kv={k: v for k, v in kv.items()}, tensors=ts, header_bytes=hdr,
                   file_bytes=size), open(out, "w"), indent=0, default=str)
    print(f"{path}: {nt} tensors, {nkv} kv, header+table {hdr} bytes read, file {size} bytes")


main(sys.argv[1], sys.argv[2])
