#!/usr/bin/env python3
"""一次性 GGUF 探针：顺序读出 header / metadata / tensor_info，打印关键值。
用法: python3 scripts/inspect_gguf_probe.py <file.gguf>
"""
import os
import struct
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "models/qwen2.5-0.5b/Qwen2.5-0.5B-Instruct-f16.gguf"
with open(path, "rb") as f:

    # metadata 的 value 类型（GGUF 规范）
    TYPE_NAME = {0:"u8",1:"i8",2:"u16",3:"i16",4:"u32",5:"i32",6:"f32",
                 7:"bool",8:"string",9:"array",10:"u64",11:"i64",12:"f64"}
    TYPE_SIZE = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    FMT = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",10:"<Q",11:"<q",12:"<d"}

    # tensor 的 ggml 类型（和上面不是一套编号）
    GGML = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
            10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",
            16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",
            21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",
            27:"I64",28:"F64",29:"IQ1_M",30:"BF16"}

    def u32():
        return struct.unpack("<I", f.read(4))[0]

    def u64():
        return struct.unpack("<Q", f.read(8))[0]

    def string():
        n = u64()
        return f.read(n).decode("utf-8", "replace")

    def value(t):
        if t == 8:                      # string
            return string()
        if t == 9:                      # array: elem_type + count + 元素
            et = u32()
            n = u64()
            preview = []
            for i in range(n):
                v = value(et)
                if i < 3:
                    preview.append(v)
            return "[{} x {}] {}{}".format(TYPE_NAME.get(et, et), n, preview,
                               " ..." if n > 3 else "")
        if t not in TYPE_SIZE:
            raise ValueError(f"unknown value type: {t}")
        raw = f.read(TYPE_SIZE[t])
        if len(raw) != TYPE_SIZE[t]:
            raise ValueError("unexpected EOF")
        if t == 7:                      # bool
            return bool(raw[0])
        return struct.unpack(FMT[t], raw)[0]


    magic = f.read(4)
    version = u32()
    n_tensors = u64()
    n_kv = u64()
    print("magic   =", magic)
    print("version =", version)
    print("tensors =", n_tensors)
    print("kv      =", n_kv)

    meta = {}
    for i in range(n_kv):
        #print("current offset: %d" % (f.tell()))
        key = string()
        t = u32()
        v = value(t)
        meta[key] = v
        print(f"kv[{i}] {key:<32} type={TYPE_NAME.get(t, t):<6} value={v}")

    print("\n--- 关心的键 ---")
    for k in ["general.architecture", "general.name", "general.alignment",
              "qwen2.block_count", "qwen2.embedding_length", "qwen2.feed_forward_length",
              "qwen2.attention.head_count", "qwen2.attention.head_count_kv",
              "qwen2.attention.layer_norm_rms_epsilon", "qwen2.rope.freq_base",
              "qwen2.context_length", "tokenizer.ggml.model",
              "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"]:
        if k in meta:
            print(f"{k:<42} = {meta[k]}")

    align = int(meta.get("general.alignment", 32))
    print("\nalignment        =", align)
    print("tensor_info 起点 =", f.tell())

    offset_tensor_info_begin = f.tell()

    infos = []
    for i in range(n_tensors):
        name = string()
        nd = u32()
        dims = [u64() for _ in range(nd)]
        t = u32()
        off = u64()
        infos.append((name, dims, t, off))
        print(f"tensor[{i}] {name:<28} dims={dims} rev={dims[::-1]} type={GGML.get(t, t)} offset={off}")

    pos = f.tell()

    print("tensor_info 字节大小 =", pos - offset_tensor_info_begin)

    data_offset = (pos + align - 1) // align * align
    print("\ntensor_info 结束 =", pos)
    print(f"data_offset      = {data_offset} (padding {data_offset - pos})")
    print("file size        =", os.path.getsize(path))

    if infos:
        name, dims, t, off = infos[0]
        f.seek(data_offset + off)
        print(f"first tensor {name} @ {data_offset + off}, 前 16 字节: {f.read(16).hex(' ')}")