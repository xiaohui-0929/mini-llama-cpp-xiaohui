import struct
from pathlib import Path

OUT_DIR = Path("tests/data/gguf")


def u32(f, v):
    f.write(struct.pack("<I", v))


def u64(f, v):
    f.write(struct.pack("<Q", v))


def gguf_string(f, s):
    data = s.encode("utf-8")
    u64(f, len(data))
    f.write(data)


def pad_to(f, alignment=32):
    f.write(b"\x00" * ((-f.tell()) % alignment))


def build_minimal(path):
    with open(path, "wb") as f:
        f.write(b"GGUF")
        u32(f, 3)           # version
        u64(f, 1)           # tensor_count
        u64(f, 2)           # kv_count

        gguf_string(f, "general.architecture")
        u32(f, 8)           # value type = string
        gguf_string(f, "qwen2")

        gguf_string(f, "general.alignment")
        u32(f, 4)           # value type = uint32
        u32(f, 32)

        gguf_string(f, "token_embd.weight")
        u32(f, 1)           # n_dims
        u64(f, 4)           # dims[0]
        u32(f, 0)           # ggml type = F32
        u64(f, 0)           # offset

        pad_to(f, 32)       # 补齐到 160
        f.write(struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))


def build_nested_array(path):
    with open(path, "wb") as f:
        f.write(b"GGUF")
        u32(f, 3)
        u64(f, 0)           # tensor_count
        u64(f, 1)           # kv_count

        gguf_string(f, "test.nested")
        u32(f, 9)           # value type = array
        u32(f, 9)           # outer elem type = array
        u64(f, 2)           # outer count
        u32(f, 4)           # inner elem type = uint32
        u64(f, 2)           # inner count
        u32(f, 7)
        u32(f, 8)
        u32(f, 4)           # inner elem type = uint32
        u64(f, 2)           # inner count
        u32(f, 9)
        u32(f, 10)

        pad_to(f, 32)       # 补齐到 128


def build_duplicate_metadata(path):
    with open(path, "wb") as f:
        f.write(b"GGUF")
        u32(f, 3)
        u64(f, 0)           # tensor_count
        u64(f, 2)           # kv_count

        gguf_string(f, "general.name")
        u32(f, 8)
        gguf_string(f, "first")

        gguf_string(f, "general.name")
        u32(f, 8)
        gguf_string(f, "second")

        pad_to(f, 32)       # 补齐到 128


def write_patched(source, target_name, offset, data):
    content = bytearray(source.read_bytes())
    content[offset:offset + len(data)] = data
    (OUT_DIR / target_name).write_bytes(content)


def build_bad_magic(source):
    write_patched(source, "bad_magic.gguf", 0, b"FAIL")


def build_bad_version(source):
    write_patched(source, "bad_version.gguf", 4, struct.pack("<I", 2))


def build_truncated_string(source):
    # 第一个 key 的长度字段在偏移 24，改成远超剩余字节的值
    write_patched(source, "truncated_string.gguf", 24, struct.pack("<Q", 10 ** 9))


def build_bad_tensor_offset(source):
    # tensor offset 在偏移 143，改成不是 32 的倍数
    write_patched(source, "bad_tensor_offset.gguf", 143, struct.pack("<Q", 1))


def build_zero_dimension(source):
    # dims[0] 在偏移 131
    write_patched(source, "zero_dimension.gguf", 131, struct.pack("<Q", 0))


def build_tensor_out_of_bounds(source):
    # offset 在偏移 143，设成对齐但越界的值
    write_patched(source, "tensor_out_of_bounds.gguf", 143,
                  struct.pack("<Q", 1 << 20))


def build_truncated_file(source):
    (OUT_DIR / "truncated_file.gguf").write_bytes(source.read_bytes()[:12])


def build_bad_alignment(source):
    # general.alignment 的值在偏移 98，改成不是 8 的倍数
    write_patched(source, "bad_alignment.gguf", 98, struct.pack("<I", 12))


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    minimal = OUT_DIR / "minimal.gguf"

    build_minimal(minimal)
    build_nested_array(OUT_DIR / "nested_array.gguf")
    build_duplicate_metadata(OUT_DIR / "duplicate_metadata.gguf")

    build_bad_magic(minimal)
    build_bad_version(minimal)
    build_truncated_string(minimal)
    build_bad_tensor_offset(minimal)
    build_zero_dimension(minimal)
    build_tensor_out_of_bounds(minimal)
    build_truncated_file(minimal)
    build_bad_alignment(minimal)

    for path in sorted(OUT_DIR.glob("*.gguf")):
        print(f"{path.name:24s} {path.stat().st_size:5d} bytes")
