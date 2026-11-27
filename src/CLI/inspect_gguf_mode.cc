#include "mini_llama/CLI/inspect_gguf_mode.h"

#include <iostream>

namespace mini_llama {

namespace {

std::string EscapeOneLine(const std::string& s, size_t max_len) {
    std::string out;
    out.reserve(std::min(s.size(), max_len) + 3);
    for (char c : s) {
        if (out.size() >= max_len) {
            break;
        }
        switch (c) {
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c;     break;
        }
    }
    if (s.size() > max_len) {
        out += "...";
    }
    return out;
}

void PrintMetadataValue(const GgufMetadataValue& v) {
    switch (v.value_type) {
        case GgufMetadataValueType::kgguf_uint8:
        case GgufMetadataValueType::kgguf_uint16:
        case GgufMetadataValueType::kgguf_uint32:
        case GgufMetadataValueType::kgguf_uint64:
            std::cout << v.u64;
            break;
        case GgufMetadataValueType::kgguf_int8:
        case GgufMetadataValueType::kgguf_int16:
        case GgufMetadataValueType::kgguf_int32:
        case GgufMetadataValueType::kgguf_int64:
            std::cout << v.i64;
            break;
        case GgufMetadataValueType::kgguf_float32:
        case GgufMetadataValueType::kgguf_float64:
            std::cout << v.f64;
            break;
        case GgufMetadataValueType::kgguf_bool:
            std::cout << (v.bool_ ? "true" : "false");
            break;
        case GgufMetadataValueType::kgguf_string:
            // string 只打印60个字符
            std::cout << '"' << EscapeOneLine(v.string, 60) << '"';
            break;
        case GgufMetadataValueType::kgguf_array: {
            // array类型 最多打印3个
            std::cout << "[array size=" << v.array.size() << "]";
            const size_t n = std::min<size_t>(v.array.size(), 3);
            for (size_t i = 0; i < n; ++i) {
                std::cout << (i == 0 ? " " : ", ");
                PrintMetadataValue(v.array[i]);
            }
            if (v.array.size() > n) {
                std::cout << ", ...";
            }
            break;
        }
        default:
            std::cout << "<unknown>";
            break;
    }
}

}  // namespace

void InspectGguf(const GgufReader& reader) {
    std::cout << "=== GGUF Info ===" << std::endl;
    std::cout << "  version:    " << reader.Header().version << std::endl;
    std::cout << "  n_tensors:  " << reader.Header().tensor_info_count << std::endl;
    std::cout << "  n_metadata: " << reader.Header().metadata_kv_count << std::endl;
    std::cout << "  data_offset: " << reader.DataOffset() << std::endl;
    std::cout << "  alignment:   " << reader.Alignment() << std::endl;

    if (!reader.Metadata().empty()) {
        std::cout << std::endl;
        std::cout << "=== Metadata ===" << std::endl;
        for (const auto& kv : reader.Metadata()) {
            std::cout << "  " << kv.key << " (" 
                      << GgufReader::GgufMetadataValueTypeName(static_cast<uint32_t>(kv.value.value_type))
                      << "): ";
            PrintMetadataValue(kv.value);
            std::cout << std::endl;
        }
    }

    if (!reader.TensorInfo().empty()) {
        std::cout << std::endl;
        std::cout << "=== Tensors ===" << std::endl;
        for (const auto& t : reader.TensorInfo()) {
            std::cout << "  " << t.name << std::endl;
            std::cout << "    shape:  [";
            for (size_t i = 0; i < t.dims.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << t.dims[i];
            }
            std::cout << "]" << std::endl;
            std::cout << "    dtype:  " << GgufReader::GgmlTypeName(static_cast<uint32_t>(t.ggml_type)) << " (" 
                        << static_cast<uint32_t>(t.ggml_type)
                        << ")" << std::endl;
            std::cout << "    offset: " << t.offset << std::endl;
            std::cout << "    byte_size: " << t.byte_size << std::endl;
        }
    }
}

int RunInspectGguf(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " inspect-gguf <path>\n";
        return 1;
    }
    GgufReader reader;
    std::string error;
    if (!reader.Open(argv[2], error)) {
        std::cerr << "Failed to load GGUF: " << error << "\n";
        return 1;
    }
    InspectGguf(reader);
    return 0;
}

}   // namespace mini_llama