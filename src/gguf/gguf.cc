#include "mini_llama/gguf/gguf.h"

#include <cstring>
#include <stdexcept>
#include <limits>
#include <utility>

namespace mini_llama {

namespace {

constexpr int kMaxArrayDepth  = 16;

/**
 * 获取剩余字节数量
 */
uint64_t Remaining(std::ifstream& f, uint64_t file_size) {
    auto pos = f.tellg();
    if (pos < 0) {
        throw std::runtime_error("Remaining: tellg failed");
    }
    uint64_t cur = static_cast<uint64_t>(pos);
    if (cur > file_size) {
        throw std::runtime_error("Remaining: position beyond EOF");
    }
    return file_size - cur;
}

/**
 * 确保读取完整的长度
 */
void ReadExact(std::ifstream&f, void* dst, size_t n) {
    if (n > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("ReadExact: n oversize");
    }
    auto before = f.tellg();
    if (before < 0) {
        throw std::runtime_error("ReadExact: tellg failed");
    }
    f.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    if (f.gcount() != static_cast<std::streamsize>(n)) {
        throw std::runtime_error("ReadExact: unexpected EOF at offset " +
                                std::to_string(static_cast<long long>(before)));
    }
}

/**
 * 读取指定类型的数据   小端序
 */
template<typename T>
void ReadValueByType(std::ifstream& f, T& out) {
    char dst[sizeof(T)];
    ReadExact(f, dst, sizeof(T));
    std::memcpy(&out, dst, sizeof(T));
}

void ReadU8(std::ifstream& f, uint8_t& out) {
    ReadValueByType(f, out);
}

void ReadI8(std::ifstream& f, int8_t& out) {
    ReadValueByType(f, out);
}

void ReadU16(std::ifstream& f, uint16_t& out) {
    ReadValueByType(f, out);
}

void ReadI16(std::ifstream& f, int16_t& out) {
    ReadValueByType(f, out);
}

void ReadU32(std::ifstream& f, uint32_t& out) {
    ReadValueByType(f, out);
}

void ReadI32(std::ifstream& f, int32_t& out) {
    ReadValueByType(f, out);
}

void ReadF32(std::ifstream& f, float& out) {
    ReadValueByType(f, out);
}

void ReadU64(std::ifstream& f, uint64_t& out) {
    ReadValueByType(f, out);
}

void ReadI64(std::ifstream& f, int64_t& out) {
    ReadValueByType(f, out);
}

void ReadF64(std::ifstream& f, double& out) {
    ReadValueByType(f, out);
}

/**
 * 读取string类型的数据
 */
std::string ReadString(std::ifstream& f, uint64_t remaining) {
    if (remaining < sizeof(uint64_t)) {
        throw std::runtime_error("ReadString: not enough bytes for string length");
    }
    // 通过前8个字节获取字符串长度
    uint64_t len = 0;
    ReadU64(f, len);
    if (len > (remaining - sizeof(uint64_t))) {
        throw std::runtime_error("ReadString: string len longer than remaining");
    }
    std::string result(static_cast<size_t>(len), '\0');
    if (!result.empty()) {
        ReadExact(f, result.data(), result.size());
    }
    return result;
}

/**
 * 读取Header
 */
void ReadHeader(std::ifstream& f, GgufHeader& header) {
    ReadU32(f, header.magic);
    ReadU32(f, header.version);
    ReadU64(f, header.tensor_info_count);
    ReadU64(f, header.metadata_kv_count);
    if (header.magic != 0x46554747u) {
        throw std::runtime_error("ReadHeader: not a gguf file, bad magic: " + std::to_string(header.magic));
    }
    if (header.version != 3) {
        throw std::runtime_error("ReadHeader: gguf version error");
    }
    if (header.tensor_info_count > 1000000 ||
        header.metadata_kv_count > 1000000) {
        throw std::runtime_error("ReadHeader: entry count too large");
    }
}

/**
 * 读取metadata_value
 */
GgufMetadataValue ReadMetadataValue(
    std::ifstream& f, const GgufMetadataValueType& type, uint64_t file_byte_size, int depth) {
    // depth放深度递归
    if (depth > kMaxArrayDepth) {
        throw std::runtime_error("ReadMetadataValue: array nesting too deep: " +
                                 std::to_string(depth));
    }

    GgufMetadataValue result;
    result.value_type = type;
    if (type == GgufMetadataValueType::kgguf_uint8) {
        uint8_t value;
        ReadU8(f, value);
        result.u64 = static_cast<uint64_t>(value);
    } else if (type == GgufMetadataValueType::kgguf_uint16) {
        uint16_t value;
        ReadU16(f, value);
        result.u64 = static_cast<uint64_t>(value);
    } else if (type == GgufMetadataValueType::kgguf_uint32) {
        uint32_t value;
        ReadU32(f, value);
        result.u64 = static_cast<uint64_t>(value);
    } else if (type == GgufMetadataValueType::kgguf_uint64) {
        uint64_t value;
        ReadU64(f, value);
        result.u64 = static_cast<uint64_t>(value);
    } else if (type == GgufMetadataValueType::kgguf_int8) {
        int8_t value;
        ReadI8(f, value);
        result.i64 = static_cast<int64_t>(value);
    } else if (type == GgufMetadataValueType::kgguf_int16) {
        int16_t value;
        ReadI16(f, value);
        result.i64 = static_cast<int64_t>(value);
    } else if (type == GgufMetadataValueType::kgguf_int32) {
        int32_t value;
        ReadI32(f, value);
        result.i64 = static_cast<int64_t>(value);
    } else if (type == GgufMetadataValueType::kgguf_int64) {
        int64_t value;
        ReadI64(f, value);
        result.i64 = static_cast<int64_t>(value);
    } else if (type == GgufMetadataValueType::kgguf_float32) {
        float value;
        ReadF32(f, value);
        result.f64 = static_cast<double>(value);
    } else if (type == GgufMetadataValueType::kgguf_float64) {
        double value;
        ReadF64(f, value);
        result.f64 = value;
    } else if (type == GgufMetadataValueType::kgguf_bool) {
        uint8_t value;
        ReadU8(f, value);
        if (value == 0) {
            result.bool_ = false;
        } else if (value == 1) {
            result.bool_ = true;
        } else {
            throw std::runtime_error("ReadMetadataValue: read bool type error");
        }
    } else if (type == GgufMetadataValueType::kgguf_string) {
        uint64_t remaining = Remaining(f, file_byte_size);
        result.string = ReadString(f, remaining);
    } else if (type == GgufMetadataValueType::kgguf_array) {
        // array时 结构为: 元素类型uint32 元素个数uint64 元素数组
        uint32_t array_value_type = 0;
        ReadU32(f, array_value_type);
        // type检查
        if (array_value_type > 12) {
            throw std::runtime_error("ReadMetadataValue: bad array elem type");
        }
        uint64_t array_value_count = 0;
        ReadU64(f, array_value_count);
        uint64_t remaining = Remaining(f, file_byte_size);
        uint64_t min_elem = 0;
        switch (array_value_type) {
            case 0: case 1: case 7:              min_elem = 1;  break;  // u8/i8/bool
            case 2: case 3:                      min_elem = 2;  break;  // u16/i16
            case 4: case 5: case 6:              min_elem = 4;  break;  // u32/i32/f32
            case 10: case 11: case 12:           min_elem = 8;  break;  // u64/i64/f64
            case 8:                              min_elem = 8;  break;  // string: 长度前缀
            case 9:                              min_elem = 12; break;  // array: elem_type + count
            default: break;
        }
        // count数量检查
        constexpr uint64_t kMaxArrayCount = 1000000;
        if (array_value_count > kMaxArrayCount) {
            throw std::runtime_error("ReadMetadata: array count too large");
        }
        if (min_elem == 0 || array_value_count > remaining / min_elem) {
            throw std::runtime_error("ReadMetadata: array count exceeds remaining bytes");
        }
        // 递归处理
        result.array.reserve(array_value_count);
        for (uint64_t i = 0; i < array_value_count; ++i) {
            result.array.push_back(ReadMetadataValue(f, 
                static_cast<GgufMetadataValueType>(array_value_type), file_byte_size, depth + 1));
        }
    } else {
        throw std::runtime_error("ReadMetadata: unknow value type");
    }
    return result;
}

/**
 * 读取metadata_kv
 */
void ReadMetadata_kv(
    std::ifstream& f, 
    const uint64_t file_byte_size,
    const GgufHeader& header, 
    std::vector<GgufMetadataKV>& metadata_kv,
    std::unordered_map<std::string, size_t>& name_to_metadata_kv) {
    
    // 共有 header_.metadata_kv_count 个 metadata_kv
    metadata_kv.reserve(header.metadata_kv_count);
    for (uint64_t i = 0; i < header.metadata_kv_count; ++i) {
        GgufMetadataKV result;

        // 获取key
        uint64_t remaining = Remaining(f, file_byte_size);
        result.key = ReadString(f, remaining);

        if (name_to_metadata_kv.find(result.key) != name_to_metadata_kv.end()) {
            throw std::runtime_error("ReadMetadata: duplicate key: " + result.key);
        }
        // 获取value_type
        uint32_t value_type = 0;
        ReadU32(f, value_type);
        if (value_type > 12) {
            throw std::runtime_error("ReadMetadata: read value type error");
        }
        // 递归获取value
        result.value = ReadMetadataValue(f, 
            static_cast<GgufMetadataValueType>(value_type), file_byte_size, 0);

        name_to_metadata_kv[result.key] = metadata_kv.size();
        metadata_kv.push_back(std::move(result));
    }
}

/**
 * 获取alignment
 */
void GetAlignment(
    const std::vector<GgufMetadataKV>& metadata_kv,
    const std::unordered_map<std::string, size_t>& name_to_metadata_kv,
    uint32_t& alignment) {
    auto it_find = name_to_metadata_kv.find("general.alignment");
    if (it_find != name_to_metadata_kv.end()) {
        // 接引用 防止大数据拷贝
        const GgufMetadataValue& value = metadata_kv[it_find->second].value;
        if (value.value_type != GgufMetadataValueType::kgguf_uint32) {
            throw std::runtime_error("find alignment: alignment type error");
        }
        if (value.u64 % 8 != 0 || value.u64 == 0 || value.u64 > UINT32_MAX) {
            throw std::runtime_error("find alignment: alignment value error");
        }
        
        alignment = static_cast<uint32_t>(value.u64);
    } else {
        alignment = 32;    // kv内不存在alignment的记录 默认记为32
    }
}

void ReadTensorInfo(
    std::ifstream& f, 
    const uint64_t file_byte_size,
    const GgufHeader& header, 
    const uint32_t alignment,
    std::vector<GgufTensorInfo>& tensor_info,
    std::unordered_map<std::string, size_t>& name_to_tensor_info) {

    tensor_info.reserve(header.tensor_info_count);
    for (uint64_t i = 0; i < header.tensor_info_count; ++i) {
        GgufTensorInfo result;
        // 获取tensor name
        uint64_t remaining = Remaining(f, file_byte_size);
        std::string tensor_name = ReadString(f, remaining);
        if (name_to_tensor_info.find(tensor_name) != name_to_tensor_info.end()) {
            throw std::runtime_error("ReadTensorInfo: duplicated tensor name");
        }
        result.name = std::move(tensor_name);
        // 获取tensor 维度个数
        uint32_t dim_count = 0;
        ReadU32(f, dim_count);
        if (dim_count == 0 || dim_count > 8) {
            throw std::runtime_error("ReadTensorInfo: bad dim count");
        }
        // 获取具体维度值
        uint64_t element_number = 1;
        std::vector<uint64_t> dims;
        dims.reserve(dim_count);
        for (uint32_t n = 0; n < dim_count; ++n) {
            uint64_t dim = 0;
            ReadU64(f, dim);
            if (dim == 0) {
                throw std::runtime_error("ReadTensorInfo: zero dimension");
            }
            if (element_number > std::numeric_limits<uint64_t>::max() / dim) {
                throw std::runtime_error("ReadTensorInfo: element number overflow");
            }
            element_number *= dim;
            dims.push_back(dim);
        }
        result.dims = std::move(dims);
        // 获取ggml类型
        uint32_t ggml_type = 0;
        ReadU32(f, ggml_type);
        if (ggml_type != 0 && ggml_type != 1 && ggml_type != 8 && ggml_type != 12) {
            throw std::runtime_error("ReadTensorInfo: current tensor ggml type unsupported");
        }
        result.ggml_type = static_cast<GgufGgmlType>(ggml_type);
        // 获取offset
        uint64_t offset = 0;
        ReadU64(f, offset);
        if (offset % alignment != 0) {
            throw std::runtime_error("ReadTensorInfo: tensor offset not align");
        }
        result.offset = offset;
        // 计算占用的大小
        uint64_t elem_bytes = 0;
        // 暂不考虑Q8_0/Q4_K
        if (result.ggml_type == GgufGgmlType::kggml_F32) {
            elem_bytes = sizeof(float);
        } else if (result.ggml_type == GgufGgmlType::kggml_F16) {
            elem_bytes = sizeof(uint16_t);
        }
        if (elem_bytes != 0) {
            if (element_number > std::numeric_limits<uint64_t>::max() / elem_bytes) {
                throw std::runtime_error("ReadTensorInfo: byte_size overflow");
            }
            result.byte_size = element_number * elem_bytes;
        }

        name_to_tensor_info[result.name] = tensor_info.size();
        tensor_info.push_back(std::move(result));
    }
    
}
    
}   // namespace

const char* GgufReader::GgmlTypeName(uint32_t type) {
    if (type == 0) {
        return "F32";
    } else if (type == 1) {
        return "F16";
    }
    return "UNSUPPORTED";
}

const char* GgufReader::GgufMetadataValueTypeName(uint32_t type) {
    if (type == 0) {
        return "uint8";
    } else if (type == 1) {
        return "int8";
    } else if (type == 2) {
        return "uint16";
    } else if (type == 3) {
        return "int16";
    } else if (type == 4) {
        return "uint32";
    } else if (type == 5) {
        return "int32";
    } else if (type == 6) {
        return "float32";
    } else if (type == 7) {
        return "bool";
    } else if (type == 8) {
        return "string";
    } else if (type == 9) {
        return "array";
    } else if (type == 10) {
        return "uint64";
    } else if (type == 11) {
        return "int64";
    } else if (type == 12) {
        return "float64";
    }
    return "";
}

bool GgufReader::Open(const std::string& path, std::string& error) {
    // 开启文件前先清理
    Clear();

    try {
        file_.open(path, std::ios_base::binary);
        if (!file_.is_open()) {
            throw std::runtime_error("Open: gguf file open failed");
        }
        
        // 获取文件大小
        file_.seekg(0, std::ios_base::end);
        auto end = file_.tellg();
        if (end < 0) {
            throw std::runtime_error("Open: failed to get file size");
        }
        file_byte_size_ = static_cast<uint64_t>(end);
        file_.seekg(0, std::ios_base::beg);

        // 读取header
        ReadHeader(file_, header_);

        // 读取metadata
        ReadMetadata_kv(file_, file_byte_size_, header_, metadata_kv_, name_to_metadata_kv_);

        // 找到alignment
        GetAlignment(metadata_kv_, name_to_metadata_kv_, alignment_);

        // 获取tensorInfo
        ReadTensorInfo(file_, file_byte_size_, header_, alignment_, tensor_info_, name_to_tensor_info_);

        // 记录data_offset
        auto tell = file_.tellg();
        if (tell < 0) {
            throw std::runtime_error("Open: tellg failed after tensor info");
        }
        uint64_t pos = static_cast<uint64_t>(tell);
        uint64_t rem = pos % alignment_;
        if (rem != 0) {
            uint64_t pad = alignment_ - rem;
            if (pos > std::numeric_limits<uint64_t>::max() - pad) {
                throw std::runtime_error("Open: data_offset overflow");
            }
            pos += pad;
        }
        data_offset_ = pos;
        if (data_offset_ > file_byte_size_) {
            throw std::runtime_error("Open: data_offset override");
        }

        for (const auto& t : tensor_info_) {
            if (t.offset > file_byte_size_ - data_offset_) {
                throw std::runtime_error("Open: tensor offset out of range");
            }
            uint64_t abs = data_offset_ + t.offset;
            if (t.byte_size > file_byte_size_ - abs) {
                throw std::runtime_error("Open: tensor data out of range");
            }
        }
        
        loaded_ = true;
        error.clear();
        
        return true;

    } catch (const std::exception& e) {
        Clear();
        error = e.what();
        return false;
    }
}

void GgufReader::Clear() {
    loaded_ = false;
    file_byte_size_ = 0;
    data_offset_ = 0;
    alignment_ = 0;
    if (file_.is_open()) {
        file_.close();
    }
    header_ = GgufHeader{};
    metadata_kv_.clear();
    tensor_info_.clear();
    name_to_metadata_kv_.clear();
    name_to_tensor_info_.clear();
}

const GgufMetadataKV* GgufReader::GetMetadataByKey(const std::string& key) const {
    auto it_find = name_to_metadata_kv_.find(key);
    if (it_find == name_to_metadata_kv_.end()) {
        return nullptr;
    }
    return &metadata_kv_[it_find->second];
}

const GgufTensorInfo* GgufReader::GetTensorInfo(const std::string& tensor_name) const {
    auto it_find = name_to_tensor_info_.find(tensor_name);
    if (it_find == name_to_tensor_info_.end()) {
        return nullptr;
    }
    return &tensor_info_[it_find->second];
}

bool GgufReader::ReadTensorData(const std::string& tensor_name, std::vector<uint8_t>& data, std::string& error) {

    if (loaded_ == false) {
        error = "ReadTensorData: GgufReader load failed, please load right gguf file first";
        return false;
    }

    auto it_find = name_to_tensor_info_.find(tensor_name);
    if (it_find == name_to_tensor_info_.end()) {
        error = "ReadTensorData: fail to find tensor by name";
        return false;
    }

    // 找到对应的tensor_info
    auto& tensorInfo = tensor_info_[it_find->second];
    if (tensorInfo.byte_size == 0) {
        error = "ReadTensorData: tensorInfo byte size is zero, unsupported now";
        return false;
    }
    // 检查是否越界
    if (data_offset_ > file_byte_size_ - tensorInfo.offset) {
        error = "ReadTensorData: tensorInfo offset overflow";
        return false;
    }
    uint64_t offset = data_offset_ + tensorInfo.offset;
    if (offset > file_byte_size_ - tensorInfo.byte_size) {
        error = "ReadTensorData: tensorInfo byte size overflow";
        return false;
    }
    // 判断file_是否开启
    if (!file_.is_open()) {
        error = "ReadTensorData: gguf file open error";
        return false;
    }
    try {
        file_.clear();  // 清掉可能残留的 eof/fail
        // 偏移文件指针 读取指定长度
        if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            error = "ReadTensorData: offset too large for streamoff";
            return false;
        }

        file_.seekg(static_cast<std::streamsize>(offset), std::ios_base::beg);
        if (!file_) {
            error = "ReadTensorData: seek failed";
            return false;
        }
        data.resize(static_cast<size_t>(tensorInfo.byte_size));
        ReadExact(file_, data.data(), data.size());
        error.clear();
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }

    return true;
}

}   // namespace mini_llama