#ifndef INCLUDE_MINI_LLAMA_GGUF_H_
#define INCLUDE_MINI_LLAMA_GGUF_H_

#include <string>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <fstream>

namespace mini_llama {

/**
 * ggml type    仅列出部分类型 其余类型后续补充
 */
enum class GgufGgmlType {
    kggml_F32   = 0,
    kggml_F16   = 1,
    //kggml_Q8_0  = 8,  // 暂不考虑
    //kggml_Q4_K  = 12  // 暂不考虑
};

/**
 * gguf metadata的值类型
 */
enum class GgufMetadataValueType {
    kgguf_uint8      = 0,
    kgguf_int8       = 1,
    kgguf_uint16     = 2,
    kgguf_int16      = 3,
    kgguf_uint32     = 4,
    kgguf_int32      = 5,
    kgguf_float32    = 6,
    kgguf_bool       = 7,
    kgguf_string     = 8,
    kgguf_array      = 9,
    kgguf_uint64     = 10,
    kgguf_int64      = 11,
    kgguf_float64    = 12
};

/**
 * gguf metadata的value结构体
 * 当前实现按照实际文件组成对array类型的value进行了嵌套 可能导致加载时内存激增 是未来优化的一个点
 */
struct GgufMetadataValue {
    GgufMetadataValueType value_type = GgufMetadataValueType::kgguf_uint8;  // 本值的类型

    uint64_t    u64 = 0;                    // 所有无符号整型标量
    int64_t     i64 = 0;                    // 所有有符号整型标量
    double      f64 = 0.0;                  // f32/f64
    bool        bool_ = false;              // bool
    std::string string;                     // 字符串
    std::vector<GgufMetadataValue> array;   // 嵌套array
};

/**
 *  gguf格式文件 header信息 
 */ 
struct GgufHeader {
    uint32_t magic = 0;             // 文件magic "GGUF" = 0x46554747
    uint32_t version = 0;           // 版本
    uint64_t tensor_info_count = 0; // tensor_info数量
    uint64_t metadata_kv_count = 0; // meatadata_kv数量
};

/**
 * gguf格式文件 metadata_kv
 */
struct GgufMetadataKV {
    std::string key;
    GgufMetadataValue value;
};

/**
 * gguf格式文件 tensor_info
 */
struct GgufTensorInfo {
    std::string name;
    std::vector<uint64_t> dims;
    GgufGgmlType ggml_type = GgufGgmlType::kggml_F32;
    uint64_t offset = 0;
    uint64_t byte_size = 0;     // 仅 F32/F16 计算；其他类型为 0，ReadTensorData 时报 unsupported
};

/**
 * 解析gguf格式文件 文件信息缓存进该Reader内
 */
class GgufReader {
public:
    GgufReader() = default;
    ~GgufReader() = default;
    
    GgufReader(const GgufReader&) = delete;
    GgufReader& operator=(const GgufReader&) = delete;

    GgufReader(GgufReader&&) noexcept = default;
    GgufReader& operator=(GgufReader&&) noexcept = default;

    /**
     * uint32_t转为ggml类型
     */
    static const char* GgmlTypeName(uint32_t type);

    /**
     * uint32_t转为MetadataValueType类型
     */
    static const char* GgufMetadataValueTypeName(uint32_t type);

    /**
     * 打开文件解析
     * 可重复调用：每次进入前先 Clear()，关闭旧文件、清空缓存。
     * 失败时：返回 false，error 非空，Loaded()==false，
     *         Metadata()/TensorInfo() 为空，Header() 为默认值。
     * 成功时：清空error信息
     */
    bool Open(const std::string& path, std::string& error);

    /**
     * 获取解析的header信息
     */
    const GgufHeader& Header() const { return header_;}

    /**
     * 获取解析的alignment信息
     */
    uint32_t Alignment() const { return alignment_; }

    /**
     * 获取tensor_data的offset
     */
    uint64_t DataOffset() const { return data_offset_; }

    /**
     * 获取加载状态
     */
    bool Loaded() const { return loaded_; }

    /**
     * 清空缓存信息
     */
    void Clear();

    /**
     * 获取按序存储的metadata_kv
     */
    const std::vector<GgufMetadataKV>& Metadata() const { return metadata_kv_; }

    /**
     * 获取按序存储的tensor_info
     */
    const std::vector<GgufTensorInfo>& TensorInfo() const { return tensor_info_; }
    
    /**
     * 通过名称获取metadata_kv 获取失败时返回nullptr
     */
    const GgufMetadataKV* GetMetadataByKey(const std::string& key) const;

    /**
     * 通过名称获取tensor_info 获取失败时返回nullptr
     */
    const GgufTensorInfo* GetTensorInfo(const std::string& tensor_name) const;

    /**
     * 通过tensor_info的名称 获取对应的tensor_data数据
     * 会移动内部维护的文件句柄 非线程安全
     * 返回结果如果为false 则data内数据不应该使用
     */
    bool ReadTensorData(const std::string& tensor_name, std::vector<uint8_t>& data, std::string& error);

private:
    bool loaded_ = false;
    uint64_t file_byte_size_ = 0;
    uint64_t data_offset_ = 0;
    uint32_t alignment_ = 0;                                             
    std::ifstream file_;                                            // 文件句柄
    GgufHeader header_;                                             // gguf的header信息
    std::vector<GgufMetadataKV> metadata_kv_;                       // 按解析顺序存储的metadata_kv 
    std::vector<GgufTensorInfo> tensor_info_;                       // 按解析顺序存储的tensor_info
    std::unordered_map<std::string, size_t> name_to_metadata_kv_;   // metadata_kv的名称到下标映射
    std::unordered_map<std::string, size_t> name_to_tensor_info_;   // tensor_info的名称到下标映射
};

}   // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_GGUF_H_