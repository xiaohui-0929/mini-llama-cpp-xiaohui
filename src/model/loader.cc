#include "mini_llama/model/loader.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <limits>
#include <cmath>
#include <algorithm>

namespace mini_llama {

static MiniLlamaModel LoadFailure(const std::string& message) {
    MiniLlamaModel model;
    model.loaded = false;
    model.load_error = message;
    std::cerr << "Failed to load model: " << message << std::endl;
    return model;
}

/**
 * 寻找content中指定key的位置
 */
static size_t FindConfigKey(const std::string& content,
                            const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    size_t key_pos = content.find(marker);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing config key: " + key);
    }
    size_t colon_pos = content.find(":", key_pos + marker.size());
    if (colon_pos == std::string::npos) {
        throw std::runtime_error("missing ':' after config key: " + key);
    }
    size_t value_pos = colon_pos + 1;
    while (value_pos < content.size() && 
           std::isspace(static_cast<unsigned char>(content[value_pos]))) {
        ++value_pos;
    }
    if (value_pos >= content.size()) {
        throw std::runtime_error("missing value for config key: " + key);
    }
    return value_pos;
}

/**
 * 检查字符是否为JsonValue的终止
 */
static bool IsJsonValueTerminator(char c) {
    return std::isspace(static_cast<unsigned char>(c)) || c == ',' || c == '}' ||
        c == ']';
}

/**
 * 检查是否为有效数值的终止
 */
static void ValidateNumberTerminator(const std::string& content,
                                     size_t value_pos, size_t parsed,
                                     const std::string& key) {
    size_t pos = value_pos + parsed;
    while (pos < content.size() && 
           std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
    }
    if (pos < content.size() && !IsJsonValueTerminator(content[pos])) {
        throw std::runtime_error("invalid numeric value for config key: " + key);
    }
}

/**
 * 获取指定key的Int数值类型数据
 */
static int ParseRequiredInt(const std::string& content, 
                            const std::string& key) {
    size_t value_pos = FindConfigKey(content, key);
    size_t parsed = 0;
    int value = 0;
    try {
        value = std::stoi(content.substr(value_pos), &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer for config key: " + key);
    }
    if (parsed == 0) {
        throw std::runtime_error("invalid integer for config key: " + key);
    }
    ValidateNumberTerminator(content, value_pos, parsed, key);
    return value;
}

/**
 * 获取指定key的Float数值类型数据
 */
static float ParseRequiredFloat(const std::string& content,
                                const std::string& key) {
    size_t value_pos = FindConfigKey(content, key);
    size_t parsed = 0;
    float value = 0.0f;
    try {
        value = std::stof(content.substr(value_pos), &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid float for config key: " + key);
    }
    if (parsed == 0) {
        throw std::runtime_error("invalid float for config key: " + key);
    }
    ValidateNumberTerminator(content, value_pos, parsed, key);
    return value;
}

/**
 * 检查ModelConfig的有效性
 */
static void ValidateConfig(const ModelConfig& config) {
    if (config.vocab_size <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
        config.n_layers <= 0 || config.n_heads <= 0 || config.n_kv_heads <= 0 ||
        config.max_seq_len <= 0) {
        throw std::runtime_error("all model dimensions must be positive");
    }
    // 加载时便对必要的数学前提条件进行检查
    if (config.dim % config.n_heads != 0) {
        throw std::runtime_error("dim must be divisible by n_heads");
    }
    if (config.n_heads % config.n_kv_heads != 0) {
        throw std::runtime_error("n_heads must be divisible by n_kv_heads");
    }
    if (config.head_dim <= 0 || config.head_dim % 2 != 0) {
        throw std::runtime_error("head_dim must be a positive even number");
    }
    if (!std::isfinite(config.rope_theta) || config.rope_theta <= 0.0f) {
        throw std::runtime_error("rope_theta must be positive");
    }
    if (!std::isfinite(config.rms_norm_eps) || config.rms_norm_eps <= 0.0f) {
        throw std::runtime_error("rms_norm_eps must be positive");
    }
}

/**
 * 
 */
static std::vector<std::pair<std::string, std::vector<int>>>
ExpectedTensorShapes(const ModelConfig& config) {
    std::vector<std::pair<std::string, std::vector<int>>> expected = {
        {"token_embedding", {config.vocab_size, config.dim}},
        {"final_norm", {config.dim}},
        {"lm_head", {config.vocab_size, config.dim}},
    };

    for (int layer = 0; layer < config.n_layers; ++layer) {
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        expected.push_back({prefix + "attention_norm", {config.dim}});
        expected.push_back(
            {prefix + "wq", {config.n_heads * config.head_dim, config.dim}});
        expected.push_back(
            {prefix + "wk", {config.n_kv_heads * config.head_dim, config.dim}});
        expected.push_back(
            {prefix + "wv", {config.n_kv_heads * config.head_dim, config.dim}});
        expected.push_back(
            {prefix + "wo", {config.dim, config.n_heads * config.head_dim}});
        expected.push_back({prefix + "ffn_norm", {config.dim}});
        expected.push_back({prefix + "w_gate", {config.hidden_dim, config.dim}});
        expected.push_back({prefix + "w_up", {config.hidden_dim, config.dim}});
        expected.push_back({prefix + "w_down", {config.dim, config.hidden_dim}});
    }
    return expected;
}

/**
 * 解析Json格式的ModelConfig信息
 */
static ModelConfig ParseConfigJson(const std::string& content) {
    ModelConfig config;

    config.vocab_size = ParseRequiredInt(content, "vocab_size");
    config.dim = ParseRequiredInt(content, "dim");
    config.hidden_dim = ParseRequiredInt(content, "hidden_dim");
    config.n_layers = ParseRequiredInt(content, "n_layers");
    config.n_heads = ParseRequiredInt(content, "n_heads");
    config.n_kv_heads = ParseRequiredInt(content, "n_kv_heads");
    config.max_seq_len = ParseRequiredInt(content, "max_seq_len");
    config.rope_theta = ParseRequiredFloat(content, "rope_theta");
    config.rms_norm_eps = ParseRequiredFloat(content, "rms_norm_eps");
    config.head_dim = config.dim / config.n_heads;

    ValidateConfig(config);

    return config;
}

/**
 * 获取指定key的JsonObject(std::string格式)
 */
static std::string ParseObjectForKey(const std::string& content,
                                     const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    size_t key_pos = content.find(marker);
    if (key_pos == std::string::npos) {
        return "";
    }
    size_t obj_start = content.find("{", key_pos + marker.size());
    if (obj_start == std::string::npos) {
        throw std::runtime_error("missing object for key: " + key);
    }
    // 通过'{' '}'对的匹配 获取object的范围
    int depth = 0;
    for (size_t pos = obj_start; pos < content.size(); ++pos) {
        if (content[pos] == '{') {
            ++depth;
        } else if (content[pos] == '}') {
            --depth;
            if (depth == 0) {
                return content.substr(obj_start, pos - obj_start + 1);
            }
        }
    }
    throw std::runtime_error("unclosed object for key: " + key);
}

/**
 * 尝试获取可能存在的Key
 */
static size_t FindOptionalKey(const std::string& content,
                              const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    size_t key_pos = content.find(marker);
    if (key_pos == std::string::npos) {
        return std::string::npos;
    }
    size_t colon_pos = content.find(":", key_pos + marker.size());
    if (colon_pos == std::string::npos) {
        throw std::runtime_error("missing ':' after key: " + key);
    }
    size_t value_pos = colon_pos + 1;
    while (value_pos < content.size() &&
            std::isspace(static_cast<unsigned char>(content[value_pos]))) {
        ++value_pos;
    }
    if (value_pos >= content.size()) {
        throw std::runtime_error("missing value for key: " + key);
    }
    return value_pos;
}

/**
 * 尝试获取可能存在的字符串字段
 */
static std::string ParseOptionalStringField(const std::string& content,
                                            const std::string& key,
                                            const std::string& fallback) {
    size_t value_pos = FindOptionalKey(content, key);
    if (value_pos == std::string::npos) {
        return fallback;
    }
    if (content[value_pos] != '"') {
        throw std::runtime_error("expected string value for key: " + key);
    }
    size_t end = content.find('"', value_pos + 1);
    if (end == std::string::npos) {
        throw std::runtime_error("unterminated string value for key: " + key);
    }
    return content.substr(value_pos + 1, end - value_pos - 1);
}

/**
 * 尝试获取可能存在的Int数值类型字段
 */
static int ParseOptionalIntField(const std::string& content,
                                 const std::string& key, int fallback) {
    size_t value_pos = FindOptionalKey(content, key);
    if (value_pos == std::string::npos) {
        return fallback;
    }
    size_t parsed = 0;
    int value = 0;
    try {
        value = std::stoi(content.substr(value_pos), &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer for key: " + key);
    }
    if (parsed == 0) {
        throw std::runtime_error("invalid integer for key: " + key);
    }
    ValidateNumberTerminator(content, value_pos, parsed, key);
    return value;
}

/**
 * 解析Json格式的Tokenizer信息
 */
static TokenizerInfo ParseTokenizerJson(const std::string& content) {
    TokenizerInfo tokenizer;

    std::string obj = ParseObjectForKey(content, "tokenizer");
    if (obj.empty()) {
        return tokenizer;
    }

    tokenizer.type = ParseOptionalStringField(obj, "type", tokenizer.type);
    tokenizer.path = ParseOptionalStringField(obj, "path", tokenizer.path);
    tokenizer.bos_id = ParseOptionalIntField(obj, "bos_id", tokenizer.bos_id);
    tokenizer.eos_id = ParseOptionalIntField(obj, "eos_id", tokenizer.eos_id);
    tokenizer.unk_id = ParseOptionalIntField(obj, "unk_id", tokenizer.unk_id);

    // 几种错误的解析情况
    if (tokenizer.type != "ascii" && tokenizer.type != "json_vocab") {
        throw std::runtime_error("unsupported tokenizer type: " + tokenizer.type);
    }
    if (tokenizer.bos_id < 0 || tokenizer.eos_id < 0 || tokenizer.unk_id < 0) {
        throw std::runtime_error("tokenizer special ids must be non-negative");
    }
    if (tokenizer.bos_id == tokenizer.eos_id ||
        tokenizer.bos_id == tokenizer.unk_id ||
        tokenizer.eos_id == tokenizer.unk_id) {
        throw std::runtime_error("tokenizer special ids must be distinct");
    }
    if (tokenizer.type == "json_vocab" && tokenizer.path.empty()) {
        throw std::runtime_error("json_vocab tokenizer requires a path");
    }

    return tokenizer;
}

/**
 * 解析Json格式的Tensor metadata数组
 */
static std::vector<TensorInfo> ParseTensorMetadata(const std::string& content) {
    std::vector<TensorInfo> result;

    const std::string tensors_marker = "\"tensors\"";
    size_t tensors_pos = content.find(tensors_marker);
    if (tensors_pos == std::string::npos) {
        throw std::runtime_error("missing 'tensors' field in model.json");
    }

    size_t bracket_open = content.find("[", tensors_pos);
    if (bracket_open == std::string::npos) {
        throw std::runtime_error("missing '[' after 'tensors'");
    }

    // 解析TensorInfo
    size_t i = bracket_open + 1;
    int depth = 1;
    while (i < content.size() && depth > 0) {
        // 寻找下一个TensorInfo的开始位置
        while (i < content.size() && content[i] != '{') {
            if (content[i] == '[') {
                ++depth;
            } else if (content[i] == ']') {
                --depth;
            }
            ++i;
        }
        if (depth <= 0 || i >= content.size()) {
            break;
        }

        // 获取单个TensorInfo的范围
        TensorInfo info;
        size_t obj_start = i;
        size_t obj_end = content.find("}", obj_start);
        if (obj_end == std::string::npos) {
            throw std::runtime_error("unclosed tensor object");
        }
        std::string obj = content.substr(obj_start, obj_end - obj_start + 1);

        bool has_name = false;
        bool has_dtype = false;
        bool has_shape = false;
        bool has_offset = false;
        bool has_byte_size = false;

        // Parse name
        size_t name_pos = obj.find("\"name\"");
        if (name_pos != std::string::npos) {
            size_t name_colon = obj.find(":", name_pos);
            size_t name_quote = obj.find("\"", name_colon + 1);
            size_t name_end = obj.find("\"", name_quote + 1);
            if (name_quote != std::string::npos && name_end != std::string::npos) {
                info.name = obj.substr(name_quote + 1, name_end - name_quote - 1);
                has_name = true;
            }
        }

        // Parse dtype
        size_t dtype_pos = obj.find("\"dtype\"");
        if (dtype_pos != std::string::npos) {
            size_t dtype_colon = obj.find(":", dtype_pos);
            size_t dtype_quote = obj.find("\"", dtype_colon + 1);
            size_t dtype_end = obj.find("\"", dtype_quote + 1);
            if (dtype_quote != std::string::npos && dtype_end != std::string::npos) {
                info.dtype = obj.substr(dtype_quote + 1, dtype_end - dtype_quote - 1);
                has_dtype = true;
            }
        }

        // Parse shape array
        size_t shape_pos = obj.find("\"shape\"");
        if (shape_pos != std::string::npos) {
            size_t shape_bracket = obj.find("[", shape_pos);
            size_t shape_end = obj.find("]", shape_bracket);
            if (shape_bracket != std::string::npos &&
                shape_end != std::string::npos) {
                std::string shape_string_short =
                    obj.substr(shape_bracket + 1, shape_end - shape_bracket - 1);
                std::stringstream ss(shape_string_short);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    // Trim whitespace
                    size_t start = token.find_first_not_of(" \t\n\r");
                    size_t end = token.find_last_not_of(" \t\n\r");
                    if (start != std::string::npos) {
                        int dim = std::stoi(token.substr(start, end - start + 1));
                        info.shape.push_back(dim);
                    }
                }
                has_shape = true;
            }
        }

        // Parse offset
        size_t offset_pos = obj.find("\"offset\"");
        if (offset_pos != std::string::npos) {
            size_t offset_colon = obj.find(":", offset_pos);
            if (offset_colon == std::string::npos) {
                throw std::runtime_error("tensor offset field is malformed");
            }
            info.offset =
                static_cast<size_t>(std::stoull(obj.substr(offset_colon + 1)));
            has_offset = true;
        }

        // Parse byte_size
        size_t bs_pos = obj.find("\"byte_size\"");
        if (bs_pos != std::string::npos) {
            size_t bs_colon = obj.find(":", bs_pos);
            if (bs_colon == std::string::npos) {
                throw std::runtime_error("tensor byte_size field is malformed");
            }
            info.byte_size =
                static_cast<size_t>(std::stoull(obj.substr(bs_colon + 1)));
            has_byte_size = true;
        }

        if (!has_name || info.name.empty()) {
            throw std::runtime_error("tensor missing name field");
        }
        if (!has_dtype) {
            throw std::runtime_error("tensor missing dtype field: " + info.name);
        }
        if (!has_shape) {
            throw std::runtime_error("tensor missing shape field: " + info.name);
        }
        if (!has_offset) {
            throw std::runtime_error("tensor missing offset field: " + info.name);
        }
        if (!has_byte_size) {
            throw std::runtime_error("tensor missing byte_size field: " + info.name);
        }

        result.push_back(info);
        i = obj_end + 1;
    }

    return result;
}

/**
 * 检查Tensor字节大小
 */
static size_t CheckedTensorByteSize(const TensorInfo& info) {
    size_t num_elements = 1;
    // 检查Tensor的大小是否越界
    for (int dim : info.shape) {
        if (dim <= 0) {
            throw std::runtime_error("invalid tensor shape for " + info.name);
        }
        const size_t d = static_cast<size_t>(dim);
        if (num_elements > std::numeric_limits<size_t>::max() / d) {
            throw std::runtime_error("tensor size overflow for " + info.name);
        }
        num_elements *= d;
    }
    // 检查是否有足够的byte容纳Tensor
    if (num_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::runtime_error("tensor byte size overflow for " + info.name);
    }
    return num_elements * sizeof(float);
}

/**
 * shape 转为 string 形式
 */
static std::string ShapeToString(const std::vector<int>& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
        result += ", ";
        }
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

/**
 * 检查Manifest正确性
 */
static void ValidateManifest(const ModelManifest& manifest) {
    if (manifest.tensors.empty()) {
        throw std::runtime_error("model.json contains no tensors");
    }

    std::map<std::string, TensorInfo> tensor_map;
    std::vector<TensorInfo> by_offset;
    by_offset.reserve(manifest.tensors.size());

    // 逐项检查TensorInfo
    for (const auto& info : manifest.tensors) {
        if (info.dtype != "float32") {
            throw std::runtime_error("unsupported dtype '" + info.dtype +
                                    "' for tensor: " + info.name);
        }
        if (info.shape.empty()) {
            throw std::runtime_error("tensor shape must not be empty: " + info.name);
        }
        // 检查每个Tensor的大小
        size_t expected_bytes = CheckedTensorByteSize(info);
        if (expected_bytes != info.byte_size) {
            throw std::runtime_error("tensor byte_size mismatch for " + info.name +
                                    ": expected " + std::to_string(expected_bytes) +
                                    ", manifest says " +
                                    std::to_string(info.byte_size));
        }
        if (info.offset > std::numeric_limits<size_t>::max() - info.byte_size) {
            throw std::runtime_error("tensor offset overflow for " + info.name);
        }
        // Tensor 的名称不能重复
        auto inserted = tensor_map.emplace(info.name, info);
        if (!inserted.second) {
            throw std::runtime_error("duplicate tensor name: " + info.name);
        }
        by_offset.push_back(info);
    }

    // 对Tensor按照offset大小 由小到大进行排序 
    // 检查[offset, offset + byte_size)是否重叠
    std::sort(by_offset.begin(), by_offset.end(),
              [](const TensorInfo& a, const TensorInfo& b) {
                    return a.offset < b.offset;
              });
    for (size_t i = 1; i < by_offset.size(); ++i) {
        size_t previous_end = by_offset[i - 1].offset + by_offset[i - 1].byte_size;
        if (by_offset[i].offset < previous_end) {
            throw std::runtime_error(
                "tensor ranges overlap: " + by_offset[i - 1].name + " and " +
                by_offset[i].name);
        }
    }

    // 全景检查tensor是否有遗漏 shape是否有误
    for (const auto& expected : ExpectedTensorShapes(manifest.config)) {
        auto it = tensor_map.find(expected.first);
        if (it == tensor_map.end()) {
            throw std::runtime_error("missing required tensor: " + expected.first);
        }
        if (it->second.shape != expected.second) {
            throw std::runtime_error("tensor shape mismatch for " + expected.first +
                                    ": expected " + ShapeToString(expected.second) +
                                    ", manifest says " +
                                    ShapeToString(it->second.shape));
        }
    }
}

ModelManifest ParseManifest(const std::string& config_path) {
    ModelManifest manifest;

    std::ifstream cf(config_path);
    if (!cf.is_open()) {
        throw std::runtime_error("failed to open config: " + config_path);
    }
    std::stringstream buffer;
    buffer << cf.rdbuf();
    if (cf.bad()) {
        throw std::runtime_error("failed to read config: " + config_path);
    }

    std::string content = buffer.str();
    manifest.config = ParseConfigJson(content);
    manifest.tokenizer = ParseTokenizerJson(content);
    manifest.tensors = ParseTensorMetadata(content);
    // 检查Manifest正确性
    ValidateManifest(manifest);

    return manifest;
}

/**
 * 根据名称加载Tensor
 */
static Tensor LoadTensorByName(std::ifstream& wf, 
                               const std::map<std::string, TensorInfo>& tensor_map,
                               const std::string& name) {
    // 从tensor_map内找到对应名称的TensorInfo
    auto it = tensor_map.find(name);
    if (it == tensor_map.end()) {
        throw std::runtime_error("missing required tensor: " + name);
    }
    const TensorInfo& info = it->second;
    // 偏移量超过了流偏移上限
    if (info.offset >
        static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("tensor offset exceeds stream limit for " + name);
    }
    // 字节大小超过了流大小上限
    if (info.byte_size >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("tensor byte size exceeds stream limit for " + name);
    }
    // 定位到对应Tensro的偏移处
    wf.seekg(static_cast<std::streamoff>(info.offset), std::ios::beg);
    if (!wf.good()) {
        throw std::runtime_error("failed to seek to offset for tensor: " + name);
    }
    Tensor t(info.shape, 0.0f);
    // 尝试将指定大小的字节信息读出
    wf.read(reinterpret_cast<char*>(t.data_.data()),
            static_cast<std::streamsize>(info.byte_size));
    if (static_cast<size_t>(wf.gcount()) != info.byte_size) {
        throw std::runtime_error("weights file ended while reading tensor: " + name);
    }

    return t;
}

/**
 * 将权重信息解析进入Model
 */
static void ParseWeightInfoToModel(MiniLlamaModel& model, 
                                   std::ifstream& wf,
                                   const std::map<std::string, TensorInfo>& tensor_map,
                                   const ModelManifest& manifest) {
    // 加载embedding 
    model.token_embedding = LoadTensorByName(wf, tensor_map, "token_embedding");
    model.layers.resize(manifest.config.n_layers);
    for (int layer = 0; layer < manifest.config.n_layers; ++layer) {
        LayerWeights& lw = model.layers[layer];
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        lw.attention_norm =
            LoadTensorByName(wf, tensor_map, prefix + "attention_norm");
        lw.wq =
            LoadTensorByName(wf, tensor_map, prefix + "wq");
        lw.wk =
            LoadTensorByName(wf, tensor_map, prefix + "wk");
        lw.wv =
            LoadTensorByName(wf, tensor_map, prefix + "wv");
        lw.wo = 
            LoadTensorByName(wf, tensor_map, prefix + "wo");
        lw.ffn_norm = 
            LoadTensorByName(wf, tensor_map, prefix + "ffn_norm");
        lw.w_gate = 
            LoadTensorByName(wf, tensor_map, prefix + "w_gate");
        lw.w_up = 
            LoadTensorByName(wf, tensor_map, prefix + "w_up");
        lw.w_down = 
            LoadTensorByName(wf, tensor_map, prefix + "w_down");
    }
    model.final_norm = LoadTensorByName(wf, tensor_map, "final_norm");
    model.lm_head = LoadTensorByName(wf, tensor_map, "lm_head");
}

MiniLlamaModel LoadModel(const std::string& config_path,
                         const std::string& weights_path) {
    // 解析配置文件信息
    ModelManifest manifest;
    try {
        manifest = ParseManifest(config_path);
    } catch (const std::exception& e) {
        return LoadFailure(e.what());
    }
    // 打印解析信息
    std::cout << "Loaded config: vocab=" << manifest.config.vocab_size
                << " dim=" << manifest.config.dim
                << " layers=" << manifest.config.n_layers
                << " heads=" << manifest.config.n_heads
                << " kv_heads=" << manifest.config.n_kv_heads
                << " head_dim=" << manifest.config.head_dim
                << " hidden_dim=" << manifest.config.hidden_dim << std::endl;

    std::map<std::string, TensorInfo> tensor_map;
    for (const auto& info : manifest.tensors) {
        tensor_map[info.name] = info;
    }

    // 打开二进制权重文件
    std::ifstream wf(weights_path, std::ios::binary);
    if (!wf.is_open()) {
        return LoadFailure("failed to open weights: " + weights_path);
    }

    MiniLlamaModel model;
    model.config = manifest.config;
    // 解析权重信息
    try {
        ParseWeightInfoToModel(model, wf, tensor_map, manifest);
    } catch (const std::exception& e) {
        return LoadFailure(e.what());
    }
    // 获取实际的文件大小
    wf.seekg(0, std::ios::end);
    std::streamoff file_size = wf.tellg();
    if (file_size < 0) {
        return LoadFailure("failed to determine weights file size");
    }
    // 获取期望的文件大小
    size_t expected_end = 0;
    for (const auto& info : manifest.tensors) {
        size_t tensor_end = info.offset + info.byte_size;
        if (tensor_end > expected_end) {
            expected_end = tensor_end;
        }
    }
    // 对两者不相等情况的处理
    if (static_cast<size_t>(file_size) != expected_end) {
        if (static_cast<size_t>(file_size) > expected_end) {
            return LoadFailure(
                "weights file has trailing bytes: file size=" +
                std::to_string(file_size) +
                ", expected=" + std::to_string(expected_end));
        } else {
            return LoadFailure(
                "weights file is too short: file size=" + 
                std::to_string(file_size) +
                ", expected=" + std::to_string(expected_end));
        }
    }

    model.loaded = true;
    std::cout << "Model loaded successfully from " << weights_path << std::endl;
    return model;
}

bool InspectModel(const std::string& config_path) {
    ModelManifest manifest;
    try {
        manifest = ParseManifest(config_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to inspect model: " << e.what() << std::endl;
        return false;
    }

    const ModelConfig& c = manifest.config;
    std::cout << "=== Model Config ===" << std::endl;
    std::cout << "  vocab_size:    " << c.vocab_size << std::endl;
    std::cout << "  dim:           " << c.dim << std::endl;
    std::cout << "  hidden_dim:    " << c.hidden_dim << std::endl;
    std::cout << "  n_layers:      " << c.n_layers << std::endl;
    std::cout << "  n_heads:       " << c.n_heads << std::endl;
    std::cout << "  n_kv_heads:    " << c.n_kv_heads << std::endl;
    std::cout << "  head_dim:      " << c.head_dim << std::endl;
    std::cout << "  max_seq_len:   " << c.max_seq_len << std::endl;
    std::cout << "  rope_theta:    " << c.rope_theta << std::endl;
    std::cout << "  rms_norm_eps:  " << c.rms_norm_eps << std::endl;

    std::cout << std::endl;
    std::cout << "=== Tensors (" << manifest.tensors.size()
                << ") ===" << std::endl;

    size_t total_bytes = 0;
    for (const auto& t : manifest.tensors) {
        std::cout << "  " << t.name << std::endl;
        std::cout << "    shape:     [";
        for (size_t i = 0; i < t.shape.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << t.shape[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "    dtype:     " << t.dtype << std::endl;
        std::cout << "    offset:    " << t.offset << std::endl;
        std::cout << "    byte_size: " << t.byte_size << std::endl;
        total_bytes += t.byte_size;
    }

    std::cout << std::endl;
    std::cout << "=== Tokenizer ===" << std::endl;
    std::cout << "  type: " << manifest.tokenizer.type << std::endl;
    std::cout << "  path: "
                << (manifest.tokenizer.path.empty() ? "<none>"
                                                    : manifest.tokenizer.path)
                << std::endl;
    std::cout << "  bos_id: " << manifest.tokenizer.bos_id << std::endl;
    std::cout << "  eos_id: " << manifest.tokenizer.eos_id << std::endl;
    std::cout << "  unk_id: " << manifest.tokenizer.unk_id << std::endl;
    std::cout << std::endl;
    std::cout << "Total tensor bytes: " << total_bytes << std::endl;
    return true;
}

}   // namespace mini_llama