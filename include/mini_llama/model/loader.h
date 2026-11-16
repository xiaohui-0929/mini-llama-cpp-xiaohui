#ifndef INCLUDE_MINI_LLAMA_LOADER_H_
#define INCLUDE_MINI_LLAMA_LOADER_H_

#include "mini_llama/model/model.h"

namespace mini_llama {

// Tensor信息
struct TensorInfo {
    std::string name;
    std::vector<int> shape;
    std::string dtype;
    size_t offset = 0;
    size_t byte_size = 0;
};

// Tokenizer信息
struct TokenizerInfo {
    std::string type = "ascii";
    std::string path;
    int bos_id = 1;
    int eos_id = 2;
    int unk_id = 0;
};

struct ModelManifest {
    ModelConfig config;
    TokenizerInfo tokenizer;
    std::vector<TensorInfo> tensors;
};

/**
 * 解析元数据信息 json文件
 */
ModelManifest ParseManifest(const std::string& config_path);

/**
 * 加载模型 json文件 + bin文件
 */
MiniLlamaModel LoadModel(const std::string& config_path,
                         const std::string& weights_path);
/**
 * json文件快照
 */
bool InspectModel(const std::string& config_path);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_LOADER_H_