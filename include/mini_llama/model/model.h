#ifndef INCLUDE_MINI_LLAMA_MODEL_H_
#define INCLUDE_MINI_LLAMA_MODEL_H_

#include "mini_llama/tensor_and_Ops/tensor.h"

namespace mini_llama {

enum class RopeType {
    kNormal,    // LLaMA风格
    kNeoX,      // Qwen风格
};

// 模型所需的配置
struct ModelConfig {
    int vocab_size = 128;
    int dim = 32;
    int hidden_dim = 86;
    int n_layers = 2;
    int n_heads = 4;
    int n_kv_heads = 4;
    int head_dim = 8;
    int max_seq_len = 128;
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-5f;
    RopeType rope_type = RopeType::kNormal;
};

// 每层权重
struct LayerWeights {
    Tensor attention_norm;
    Tensor bq;
    Tensor bk;
    Tensor bv;
    Tensor ffn_norm;
};

struct MiniLlamaModel {
    ModelConfig config;
    bool loaded = false;
    std::string load_error;
    Tensor token_embedding;
    std::vector<LayerWeights> layers;
    Tensor final_norm;
};

}   // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_MODEL_H_