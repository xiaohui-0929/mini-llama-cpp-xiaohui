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
    // attention层
    Tensor attention_norm;  // norm权重 [dim] F32
    Tensor wq;              // q的w权重 [n_heads * head_dim, dim] F32(未来会进行量化)
    Tensor wk;              // k的w权重 [n_kv_heads * head_dim, dim] F32(未来会进行量化)
    Tensor wv;              // v的w权重 [n_kv_heads * head_dim, dim] F32(未来会进行量化)
    Tensor bq;              // q的偏置项(可选) [n_heads * head_dim] F32
    Tensor bk;              // k的偏置项(可选) [n_kv_heads * head_dim] F32
    Tensor bv;              // v的偏置项(可选) [n_kv_heads * head_dim] F32
    Tensor wo;              // Attention后的融合权重矩阵 [dim, n_heads * head_dim]

    // ffn层
    Tensor ffn_norm;        // norm权重 [dim] F32
    Tensor w_gate;          // 门控权重 [hidden_dim, dim] F32(未来会进行量化)
    Tensor w_up;            // 升维权重 [hidden_dim, dim] F32(未来会进行量化)
    Tensor w_down;          // 降维权重 [dim, hidden_dim] F32(未来会进行量化)
};

struct MiniLlamaModel {
    ModelConfig config;
    bool loaded = false;
    std::string load_error;
    Tensor token_embedding;             // 词表内对应token的特征向量矩阵 [vocab_size, dim] F32
    std::vector<LayerWeights> layers;   // 每层的权重信息
    Tensor final_norm;                  // norm权重 [dim] F32
    Tensor lm_head;                     // 语言模型头 [vocab_size, dim] F32(未来会进行量化)
};

}   // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_MODEL_H_