#include "mini_llama/runtime/context.h"

namespace mini_llama {

MiniLlamaContext::MiniLlamaContext(const MiniLlamaModel* model_ptr)
    : model(model_ptr), pos(0) {
    if (model) {
        const auto& config = model->config;
        kv_cache = KvCache(config.n_layers, config.max_seq_len, 
                           config.n_kv_heads, config.head_dim);
    }
}

}   // namespace mini_llama