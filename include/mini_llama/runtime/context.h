#ifndef INCLUDE_MINI_LLAMA_CONTEXT_H_
#define INCLUDE_MINI_LLAMA_CONTEXT_H_

#include "mini_llama/model/model.h"
#include "mini_llama/runtime/kv_cache.h"

namespace mini_llama {

// 推理上下文
struct MiniLlamaContext {
    const MiniLlamaModel* model = nullptr;
    KvCache kv_cache;
    // 位置游标 记录当前正在处理的序列绝对位置
    int pos = 0;

    // 所有经过此上下文处理的token
    std::vector<int> token_history;

    // 计数器
    int n_prefill_tokens = 0;
    int n_decode_tokens = 0;

    MiniLlamaContext() = default;
    MiniLlamaContext(const MiniLlamaContext&) = delete;
    MiniLlamaContext& operator=(const MiniLlamaContext&) = delete;
    MiniLlamaContext(MiniLlamaContext&&) noexcept = default;
    MiniLlamaContext& operator=(MiniLlamaContext&&) noexcept = default;
    explicit MiniLlamaContext(const MiniLlamaModel* model);
};

}   // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CONTEXT_H_