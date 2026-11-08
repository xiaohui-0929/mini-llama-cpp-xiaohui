#ifndef INCLUDE_MINI_LLAMA_KV_CACHE_H_
#define INCLUDE_MINI_LLAMA_KV_CACHE_H_

#include "mini_llama/tensor_and_Ops/tensor.h"

namespace mini_llama {

// keys:   [n_layers, max_seq_len, n_kv_heads, head_dim]
// values: [n_layers, max_seq_len, n_kv_heads, head_dim]
struct KvCache {
    Tensor keys;
    Tensor values;

    KvCache() = default;
    KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim);

    /**
     * 在指定层layer、指定位置pos，写入对应的k、v
     * k: [n_kv_heads, head_dim]
     * v: [n_kv_heads, head_dim]
     */
    void Write(int layer, int pos, const Tensor& k, const Tensor& v);

    /**
     * 以首位元素指针的形式
     * 返回指定layer 指定pos 指定kv_head 的Key/Value
     * 避免了返回整个Tensor或数组产生的拷贝 进一步提升性能
     */
    const float* KeyPtr(int layer, int pos, int kv_head) const;
    const float* ValuePtr(int layer, int pos, int kv_head) const;
};

}   // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_KV_CACHE_H_