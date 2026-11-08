#include "mini_llama/runtime/kv_cache.h"

#include <cstring>

namespace mini_llama {

static void Require4DCacheShape(const Tensor& t, const char* name) {
    if (t.num_dims() != 4) {
        throw std::runtime_error(std::string(name) + " must be a 4D tensor");
    }
}

KvCache::KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim) {
    keys = MakeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
    values = MakeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
}

void KvCache::Write(int layer, int pos, const Tensor& k, const Tensor& v) {
    // k: [n_kv_heads, head_dim]
    // v: [n_kv_heads, head_dim]
    Require4DCacheShape(keys, "keys");
    Require4DCacheShape(values, "values");
    // 检查k v
    if (k.num_dims() != 2 || v.num_dims() != 2 || k.shape_ != v.shape_) {
        throw std::out_of_range(
            "KvCache::Write expected matching [n_kv_heads, head_dim] tensors");
    }
    int n_kv_heads = k.shape_[0];
    int head_dim = k.shape_[1];
    if (n_kv_heads != keys.shape_[2] || head_dim != keys.shape_[3]) {
        throw std::out_of_range(
            "KvCache::Write tensor shape does not match cache shape");
    }
    // 检查layer pos
    if (layer < 0 || layer >= keys.shape_[0]) {
        throw std::out_of_range("KvCache::Write layer out of range");
    }
    if (pos < 0 || pos >= keys.shape_[1]) {
        throw std::out_of_range("KvCache::Write position out of range");
    }

    // 计算一些基础步长
    size_t layer_stride =
            static_cast<size_t>(keys.shape_[1] * keys.shape_[2] * keys.shape_[3]);
    size_t pos_stride = 
            static_cast<size_t>(keys.shape_[2] * keys.shape_[3]);
    size_t head_stride = 
            static_cast<size_t>(head_dim);
    size_t base = layer * layer_stride + pos * pos_stride;
    // 利用张量底层为连续数组的特性
    // 按注意力头数 调用memcpy 批量执行拷贝
    // 每次拷贝单头的维度
    for (int h = 0; h < n_kv_heads; ++h) {
        size_t offset = base + h * head_stride;
        std::memcpy(&keys.data_[offset], &k.data_[h * head_dim],
                    head_dim * sizeof(float));
        std::memcpy(&values.data_[offset], &v.data_[h * head_dim],
                    head_dim * sizeof(float));
    }
}

const float* KvCache::KeyPtr(int layer, int pos, int kv_head) const {
    Require4DCacheShape(keys, "keys");

    if (layer < 0 || layer >= keys.shape_[0] || pos < 0 || pos >= keys.shape_[1] ||
        kv_head < 0 || kv_head >= keys.shape_[2]) {
        throw std::out_of_range("KvCache::KeyPtr FlatIndex out of range");
    }

    size_t layer_stride = 
            static_cast<size_t>(keys.shape_[1] * keys.shape_[2] * keys.shape_[3]);
    size_t pos_stride =
            static_cast<size_t>(keys.shape_[2] * keys.shape_[3]);
    size_t head_stride =
            static_cast<size_t>(keys.shape_[3]);
    
    return &keys.data_[layer * layer_stride + pos * pos_stride + 
                       kv_head * head_stride];
}

const float* KvCache::ValuePtr(int layer, int pos, int kv_head) const {
    Require4DCacheShape(values, "values");

    if (layer < 0 || layer >= values.shape_[0] || pos < 0 || pos >= values.shape_[1] ||
        kv_head < 0 || kv_head >= values.shape_[2]) {
        throw std::out_of_range("KvCache::ValuePtr FlatIndex out of range");
    }

    size_t layer_stride = 
            static_cast<size_t>(values.shape_[1] * values.shape_[2] * values.shape_[3]);
    size_t pos_stride =
            static_cast<size_t>(values.shape_[2] * values.shape_[3]);
    size_t head_stride =
            static_cast<size_t>(values.shape_[3]);
    
    return &values.data_[layer * layer_stride + pos * pos_stride + 
                       kv_head * head_stride];
}

}   // namespace mini_llama