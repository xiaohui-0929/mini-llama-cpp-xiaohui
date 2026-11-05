
#include "mini_llama/tensor_and_Ops/ops.h"

#include <stdexcept>
#include <cmath>
#include <iostream>

#include "mini_llama/tensor_and_Ops/matmul_dispatch.h"

namespace mini_llama {

Tensor Matmul(const Tensor& a, const Tensor&b) {
    return MatmulDispatch(a, b, DefaultMatmulMode());
}

Tensor Linear(const Tensor& x, const Tensor& weight) {
    return LinearDispatch(x, weight, DefaultMatmulMode());
}

Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps) {
    // 一维张量才可以进行归一化处理
    if (x.num_dims() != 1 || weight.num_dims() != 1) {
        throw std::runtime_error("RmsNorm: expected 1D tensors");
    }
    int dim = x.shape_[0];
    if (weight.shape_[0] != dim) {
        // 维度的长度需要匹配
        throw std::runtime_error("RmsNorm: dimension mismatch");
    }

    // 计算数据平方和
    float ss = 0.0f;
    for (int i = 0; i < dim; ++i) {
        ss += x.data_[i] * x.data_[i];
    }
    // 计算均方
    ss /= static_cast<float>(dim);
    // 计算均方根
    float scale = 1.0f / std::sqrt(ss + eps);

    Tensor y({dim}, 0.0f);
    for (int i = 0; i < dim; ++i) {
        // 需要乘以学习权重向量 
        y.data_[i] = x.data_[i] * scale * weight.data_[i];
    }
    return y;
}

Tensor Softmax(const Tensor& x) {
    // 一维张量才可以进行Softmax处理
    if (x.num_dims() != 1) {
        throw std::runtime_error("Softmax: expected 1D tensor, got " +
                                 x.ShapeStringShort());
    }
    if (x.num_elements() == 0) {
        throw std::runtime_error("Softmax: cannot compute Softmax of empty tensor");
    }
    int n = x.shape_[0];
    float max_val = x.data_[0];
    // 寻找最大值
    for (int i = 1; i < n; ++i) {
        if (x.data_[i] > max_val) {
            max_val = x.data_[i];
        }
    }
    float sum = 0.0f;
    Tensor y({n}, 0.0f);
    for (int i = 0; i < n; ++i) {
        // 减去最大值后进行指数化 保证范围落在0-1之间
        y.data_[i] = std::exp(x.data_[i] - max_val);
        sum += y.data_[i];
    }
    for (int i = 0; i < n; ++i) {
        y.data_[i] /= sum;
    }
    return y;
}

Tensor Silu(const Tensor& x) {
    Tensor y(x.shape_, 0.0f);
    for (size_t i = 0; i < x.data_.size(); ++i) {
        float val = x.data_[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-val));
        y.data_[i] = val * sigmoid;
    }
    return y;
}

Tensor ElementwiseMul(const Tensor& a, const Tensor& b) {
    if (a.shape_ != b.shape_) {
        throw std::runtime_error("ElementwiseMul: shape mismatch " +
                                a.ShapeStringShort() + " vs " +
                                b.ShapeStringShort());
    }
    Tensor y(a.shape_, 0.0f);
    for (size_t i = 0; i < a.data_.size(); ++i) {
        y.data_[i] = a.data_[i] * b.data_[i];
    }
    return y;
}

Tensor SwiGlu(const Tensor& gate, const Tensor& up) {
    if (gate.shape_ != up.shape_) {
        throw std::runtime_error("SwiGlu: shape mismatch " +
                                 gate.ShapeStringShort() + " vs " +
                                 up.ShapeStringShort());
    }

    Tensor y(gate.shape_, 0.0f);
    for (size_t i = 0; i < gate.data_.size(); ++i) {
        float val = gate.data_[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-val));
        y.data_[i] = val * sigmoid * up.data_[i];
    }
    return y;
}

static void ApplyRopeNormalToHeads(Tensor& x, int pos, float theta) {
    int n_heads = x.shape_[0];
    int head_dim = x.shape_[1];

    for (int h = 0; h < n_heads; ++h) {
        int base = h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / std::pow(theta, static_cast<float>(i) /
                                            static_cast<float>(head_dim));
            float cos_val = std::cos(pos * freq);
            float sin_val = std::sin(pos * freq);
            float x0 = x.data_[base + i];
            float x1 = x.data_[base + i + 1];
            // 将两点进行旋转
            x.data_[base + i] = x0 * cos_val - x1 * sin_val;
            x.data_[base + i + 1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

static void ApplyRopeNeoXToHeads(Tensor& x, int pos, float theta) {
    int n_heads = x.shape_[0];
    int head_dim = x.shape_[1];
    int half_dim = head_dim / 2;
    for (int h = 0; h < n_heads; ++h) {
        int base = h * head_dim;
        for (int i = 0; i < half_dim; ++i) {
            float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) /
                                            static_cast<float>(head_dim));
            float cos_val = std::cos(pos * freq);
            float sin_val = std::sin(pos * freq);
            float x0 = x.data_[base + i];
            float x1 = x.data_[base + half_dim + i];
            // 将两点进行旋转
            x.data_[base + i] = x0 * cos_val - x1 * sin_val;
            x.data_[base + half_dim + i] = x0 * sin_val + x1 * cos_val;
        }
    }
}

void Rope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type) {
    // q: [n_heads, head_dim], k: [n_kv_heads, head_dim]
    if (q.num_dims() != 2 || k.num_dims() != 2) {
        throw std::runtime_error("Rope: expected 2D tensors");
    }
    if (pos < 0) {
        throw std::out_of_range("Rope: position must be non-negative");
    }
    if (!std::isfinite(theta) || theta <= 0.0f) {
        throw std::runtime_error("Rope: theta must be finite and positive");
    }

    int head_dim = q.shape_[1];
    int k_head_dim = k.shape_[1];
    if (head_dim != k_head_dim) {
        throw std::runtime_error("Rope: q and k head_dim mismatch " +
                                q.ShapeStringShort() + " vs " +
                                k.ShapeStringShort());
    }
    if (head_dim <= 0 || head_dim % 2 != 0) {
        throw std::runtime_error("Rope: head_dim must be positive and even, got " +
                                std::to_string(head_dim));
    }
    if (rope_type == RopeType::kNormal) {
        ApplyRopeNormalToHeads(q, pos, theta);
        ApplyRopeNormalToHeads(k, pos, theta);
    } else if (rope_type == RopeType::kNeoX) {
        ApplyRopeNeoXToHeads(q, pos, theta);
        ApplyRopeNeoXToHeads(k, pos, theta);
    }
}

int ArgMax(const Tensor& x) {
    if (x.size() == 0) {
        throw std::runtime_error("ArgMax: cannot compute ArgMax of empty tensor");
    }
    int best = 0;
    float best_val = x.data_[0];
    for (size_t i = 1; i < x.data_.size(); ++i) {
        if (x.data_[i] > best_val) {
            best_val = x.data_[i];
            best = static_cast<int>(i);
        }
    }
    return best;
}

}