
#include "mini_llama/tensor_and_Ops/matmul_dispatch.h"

#include <stdexcept>

#include "mini_llama/tensor_and_Ops/thread_pool.h"

namespace mini_llama {

MatmulMode DefaultMatmulMode() { return MatmulMode::kThreadedSimd; }

/**
 * Native implementations
 */

static Tensor MatmulNaive(const Tensor& a, const Tensor& b) {
    int m = a.shape_[0];
    int inner_dim = a.shape_[1];
    int n = b.shape_[1];
    Tensor c({m, n}, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < inner_dim; ++k) {
                sum += a.data_[i * inner_dim + k] * b.data_[k * n + j];
            }
            c.data_[i * n + j] = sum;
        }
    }
    return c;
}

static Tensor LinearNaive(const Tensor& x, const Tensor& weight) {
    int in_features = weight.shape_[1];
    int out_features = weight.shape_[0];
    bool is_1d = (x.num_dims() == 1);

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{1, out_features},
                  0.0f);
    for (int j = 0; j < out_features; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += x.data_[k] * weight.data_[j * in_features + k];
        }
        result.data_[j] = sum;
    }

    return result;
}

/**
 * Threaded implementations
 */

static Tensor MatmulThreaded(const Tensor& a, const Tensor& b) {
    int m = a.shape_[0];
    int inner_dim = a.shape_[1];
    int n = b.shape_[1];
    Tensor c({m, n}, 0.0f);
    ParallelFor(m, [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < inner_dim; ++k) {
                    sum += a.data_[i * inner_dim + k] * b.data_[k * n + j];
                }
                c.data_[i * n + j] = sum;
            }
        }
    });
    return c;
}

static Tensor LinearThreaded(const Tensor& x, const Tensor& weight) {
    int in_features = weight.shape_[1];
    int out_features = weight.shape_[0];
    bool is_1d = (x.num_dims() == 1);

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{1, out_features},
                  0.0f);

    ParallelFor(out_features, [&](int begin, int end) {
        for (int j = begin; j < end; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < in_features; ++k) {
                sum += x.data_[k] * weight.data_[j * in_features + k];
            }
            result.data_[j] = sum;
        }
    });

    return result;
}

/**
 * SIMD dot-product helpers
 */

static float DotSimd(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

/**
 * SIMD implementations
 */

static Tensor LinearSimd(const Tensor& x, const Tensor& weight) {
    int in_features = weight.shape_[1];
    int out_features = weight.shape_[0];
    bool is_1d = (x.num_dims() == 1);

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{1, out_features},
                  0.0f);
    
    for (int j = 0; j < out_features; ++j) {
        result.data_[j] = DotSimd(x.data_.data(),
                                 weight.data_.data() + j * in_features, in_features);
    }

    return result;
}

static Tensor LinearThreadedSimd(const Tensor& x, const Tensor& weight) {
    int in_features = weight.shape_[1];
    int out_features = weight.shape_[0];
    bool is_1d = (x.num_dims() == 1);

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{1, out_features},
                  0.0f);

    ParallelFor(out_features, [&](int begin, int end) {
        for (int j = begin; j < end; ++j) {
            result.data_[j] = DotSimd(x.data_.data(),
                                      weight.data_.data() + j * in_features, in_features);
        }
    });

    return result;
}

/**
 * Dispatch
 */

Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode) {
    if (a.num_dims() != 2 || b.num_dims() != 2) {
        throw std::runtime_error("MatmulDispatch: expected 2D tensors");
    }
    if (a.shape_[1] != b.shape_[0]) {
        throw std::runtime_error("MatmulDispatch: dimension mismatch");
    }

    switch (mode) {
        case MatmulMode::kNaive:
            return MatmulNaive(a, b);
        case MatmulMode::kThreaded:
            return MatmulThreaded(a, b);
        case MatmulMode::kSimd:
        case MatmulMode::kThreadedSimd:
            return MatmulThreaded(a, b);
    }
    return MatmulThreaded(a, b);
}

Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode) {
    if (weight.num_dims() != 2) {
        throw std::runtime_error(
            "LinearDispatch: expected weight shape [out_features, in_features]");
    }
    int in_features;
    if (x.num_dims() == 1) {
        in_features = x.shape_[0];
    } else if (x.num_dims() == 2 && x.shape_[0] == 1) {
        in_features = x.shape_[1];
    } else {
        throw std::runtime_error(
            "LinearDispatch: expected x shape [in_features] or [1, in_features]");
    }
    if (weight.shape_[1] != in_features) {
        throw std::runtime_error("LinearDispatch: dimension mismatch");
    }

    switch (mode) {
        case MatmulMode::kNaive:
            return LinearNaive(x, weight);
        case MatmulMode::kThreaded:
            return LinearThreaded(x, weight);
        case MatmulMode::kSimd:
            return LinearSimd(x, weight);
        case MatmulMode::kThreadedSimd:
            return LinearThreadedSimd(x, weight);
    }
    return LinearThreadedSimd(x, weight);
}

}   // mini_llama
