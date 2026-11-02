#ifndef INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
#define INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_

#include "mini_llama/tensor_and_Ops/tensor.h"

namespace mini_llama {

// 调度枚举 
enum class MatmulMode {
    kNaive,         // 基本操作 三层循环 正确性基线
    kThreaded,      // 多线程按行切分
    kSimd,          // 单线程 SIMD 点积（仅 Linear 有效）
    kThreadedSimd,  // 多线程 × SIMD
};

// 默认调度
MatmulMode DefaultMatmulMode();

// 矩阵乘法 分发调度
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode);

// 仿射变换 分发调度
Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode);

}   // mini_llama

#endif  // INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_