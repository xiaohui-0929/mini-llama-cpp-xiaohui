#ifndef INCLUDE_MINI_LLAMA_OPS_H_
#define INCLUDE_MINI_LLAMA_OPS_H_

#include "mini_llama/tensor_and_Ops/tensor.h"
#include "mini_llama/model/model.h"

namespace mini_llama {

// matmul 矩阵乘法
Tensor Matmul(const Tensor& a, const Tensor& b);

// linear 仿射变换
Tensor Linear(const Tensor& x, const Tensor& weight);

// rmsNorm 均方根归一化 防止数值发散
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps);

// softMax 指数化、归一化获取概率分布
Tensor Softmax(const Tensor& x);

// SiLU 激活函数 Silu(x) = x * sigmoid(x)
Tensor Silu(const Tensor& x);

// ElementwiseMul 逐元素乘
Tensor ElementwiseMul(const Tensor& a, const Tensor& b);

// SwiGlu 激活函数 gate门控 可动态、连续地调节信息
// SwiGlu(gate, up) = Silu(gate) * up
Tensor SwiGlu(const Tensor& gate, const Tensor& up);

// RoPE 旋转位置编码 通过旋转操作将相对位置信息注入到注意力机制
void Rope(Tensor& q, Tensor& k, int pos, float theta,
          RopeType rope_type = RopeType::kNormal);

// ArgMax 获取最大值下标
int ArgMax(const Tensor& x);

}

#endif  // INCLUDE_MINI_LLAMA_OPS_H_