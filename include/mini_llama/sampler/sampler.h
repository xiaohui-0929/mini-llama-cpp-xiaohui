#ifndef INCLUDE_MINI_LLAMA_SAMPLER_H_
#define INCLUDE_MINI_LLAMA_SAMPLER_H_

#include <random>

#include "mini_llama/tensor_and_Ops/tensor.h"

namespace mini_llama {

/**
 * 采样参数
 */
struct SamplingParams {
    float temperature = 0.0;    // 调节概率分布的平滑度
    int top_k = 0;              // 截断策略
    unsigned int seed = 0;      // 随机种子
};

class MiniSampler {
public:
    explicit MiniSampler(unsigned int seed = 0);
    explicit MiniSampler(const SamplingParams& params);

    /**
     * 统一对外接口 内部执行调度
     * 使用提供的params对logits进行采样
     */
    int Sample(const Tensor& logits, const SamplingParams& params);

    /**
     * 贪婪采样 选取得分最高的候选词
     */
    int SampleGreedy(const Tensor& logits);

    /**
     * Temperature采样 
     * temperature调节候选词的平滑度 影响每个候选词的概率权重分布
     * 进而影响累积概率区间随机取样时的结果
     */
    int SampleTemperature(const Tensor& logits, float temperature);

    /**
     * Top-K采样 截断指定数量的候选词
     */
    int SampleTopK(const Tensor& logits, float temperature, int top_k);

private:
    std::mt19937 rng_;
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_SAMPLER_H_