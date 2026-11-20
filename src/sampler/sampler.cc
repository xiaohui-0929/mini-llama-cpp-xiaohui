#include "mini_llama/sampler/sampler.h"

#include <algorithm>

#include "mini_llama/tensor_and_Ops/ops.h"


namespace mini_llama {
    
namespace {

/**
 * temperature最小值 低于该值则策略退化为Greedy
 */
constexpr float kGreedyTemperature = 1e-6f;

void ValidateLogits(const Tensor& logits) {
    if (logits.size() == 0) {
        throw std::runtime_error("sampler: cannot sample from empty logits");
    }
    // 检查每个位置数值的有效性
    for (const auto& value : logits.data) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("sampler: logits must be finite");
        }
    }
}

void ValidateTemperature(float temperature) {
    if (!std::isfinite(temperature) || temperature < 0.0f) {
        throw std::runtime_error( "sampler: temperature must be finite and non-negative");
    }
}

void ValidateTopK(int top_k) {
    if (top_k < 0) {
        throw std::runtime_error("sampler: top_k must be non-negative");
    }
}

int SampleFromCandidates(const std::vector<std::pair<float, int>>& candidates,
                         float temperature, std::mt19937& rng) {
    if (candidates.size() <= 0) {
        throw std::runtime_error( "sampler: candidates must have value");
    }
    // 防止传入的temperature过小 过小则采用Greedy采样
    if (temperature < kGreedyTemperature) {
        int result = candidates[0].second;
        float max = candidates[0].first;
        for (const auto& value : candidates) {
            if (value.first > max) {
                result = value.second;
                max = value.first;
            }
        }
        return result;
    }

    // 对candidates内的数据进行temperature调节
    // 先找出最大值
    float max = candidates[0].first;
    for (const auto& value : candidates) {
        if (value.first > max) {
            max = value.first;
        }
    }
    // 在转换为概率分布的过程中 引入temperature调整平滑度
    std::vector<float> probs(candidates.size(), 0.0f);
    float sum = 0.0f;
    for (size_t i = 0; i < candidates.size(); ++i) {
        probs[i] = std::exp((candidates[i].first - max) / temperature);
        sum += probs[i];
    }
    // 确定0~1的均匀分布范围
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cumulative = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i] / sum;
        if (r <= cumulative) {
            return candidates[i].second;
        }
    }
    return candidates.back().second;
}

}   // namespace

/**
 * 构造
 */
MiniSampler::MiniSampler(unsigned int seed) {
    if (seed != 0) {
        rng_.seed(seed);
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }
}

/**
 * 构造
 */
MiniSampler::MiniSampler(const SamplingParams& params) 
    : MiniSampler(params.seed) {}

/**
 * 贪婪采样 选取得分最高的候选词
 */
int MiniSampler::SampleGreedy(const Tensor& logits) {
    // 先检查logits有效性
    ValidateLogits(logits);
    // 再进行采样
    return ArgMax(logits);
}

/**
 * Temperature采样 
 * temperature调节候选词的平滑度 影响每个候选词的概率权重分布
 * 进而影响累积概率区间随机取样时的结果
 */
int MiniSampler::SampleTemperature(const Tensor& logits, float temperature) {
    // 检查logits有效性
    ValidateLogits(logits);
    // 检查temperature有效性
    ValidateTemperature(temperature);
    if (temperature < kGreedyTemperature) {
        SampleGreedy(logits);
    }

    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        candidates.emplace_back(logits.data[i], static_cast<int>(i));
    }
    return SampleFromCandidates(candidates, temperature, rng_);
}

/**
 * Top-K采样 截断指定数量的候选词
 */
int MiniSampler::SampleTopK(const Tensor& logits, float temperature, int top_k) {
    // 检查logits有效性
    ValidateLogits(logits);
    // 检查temperature有效性
    ValidateTemperature(temperature);
    // 检查top_k有效性
    ValidateTopK(top_k);
    if (top_k == 0) {
        throw std::runtime_error( "sampler: top_k must be positive when calling SampleTopK");
    }
    if (top_k == 1) {
        return SampleGreedy(logits);
    }
    if (top_k > static_cast<int>(logits.size())) {
        top_k = static_cast<int>(logits.size());
    }
    // 获取candidates
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        candidates.emplace_back(logits.data[i], static_cast<int>(i));
    }
    // sort是不稳定排序 候选词相等的情况下需要维持原本的相对位置不变
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) {
                if (a.first == b.first) {
                    return a.second < b.second;
                } else {
                    return a.first > b.first;
                }
              });
    // 按照top_k进行截断
    candidates.resize(static_cast<size_t>(top_k));

    return SampleFromCandidates(candidates, temperature, rng_);
}

/**
 * 统一对外接口 内部执行调度
 * 使用提供的params对logits进行采样
 */
int MiniSampler::Sample(const Tensor& logits, const SamplingParams& params) {
    ValidateTemperature(params.temperature);
    ValidateTopK(params.top_k);
    // 使用Greedy
    if (params.temperature < kGreedyTemperature || params.top_k == 1) {
        return SampleGreedy(logits);
    }
    // 使用top_k
    if (params.top_k > 1) {
        return SampleTopK(logits, params.temperature, params.top_k);
    }
    // 使用temperature
    return SampleTemperature(logits, params.temperature);
}

}   // namespace mini_llama