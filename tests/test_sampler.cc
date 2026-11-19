#include "test_main.h"
#include "test_names.h"

/**
 * 检查基本的Greedy采样
 */
static bool TestSamplerGreedy() {
    Tensor logits({5}, 0.0f);
    logits[0] = 1.0f;
    logits[1] = 5.0f;
    logits[2] = 3.0f;
    logits[3] = 5.0f;  // same max as FlatIndex 1
    logits[4] = 2.0f;
    MiniSampler sampler;
    int idx = sampler.SampleGreedy(logits);
    MINI_LLAMA_ASSERT_EQ(idx, 1);  // first occurrence of max
    return true;
}

/**
 * 检查当temperature为0时 是否退化为Greedy采样
 */
static bool TestSamplerTemperatureZeroIsGreedy() {
    Tensor logits({5}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = 2.0f;
    logits[2] = 1.0f;
    logits[3] = 0.5f;
    logits[4] = 0.2f;

    MiniSampler sampler(42);
    SamplingParams params;
    params.temperature = 0.0f;
    params.top_k = 0;

    // Run many times; should always pick the same token (greedy).
    int expected = sampler.SampleGreedy(logits);
    for (int i = 0; i < 50; ++i) {
        int idx = sampler.Sample(logits, params);
        MINI_LLAMA_ASSERT_EQ(idx, expected);
    }
    return true;
}

/**
 * 检查当top_k为1时 是否退化为Greedy采样
 */
static bool TestSamplerTopKOneIsGreedy() {
    Tensor logits({5}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = 2.0f;
    logits[2] = 1.0f;
    logits[3] = 0.5f;
    logits[4] = 0.2f;

    MiniSampler sampler(42);
    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 1;

    int expected = sampler.SampleGreedy(logits);
    for (int i = 0; i < 20; ++i) {
        int idx = sampler.Sample(logits, params);
        MINI_LLAMA_ASSERT_EQ(idx, expected);
    }
    return true;
}

/**
 * 检查随机种子相等时 temperature采样结果是否一致、是否可复现
 */
static bool TestSamplerTemperatureReproducible() {
    Tensor logits({10}, 0.0f);
    for (int i = 0; i < 10; ++i) {
        logits[i] = static_cast<float>(i) * 0.5f;
    }

    MiniSampler sampler1(123);
    MiniSampler sampler2(123);
    SamplingParams params;
    params.temperature = 0.8f; // temperature越大 每个值被选中的概率越平均
    params.top_k = 0;

    // Same seed should produce identical sequences.
    for (int i = 0; i < 10; ++i) {
        int a = sampler1.Sample(logits, params);
        int b = sampler2.Sample(logits, params);
        MINI_LLAMA_ASSERT_EQ(a, b);
    }
    return true;
}

/**
 * 检查随机种子相等时 top_k采样结果是否一致、是否可复现
 */
static bool TestSamplerTopKReproducible() {
    Tensor logits({10}, 0.0f);
    for (int i = 0; i < 10; ++i) {
        logits[i] = static_cast<float>(i) * 0.5f;
    }

    MiniSampler sampler1(456);
    MiniSampler sampler2(456);
    SamplingParams params;
    params.temperature = 0.8f;
    params.top_k = 3;

    for (int i = 0; i < 10; ++i) {
        int a = sampler1.Sample(logits, params);
        int b = sampler2.Sample(logits, params);
        MINI_LLAMA_ASSERT_EQ(a, b);
    }
    return true;
}

/**
 * 检查不同随机种子情况下 每轮得到的结果是否随机
 */
static bool TestSamplerDifferentSeedsDiffer() {
    Tensor logits({10}, 0.0f);
    for (int i = 0; i < 10; ++i) {
        logits[i] = static_cast<float>(i) * 0.3f;
    }

    MiniSampler sampler1(1);
    MiniSampler sampler2(2);
    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 0;

    int same_count = 0;
    for (int i = 0; i < 20; ++i) {
        int a = sampler1.Sample(logits, params);
        int b = sampler2.Sample(logits, params);
        if (a == b) {
        ++same_count;
        }
    }

    // With different seeds and temperature=1.0, it's extremely unlikely
    // that all 20 samples match. Allow up to 18 matches as a loose bound.
    MINI_LLAMA_ASSERT_TRUE(same_count < 18);
    return true;
}

/**
 * 检查top_k采样下 每次所得候选者在截断范围内
 */
static bool TestSamplerTopKLimitsCandidates() {
    Tensor logits({10}, 0.0f);
    for (int i = 0; i < 10; ++i) {
        logits[i] = static_cast<float>(i);
    }

    MiniSampler sampler(789);
    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 3;

    // With top_k=3 on logits [0..9], only tokens 7, 8 and 9 should ever be sampled.
    for (int i = 0; i < 50; ++i) {
        int idx = sampler.Sample(logits, params);
        MINI_LLAMA_ASSERT_TRUE(idx == 8 || idx == 9 || idx == 7);
    }
    return true;
}

/**
 * 检查默认参数的情况下 执行Greedy采样
 */
static bool TestSamplerDefaultParamsAreGreedy() {
    Tensor logits({5}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = 2.0f;
    logits[2] = 1.0f;
    logits[3] = 0.5f;
    logits[4] = 0.2f;

    MiniSampler sampler(42);
    SamplingParams params;

    int idx = sampler.Sample(logits, params);
    MINI_LLAMA_ASSERT_EQ(idx, 1);
    MINI_LLAMA_ASSERT_EQ(idx, sampler.SampleGreedy(logits));
    return true;
}

/**
 * 检查top_k为负数情况下 是否抛出异常
 */
static bool TestSamplerRejectsNegativeTopK() {
    Tensor logits({3}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = 0.2f;
    logits[2] = 0.3f;

    MiniSampler sampler(42);
    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = -1;

    try {
        sampler.Sample(logits, params);
        MINI_LLAMA_ASSERT_FAIL("expected exception for negative top_k");
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return true;
    }
}

/**
 * 检查temperature为NaN情况下 是否抛出异常
 */
static bool TestSamplerRejectsNanTemperature() {
    Tensor logits({3}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = 0.2f;
    logits[2] = 0.3f;

    MiniSampler sampler(42);
    SamplingParams params;
    params.temperature = std::numeric_limits<float>::quiet_NaN();
    params.top_k = 0;

    try {
        sampler.Sample(logits, params);
        MINI_LLAMA_ASSERT_FAIL("expected exception for NaN temperature");
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return true;
    }
}

/**
 * 检查logits为空情况下 是否抛出异常
 */
static bool TestSamplerRejectsEmptyLogits() {
    Tensor logits;
    MiniSampler sampler(42);
    SamplingParams params;
    params.temperature = 0.0f;

    try {
        sampler.Sample(logits, params);
        MINI_LLAMA_ASSERT_FAIL("expected exception for empty logits");
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return true;
    }
}

/**
 * 检查logits内存在无限值情况下 是否抛出异常
 */
static bool TestSamplerRejectsInfinityLogits() {
    Tensor logits({3}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = std::numeric_limits<float>::infinity();
    logits[2] = 0.3f;

    MiniSampler sampler(42);
    SamplingParams params;
    params.temperature = 1.0f;

    try {
        sampler.Sample(logits, params);
        MINI_LLAMA_ASSERT_FAIL("expected exception for non-finite logits");
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return true;
    }
}

/**
 * 检查logits内存在NaN情况下 是否抛出异常
 */
static bool TestSamplerRejectsNaNLogits() {
    Tensor logits({3}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = std::numeric_limits<float>::quiet_NaN();
    logits[2] = 0.3f;

    MiniSampler sampler(42);
    SamplingParams params;
    params.temperature = 1.0f;

    try {
        sampler.Sample(logits, params);
        MINI_LLAMA_ASSERT_FAIL("expected exception for NaN logits");
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return true;
    }
}

/**
 * 检查top_k值超过logits数量时 是否正确按照logits数量截断
 */
static bool TestSamplerTopKLargerThanVocabIsClamped() {
    Tensor logits({4}, 0.0f);
    logits[0] = 0.0f;
    logits[1] = 1.0f;
    logits[2] = 2.0f;
    logits[3] = 3.0f;

    MiniSampler sampler(42);
    for (int i = 0; i < 20; ++i) {
        int idx = sampler.SampleTopK(logits, 1.0f, 100);
        MINI_LLAMA_ASSERT_TRUE(idx >= 0 && idx < 4);
    }
    return true;
}

/**
 * 检查top_k值为0时 直接调用SampleTopK抛出异常
 */
static bool TestSamplerDirectTopKZeroRejected() {
    Tensor logits({3}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = 0.2f;
    logits[2] = 0.3f;

    MiniSampler sampler(42);
    try {
        sampler.SampleTopK(logits, 1.0f, 0);
        MINI_LLAMA_ASSERT_FAIL("expected exception for direct top_k zero");
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return true;
    }
}

/**
 * 检查当存在多个相同值时 top_k的截断是否稳定、是否依然保持相对位置
 */
static bool TestSamplerTopKTiesKeepExactK() {
    Tensor logits({5}, 0.0f);
    logits[0] = 5.0f;
    logits[1] = 5.0f;
    logits[2] = 5.0f;
    logits[3] = 1.0f;
    logits[4] = 0.0f;

    MiniSampler sampler(321);
    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 2;

    for (int i = 0; i < 40; ++i) {
        int idx = sampler.Sample(logits, params);
        MINI_LLAMA_ASSERT_TRUE(idx == 0 || idx == 1);
    }
    return true;
}

static struct SamplerTestRegistrar {
    SamplerTestRegistrar() {
        RegisterTest("sampler_greedy", 
                    TestSamplerGreedy);
        RegisterTest("sampler_temperature_zero_is_greedy",
                    TestSamplerTemperatureZeroIsGreedy);
        RegisterTest("sampler_top_k_one_is_greedy", 
                    TestSamplerTopKOneIsGreedy);
        RegisterTest("sampler_temperature_reproducible",
                    TestSamplerTemperatureReproducible);
        RegisterTest("sampler_top_k_reproducible", 
                    TestSamplerTopKReproducible);
        RegisterTest("sampler_different_seeds_differ",
                    TestSamplerDifferentSeedsDiffer);
        RegisterTest("sampler_top_k_limits_candidates",
                    TestSamplerTopKLimitsCandidates);
        RegisterTest("sampler_default_params_are_greedy",
                    TestSamplerDefaultParamsAreGreedy);
        RegisterTest("sampler_rejects_negative_top_k",
                    TestSamplerRejectsNegativeTopK);
        RegisterTest("sampler_rejects_nan_temperature",
                    TestSamplerRejectsNanTemperature);
        RegisterTest("sampler_rejects_empty_logits", 
                    TestSamplerRejectsEmptyLogits);
        RegisterTest("sampler_rejects_infinity_logits",
                    TestSamplerRejectsInfinityLogits);
        RegisterTest("sampler_rejects_infinity_logits",
                    TestSamplerRejectsNaNLogits);
        RegisterTest("sampler_top_k_larger_than_vocab_is_clamped",
                    TestSamplerTopKLargerThanVocabIsClamped);
        RegisterTest("sampler_direct_top_k_zero_rejected",
                    TestSamplerDirectTopKZeroRejected);
        RegisterTest("sampler_top_k_ties_keep_exact_k",
                    TestSamplerTopKTiesKeepExactK);
    }
} sampler_test_registrar;