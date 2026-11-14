#include "test_main.h"
#include "test_names.h"

#ifndef MINI_LLAMA_SOURCE_DIR
#define MINI_LLAMA_SOURCE_DIR "."
#endif

namespace {

static std::string TestDataPath(const std::string& relative) {
    return std::string(MINI_LLAMA_SOURCE_DIR) + "/" + relative;
}

}   // namespace

/**
 * 检查MiniBatch的Single接口
 */
static bool TestBatchSingle() {
    MiniBatch b = MiniBatch::Single(42, 5);
    MINI_LLAMA_ASSERT_EQ(b.num_tokens(), 1);
    MINI_LLAMA_ASSERT_EQ(b.tokens[0], 42);
    MINI_LLAMA_ASSERT_EQ(b.positions[0], 5);
    return true;
}

/**
 * 检查MiniBatch的FromTokens接口
 */
static bool TestBatchFromTokens() {
    // 检查无偏移情况
    std::vector<int> toks_1 = {10, 20, 30};
    MiniBatch a = MiniBatch::FromTokens(toks_1, 0);
    MINI_LLAMA_ASSERT_EQ(a.num_tokens(), 3);
    MINI_LLAMA_ASSERT_EQ(a.tokens[0], 10);
    MINI_LLAMA_ASSERT_EQ(a.tokens[1], 20);
    MINI_LLAMA_ASSERT_EQ(a.tokens[2], 30);
    MINI_LLAMA_ASSERT_EQ(a.positions[0], 0);
    MINI_LLAMA_ASSERT_EQ(a.positions[1], 1);
    MINI_LLAMA_ASSERT_EQ(a.positions[2], 2);

    // 检查存在偏移情况
    std::vector<int> toks_2 = {5, 6};
    MiniBatch b = MiniBatch::FromTokens(toks_2, 10);
    MINI_LLAMA_ASSERT_EQ(b.positions[0], 10);
    MINI_LLAMA_ASSERT_EQ(b.positions[1], 11);

    return true;
}

/**
 * 检查空batch的情况
 */
static bool TestBatchEmpty() {
    MiniBatch b;
    MINI_LLAMA_ASSERT_EQ(b.num_tokens(), 0);
    return true;
}

/**
 * 检查统一批token 单独按序执行 和 通过batch执行 结果是否一致
 */
static bool TestForwardBatchPrefillMatchesIndividual() {
    // 加载model
    MiniLlamaModel model =
        LoadModel(TestDataPath("models/tiny/model.json"), TestDataPath("models/tiny/model.bin"));
    if (!model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("failed to load model");
    }

    std::vector<int> tokens = {1, 104, 101, 108, 108, 111};

    // 单token按序执行forward
    MiniLlamaContext ctx1(&model);
    Tensor logits1;
    for (size_t i = 0; i < tokens.size(); ++i) {
        ctx1.pos = static_cast<int>(i);
        logits1 = ForwardToken(ctx1, model, tokens[i]);
    }

    // 相同token通过ForwardBatch接口执行forward
    MiniLlamaContext ctx2(&model);
    MiniBatch batch = MiniBatch::FromTokens(tokens, 0);
    Tensor logits2 = ForwardBatch(ctx2, model, batch);

    // 判断两种场景下的结果是否有区别
    MINI_LLAMA_ASSERT_EQ(logits1.num_dims(), 1);
    MINI_LLAMA_ASSERT_EQ(logits2.num_dims(), 1);
    MINI_LLAMA_ASSERT_EQ(logits1.shape_[0], logits2.shape_[0]);
    for (size_t i = 0; i < logits1.size(); ++i) {
        MINI_LLAMA_ASSERT_NEAR(logits1.data_[i], logits2.data_[i], 1e-6f);
    }
    return true;
}

/**
 * 检查单一token 单独执行 和 通过batch执行 结果是否一致
 */
static bool TestForwardBatchSingleMatchesIndividual() {
    MiniLlamaModel model =
        LoadModel(TestDataPath("models/tiny/model.json"), TestDataPath("models/tiny/model.bin"));
    if (!model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("failed to load model");
    }

    // Individual forward
    MiniLlamaContext ctx1(&model);
    ctx1.pos = 3;
    Tensor logits1 = ForwardToken(ctx1, model, 42);

    // Batch forward with single token
    MiniLlamaContext ctx2(&model);
    MiniBatch batch = MiniBatch::Single(42, 3);
    Tensor logits2 = ForwardBatch(ctx2, model, batch);

    MINI_LLAMA_ASSERT_EQ(logits1.num_dims(), 1);
    MINI_LLAMA_ASSERT_EQ(logits2.num_dims(), 1);
    MINI_LLAMA_ASSERT_EQ(logits1.shape_[0], logits2.shape_[0]);
    for (size_t i = 0; i < logits1.size(); ++i) {
        MINI_LLAMA_ASSERT_NEAR(logits1.data_[i], logits2.data_[i], 1e-6f);
    }
    return true;
}

/**
 * 检查经过ForwardBatch接口后 推理上下文MiniLlamaContext的信息是否正确更新
 */
static bool TestForwardBatchUpdatesContext() {
    MiniLlamaModel model =
        LoadModel(TestDataPath("models/tiny/model.json"), TestDataPath("models/tiny/model.bin"));
    if (!model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("failed to load model");
    }

    std::vector<int> tokens = {1, 104, 101};
    MiniLlamaContext ctx(&model);
    MiniBatch batch = MiniBatch::FromTokens(tokens, 0);
    ForwardBatch(ctx, model, batch);

    MINI_LLAMA_ASSERT_EQ(ctx.token_history.size(), tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        MINI_LLAMA_ASSERT_EQ(ctx.token_history[i], tokens[i]);
    }
    MINI_LLAMA_ASSERT_EQ(ctx.pos, static_cast<int>(tokens.size()) - 1);
    return true;
}

/**
 * 模拟 prefill 和 decode 两个阶段 
 * 检查推理上下文MiniLlamaContext的信息是否正确
 */
static bool TestForwardBatchAppendsDecodeHistory() {
    MiniLlamaModel model =
        LoadModel(TestDataPath("models/tiny/model.json"), TestDataPath("models/tiny/model.bin"));
    if (!model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("failed to load model");
    }

    MiniLlamaContext ctx(&model);
    MiniBatch prefill = MiniBatch::FromTokens({1, 104}, 0);
    ForwardBatch(ctx, model, prefill);
    // prefill后的检查
    MINI_LLAMA_ASSERT_EQ(ctx.token_history.size(), 2);
    MINI_LLAMA_ASSERT_EQ(ctx.token_history[0], 1);
    MINI_LLAMA_ASSERT_EQ(ctx.token_history[1], 104);
    MINI_LLAMA_ASSERT_EQ(ctx.pos, 1);

    MiniBatch decode = MiniBatch::Single(101, 2);
    ForwardBatch(ctx, model, decode);
    // decode后的检查
    MINI_LLAMA_ASSERT_EQ(ctx.token_history.size(), 3);
    MINI_LLAMA_ASSERT_EQ(ctx.token_history[0], 1);
    MINI_LLAMA_ASSERT_EQ(ctx.token_history[1], 104);
    MINI_LLAMA_ASSERT_EQ(ctx.token_history[2], 101);
    MINI_LLAMA_ASSERT_EQ(ctx.pos, 2);
    return true;
}

/**
 * 检查batch为空时 是否正确抛出异常
 */
static bool TestForwardBatchEmptyRejected() {
    MiniLlamaModel model =
        LoadModel(TestDataPath("models/tiny/model.json"), TestDataPath("models/tiny/model.bin"));
    if (!model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("failed to load model");
    }

    MiniLlamaContext ctx(&model);
    MiniBatch batch;
    try {
        ForwardBatch(ctx, model, batch);
        MINI_LLAMA_ASSERT_FAIL("expected exception for empty batch");
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }
    return true;
}

/**
 * 检查token数量和position位置数量不一致时 是否正确抛出异常
 */
static bool TestForwardBatchMismatchedSizesRejected() {
    MiniLlamaModel model =
        LoadModel(TestDataPath("models/tiny/model.json"), TestDataPath("models/tiny/model.bin"));
    if (!model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("failed to load model");
    }

    MiniLlamaContext ctx(&model);
    MiniBatch batch;
    batch.tokens = {1, 2};
    batch.positions = {0};
    try {
        ForwardBatch(ctx, model, batch);
        MINI_LLAMA_ASSERT_FAIL("expected exception for size mismatch");
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }
    return true;
}

static struct BatchTestRegistrar {
    BatchTestRegistrar() {
        RegisterTest("batch_single", 
                    TestBatchSingle);
        RegisterTest("batch_from_tokens", 
                    TestBatchFromTokens);
        RegisterTest("batch_empty", 
                    TestBatchEmpty);
        RegisterTest("forward_batch_prefill_matches_individual",
                    TestForwardBatchPrefillMatchesIndividual);
        RegisterTest("forward_batch_single_matches_individual",
                    TestForwardBatchSingleMatchesIndividual);
        RegisterTest("forward_batch_updates_context",
                    TestForwardBatchUpdatesContext);
        RegisterTest("forward_batch_appends_decode_history",
                    TestForwardBatchAppendsDecodeHistory);
        RegisterTest("forward_batch_empty_rejected", 
                    TestForwardBatchEmptyRejected);
        RegisterTest("forward_batch_mismatched_sizes_rejected",
                    TestForwardBatchMismatchedSizesRejected);
    }

} batch_test_registrar;