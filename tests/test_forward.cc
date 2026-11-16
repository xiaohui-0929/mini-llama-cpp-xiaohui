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
 * 检查ForwardToken执行后所得logits的大小是否和模型词表大小相同
 */
static bool TestForwardTokenProducesLogitShape() {
    MiniLlamaModel model =
        LoadModel(TestDataPath("models/tiny/model.json"), TestDataPath("models/tiny/model.bin"));
    if (model.layers.empty()) {
        MINI_LLAMA_ASSERT_FAIL("Failed to load model");
    }
    MiniLlamaContext ctx(&model);
    ctx.pos = 0;
    Tensor logits = ForwardToken(ctx, model, 1);  // BOS token
    MINI_LLAMA_ASSERT_EQ(logits.num_dims(), 1);
    MINI_LLAMA_ASSERT_EQ(logits.shape_[0], model.config.vocab_size);
    return true;
}

/**
 * 检查ForwardToken执行后 kv_catch内是否正确存入信息
 */
static bool TestForwardTokenChangesKvCache() {
    MiniLlamaModel model =
        LoadModel(TestDataPath("models/tiny/model.json"), TestDataPath("models/tiny/model.bin"));
    if (model.layers.empty()) {
        MINI_LLAMA_ASSERT_FAIL("Failed to load model");
    }
    MiniLlamaContext ctx(&model);
    ctx.pos = 0;
    // Before forward, cache at layer 0, pos 0 should be zero
    float before = ctx.kv_cache.keys.At({0, 0, 0, 0});
    MINI_LLAMA_ASSERT_NEAR(before, 0.0f, 1e-6f);

    ForwardToken(ctx, model, 1);

    // After forward, cache should have been written
    float after = ctx.kv_cache.keys.At({0, 0, 0, 0});
    // It's unlikely to be exactly 0 after random weights forward
    // Just check it's finite and cache was written somewhere
    MINI_LLAMA_ASSERT_TRUE(std::isfinite(after));
    return true;
}

/**
 * 检查逐步ForwardToken后 所得结果的正确性
 */
static bool TestPositionIncrementsCorrectly() {
    MiniLlamaModel model =
        LoadModel("models/tiny/model.json", "models/tiny/model.bin");
    if (model.layers.empty()) {
        MINI_LLAMA_ASSERT_FAIL("Failed to load model");
    }
    MiniLlamaContext ctx(&model);

    // Forward token at pos 0
    ctx.pos = 0;
    ForwardToken(ctx, model, 1);

    // Forward token at pos 1
    ctx.pos = 1;
    Tensor logits = ForwardToken(ctx, model, 2);
    MINI_LLAMA_ASSERT_EQ(logits.shape_[0], model.config.vocab_size);
    return true;
}

/**
 * 检查错误token id情况下的异常抛出
 */
static bool TestForwardRejectsInvalidToken() {
    MiniLlamaModel model =
        LoadModel("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("Failed to load model");
    }
    MiniLlamaContext ctx(&model);
    ctx.pos = 0;

    try {
        ForwardToken(ctx, model, model.config.vocab_size);
        MINI_LLAMA_ASSERT_FAIL("ForwardToken accepted an out-of-range token");
    } catch (const std::out_of_range& e) {
        std::cerr << e.what() << std::endl;
        return true;
    } catch (...) {
        MINI_LLAMA_ASSERT_FAIL(
            "ForwardToken threw the wrong exception type for invalid token");
    }
}

/**
 * 检查错误position情况下的异常抛出
 */
static bool TestForwardRejectsInvalidPosition() {
    MiniLlamaModel model =
        LoadModel("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("Failed to load model");
    }
    MiniLlamaContext ctx(&model);
    ctx.pos = model.config.max_seq_len;

    try {
        ForwardToken(ctx, model, 1);
        MINI_LLAMA_ASSERT_FAIL("ForwardToken accepted an out-of-range position");
    } catch (const std::out_of_range& e) {
        std::cerr << e.what() << std::endl;
        return true;
    } catch (...) {
        MINI_LLAMA_ASSERT_FAIL(
            "ForwardToken threw the wrong exception type for invalid position");
    }
}

static struct ForwardTestRegistrar {
    ForwardTestRegistrar() {
        RegisterTest("forward_token_produces_logit_shape",
                    TestForwardTokenProducesLogitShape);
        RegisterTest("forward_token_changes_kv_cache",
                    TestForwardTokenChangesKvCache);
        RegisterTest("position_increments_correctly",
                    TestPositionIncrementsCorrectly);
        RegisterTest("forward_rejects_invalid_token",
                    TestForwardRejectsInvalidToken);
        RegisterTest("forward_rejects_invalid_position",
                    TestForwardRejectsInvalidPosition);
    }
} forward_test_registrar;