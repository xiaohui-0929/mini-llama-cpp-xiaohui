#include "mini_llama/forward/batch.h"

#include <stdexcept>

#include "mini_llama/runtime/context.h"
#include "mini_llama/forward/forward.h"

namespace mini_llama {

MiniBatch MiniBatch::Single(int token, int pos) {
    MiniBatch batch;
    batch.tokens.push_back(token);
    batch.positions.push_back(pos);
    return batch;
}

MiniBatch MiniBatch::FromTokens(const std::vector<int>& toks, int start_pos) {
    MiniBatch batch;

    batch.tokens = toks;
    batch.positions.reserve(toks.size());
    for (size_t i = 0; i < toks.size(); ++i) {
        batch.positions.push_back(static_cast<int>(i) + start_pos);
    }

    return batch;
}

/**
 * 对batch内的所有token执行 forward
 */
Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const MiniBatch& batch) {
    if (batch.num_tokens() == 0) {
        throw std::runtime_error("ForwardBatch: empty batch");
    }
    if (batch.tokens.size() != batch.positions.size()) {
        throw std::runtime_error(
            "ForwardBatch: tokens and positions size mismatch"
        );
    }

    // 先校验整个batch 再写入Context 避免中途失败后残留部分token的历史和KV cache
    const ModelConfig& config = model.config;
    for (int i = 0; i < batch.num_tokens(); ++i) {
        if (batch.tokens[i] < 0 || batch.tokens[i] >= config.vocab_size) {
            throw std::out_of_range("ForwardBatch: token id out of range");
        }
        if (batch.positions[i] < 0 || batch.positions[i] >= config.max_seq_len) {
            throw std::out_of_range("ForwardBatch: position out of range");
        }
    }

    Tensor logits;
    for (int i = 0; i < batch.num_tokens(); ++i) {
        ctx.pos = batch.positions[i];
        logits = ForwardToken(ctx, model, batch.tokens[i]);
        ctx.token_history.push_back(batch.tokens[i]);
    }
    return logits;
}

}   // namespace mini_llama