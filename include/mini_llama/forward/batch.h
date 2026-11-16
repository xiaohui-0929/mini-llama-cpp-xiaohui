#ifndef INCLUDE_MINI_LLAMA_BATCH_H_
#define INCLUDE_MINI_LLAMA_BATCH_H_

#include <vector>

namespace mini_llama {

// 教学目的 实际生产环境中未必一一对应
struct MiniBatch {
    std::vector<int> tokens;    // token ids
    std::vector<int> positions; // 每个token对应的位置

    int num_tokens() const { return static_cast<int>(tokens.size()); }

    // 创建单token batch
    static MiniBatch Single(int token, int pos);

    // 创建多token batch
    static MiniBatch FromTokens(const std::vector<int>& toks, int start_pos = 0);
};

}   // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_BATCH_H_