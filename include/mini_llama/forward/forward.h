#ifndef INCLUDE_MINI_LLAMA_FORWARD_H_
#define INCLUDE_MINI_LLAMA_FORWARD_H_

#include "mini_llama/tensor_and_Ops/tensor.h"
#include "mini_llama/runtime/context.h"
#include "mini_llama/forward/batch.h"

namespace mini_llama {

Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    int token);

Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const MiniBatch& batch);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_FORWARD_H_