#ifndef INCLUDE_MINI_LLAMA_MODEL_H_
#define INCLUDE_MINI_LLAMA_MODEL_H_

namespace mini_llama {

enum class RopeType {
    kNormal,    // LLaMA风格
    kNeoX,      // Qwen风格
};

}   // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_MODEL_H_