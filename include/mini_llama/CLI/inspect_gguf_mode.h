#ifndef INCLUDE_MINI_LLAMA_INSPECT_GGUF_MODE_H_
#define INCLUDE_MINI_LLAMA_INSPECT_GGUF_MODE_H_

#include "mini_llama/gguf/gguf.h"

namespace mini_llama {

/**
 * Gguf格式文件快照
 */
void InspectGguf(const GgufReader& reader);

int RunInspectGguf(int argc, char** argv);

}   // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_INSPECT_GGUF_MODE_H_