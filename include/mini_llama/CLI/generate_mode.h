#ifndef INCLUDE_MINI_LLAMA_GENERATE_MODE_H_
#define INCLUDE_MINI_LLAMA_GENERATE_MODE_H_

namespace mini_llama {

void PrintGenerateUsage(const char* prog);

int RunGenerate(int argc, char** argv);

}   // namespace mini_llama

#endif