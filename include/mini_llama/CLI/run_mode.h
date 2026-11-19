#ifndef INCLUDE_MINI_LLAMA_RUN_MODE_H_
#define INCLUDE_MINI_LLAMA_RUN_MODE_H_

/**
 * run mode 交换式对话
 * 暂不考虑backend
 */
namespace mini_llama {

/**
 * run mode的提示信息
 */    
void PrintRunUsage(const char* prog);

/**
 * CLI run <model-path|dir> [options]
 */
int RunChat(int argc, char** argv);

}   // mini_llama

#endif  // INCLUDE_MINI_LLAMA_RUN_MODE_H_