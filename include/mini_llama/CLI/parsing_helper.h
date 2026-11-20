#ifndef INCLUDE_MINI_LLAMA_PARSING_HELPER_H_
#define INCLUDE_MINI_LLAMA_PARSING_HELPER_H_

#include <memory>
#include <iostream>

#include <mini_llama/tokenizer/tokenizer.h>
#include <mini_llama/model/model.h>
#include <mini_llama/runtime/request_context.h>

namespace mini_llama {

struct LoadedModel {
    MiniLlamaModel model;
    std::unique_ptr<ITokenizer> tokenizer;
    std::string chat_template;
};

/**
 * 打印request_context的trace信息
 */
void PrintRequestTrace(const RequestContext& request, std::ostream& out = std::cout);

bool ParseIntArg(const char* text, int& value);

bool ParseFloatArg(const char* text, float& value);

bool ParseUintArg(const char* text, unsigned int& value);

/**
 * 尝试从config_path内获取tokenizer路径
 */
std::string ResolveManifestTokenizerPath(const std::string& config_path);

/**
 * 加载model 和 tokenizer
 * 先实现 JSON+BIN 格式读取
 * todo GUFF格式后续完善
 */
LoadedModel LoadModelAndTokenizer(
    const std::string& path, const std::string& explicit_config_path = "",
    const std::string& explicit_tokenizer_path = "");

}   // mini_llama

#endif  // INCLUDE_MINI_LLAMA_PARSING_HELPER_H_