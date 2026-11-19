#ifndef INCLUDE_MINI_LLAMA_CHAT_H_
#define INCLUDE_MINI_LLAMA_CHAT_H_

#include <string>
#include <vector>

#include "mini_llama/runtime/radix_tree.h"
#include "mini_llama/sampler/sampler.h"

namespace mini_llama {

/**
 * 每次对话的记录信息
 */
struct ChatMessage {
    std::string role;       // 对话角色 "system", "user", "assistent"
    std::string content;    // 具体对话内容

    ChatMessage(const std::string r, const std::string c) : role(r), content(c) {}
};

/**
 * 多轮对话信息的管理
 */
class ChatSession {
public:
    std::vector<ChatMessage> messages;  // 多轮对话信息
    std::vector<int> token_history;     // 历史token
    RadixTree prefix_cache;             // 前缀缓存
    SamplingParams sampling_params;     // 采样参数

    int total_prompt_tokens = 0;        // 所有提示词token数
    int total_generated_tokens = 0;    // 所有生成token数
    double total_time_ms = 0.0;         // 对话耗时

public:
    ChatSession() = default;

    /**
     * 清空历史信息并重置
     */
    void Clear();

    /**
     * 增加一条对话信息
     */
    void AddMessage(const std::string& role, const std::string& content);

    /**
     * 一轮对话后记录统计值
     */
    void RecordTurn(int prompt_tokens, int generated_tokens, double time_ms);

    /**
     * 将历史token替换为入参tokens
     */
    void SetTokenHistory(const std::vector<int>& tokens);

    /**
     * 将入参token添加进入历史token内
     */
    void AppendToken(int token);

    /**
     * 将入参tokens的前缀记录进入前缀缓存中(RadixTree)
     */
    void RecordPrefix(const std::vector<int>& tokens);

    /**
     * 获取入参tokens的最长前缀数
     */
    size_t LongestCachedPrefix(const std::vector<int>& tokens) const;
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CHAT_H_