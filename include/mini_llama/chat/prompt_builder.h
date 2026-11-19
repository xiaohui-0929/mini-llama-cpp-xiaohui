#ifndef INCLUDE_MINI_LLAMA_PROMPT_BUILDER_H_
#define INCLUDE_MINI_LLAMA_PROMPT_BUILDER_H_

#include "mini_llama/chat/chat.h"

namespace mini_llama {

/**
 * prompt生成器
 * 负责将 结构化的对话信息 组装为 指定格式的prompt
 */
class PromptBuilder {
public:
    PromptBuilder() = default;

    /**
     * 设置指定格式的模板 若为空 则使用 plain模式
     */
    void SetChatTemplate(const std::string& template_str);

    /**
     * 通过入参messages 组装 prompt
     */
    std::string Build(const std::vector<ChatMessage>& messages) const;

private:
    std::string chat_template_; // prompt格式模板

private:
    /**
     * 转为plain模式的prompt
     * 本地tiny模型验证
     */
    std::string BuildPlain(const std::vector<ChatMessage>& messages) const;
    /**
     * 转为Qwen模式的prompt
     * 每条消息的格式: <|im_start|>{role}\n{content}<|im_end|>\n
     * 最后的assistant prompt: <|im_start|>assistant\n
     */
    std::string BuildQwen2(const std::vector<ChatMessage>& messages) const;
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_PROMPT_BUILDER_H_