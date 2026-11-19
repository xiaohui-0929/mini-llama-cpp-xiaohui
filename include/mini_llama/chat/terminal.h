#ifndef INCLUDE_MINI_LLAMA_TERMINAL_H_
#define INCLUDE_MINI_LLAMA_TERMINAL_H_

#include <string>

#include "mini_llama/chat/chat.h"

namespace mini_llama {

/**
 * 对命令行信息输入输出(std::cin std::cout)的封装
 */
class Terminal {
public:
    Terminal() = default;

    /**
     * 从cin中读取一行信息
     */
    std::string ReadLine() const;
    
    /**
     * 打印UserPrompt (e.g., "mini-llama > ")
     */
    void PrintUserPrompt() const;

    /**
     * 打印assistant (e.g., "assistant > ")
     */
    void PrintAssistantPrefix() const;

    /**
     * 打印一个解码的token信息 不执行flush
     */
    void PrintTokenText(const std::string& text) const;

    /**
     * 同步Flush stdout.
     */ 
    void Flush() const;

    // 换行.
    void NewLine() const;

    // 打印 /help 的输出信息.
    void PrintHelp() const;

    // 打印 session 的状态信息.
    void PrintStats(const ChatSession& session) const;

    // 打印采样参数
    void PrintParams(const SamplingParams& params) const;

    // 打印通用信息
    void PrintMessage(const std::string& msg) const;
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_TERMINAL_H_