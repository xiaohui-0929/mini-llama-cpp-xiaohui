#include "mini_llama/CLI/run_mode.h"

#include <cstring>

#include "mini_llama/CLI/parsing_helper.h"
#include "mini_llama/model/loader.h"
#include "mini_llama/chat/prompt_builder.h"
#include "mini_llama/chat/terminal.h"
#include "mini_llama/chat/chat.h"
#include "mini_llama/runtime/context.h"
#include "mini_llama/forward/batch.h"
#include "mini_llama/forward/forward.h"

namespace mini_llama {

namespace {

static size_t CommonPrefixLength(const std::vector<int>& a, const std::vector<int>& b) {
    size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return i;
}

}   // namespace

void PrintRunUsage(const char* prog) {
  std::cout
      << "Usage: " << prog << " run <model-path|dir> [options]\n"
      << "Options:\n"
      << "  --temperature <T>  Sampling temperature (default: 0.0 = greedy)\n"
      << "  --top-k <k>        Top-k sampling (default: 0 = disabled)\n"
      << "  --seed <S>         Random seed (default: 0 = random)\n"
      << "  -n, --n-predict <n> Maximum response tokens per turn (default: "
         "64)\n"
      << "  --tokenizer <path> Path to vocab.json tokenizer file\n"
      // << "  --backend cpu|cuda Execution backend (default: cpu; cuda requires "
      //    "-DMINI_LLAMA_CUDA=ON)\n"
      // << "  --device <n>       CUDA device id for --backend cuda (default: 0)\n"
      << "  -h, --help         Show this help\n";
}

int RunChat(int argc, char** argv) {
    // 第三个参数为"-h"/"--help"时 显示提示信息
    if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 ||
                      std::strcmp(argv[2], "--help") == 0)) {
        PrintRunUsage(argv[0]);
        return 0;
    }

    // 参数少于3 提示没有输入model路径
    if (argc < 3) {
        std::cerr << "Missing model directory.\n";
        PrintRunUsage(argv[0]);
        return 1;
    }

    // 参数信息(可能以options的形式赋值)
    std::string model_dir = argv[2];
    float temperature = 0.0f;
    int top_k = 0;
    unsigned int seed = 0;
    int max_response_tokens = 64;
    std::string tokenizer_path;

    // 遍历剩余的参数 收集信息
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            if (!ParseFloatArg(argv[++i], temperature)) {
                std::cerr
                    << "Invalid --temperature value. Expected a non-negative float.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            if (!ParseIntArg(argv[++i], top_k)) {
                std::cerr
                    << "Invalid --top-k value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!ParseUintArg(argv[++i], seed)) {
                std::cerr << "Invalid --seed value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if ((std::strcmp(argv[i], "-n") == 0 ||
                    std::strcmp(argv[i], "--n-predict") == 0) && i + 1 < argc) {
            if (!ParseIntArg(argv[++i], max_response_tokens) ||
                max_response_tokens <= 0) {
                std::cerr
                    << "Invalid --n-predict value. Expected a positive integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            PrintRunUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            PrintRunUsage(argv[0]);
            return 1;
        }
    }

    // 暂不考虑backend

    // 加载模型
    LoadedModel lm = LoadModelAndTokenizer(model_dir, "", tokenizer_path);
    if (!lm.model.loaded) {
        std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
        return 1;
    }
    if (!lm.tokenizer) {
        std::cerr << "Failed to load tokenizer.\n";
        return 1;
    }
    if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
        std::cerr << "Model vocab_size must be at least "
                << lm.tokenizer->vocab_size() << " for the tokenizer.\n";
        return 1;
    }
    MiniLlamaModel& model = lm.model;
    std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;

    // 准备prompt生成器
    PromptBuilder builder;
    if (!lm.chat_template.empty()) {
        builder.SetChatTemplate(lm.chat_template);
    }

    // 设置对话管理ChatSession
    ChatSession session;
    session.sampling_params.temperature = temperature;
    session.sampling_params.top_k = top_k;
    session.sampling_params.seed = seed;

    if (lm.chat_template.empty()) {
        session.AddMessage("system", "You are a helpful assistant.");
    }

    // 输出前置提示信息
    Terminal term;
    term.PrintMessage("mini-llama.cpp chat");
    term.PrintMessage("backend: cpu");
    term.PrintMessage("Type /help for commands, /exit to quit.\n");
    if (lm.chat_template.empty() && model.config.max_seq_len <= 256) {
        term.PrintMessage(
            "Tiny teaching model: random weights, small context window, smoke-test "
            "output.");
        term.PrintMessage(
            "Real chat demo: ./build/mini-llama run models/chat -n 8\n");
    }

    // 准备对话过程中的上下文
    MiniLlamaContext ctx(&model);

    // 开启对话循环
    while (true) {
        term.PrintUserPrompt();
        // 读取输入的信息
        std::string input = term.ReadLine();
        if (input.empty() && std::cin.eof()) {
            break;
        }
        // 控制命令输出
        if (input == "/help") {
            term.PrintHelp();
            continue;
        }
        if (input == "/exit") {
            break;
        }
        if (input == "/clear") {
            session.Clear();
            ctx = MiniLlamaContext(&model);
            if (lm.chat_template.empty()) {
                session.AddMessage("system", "You are a helpful assistant.");
            }
            term.PrintMessage("Chat history cleared.\n");
            continue;
        }
        if (input == "/stats") {
            term.PrintStats(session);
            continue;
        }
        if (input == "/params") {
            term.PrintParams(session.sampling_params);
            continue;
        }
        if (!input.empty() && input[0] == '/') {
            term.PrintMessage("Unknown command: " + input + "\n");
            continue;
        }
        if (input.empty()) {
            continue;
        }

        // 设置本轮对话信息的request_context
        RequestContext request = StartRequest("run", "cpu", model_dir);

        // 将用户输入的信息以结构化格式加入Messages
        std::vector<ChatMessage> candidate_messages = session.messages;
        candidate_messages.push_back({"user", input});

        // message -> prompt_text
        auto stage_start = RequestClock::now();
        std::string prompt_text = builder.Build(candidate_messages);
        request.RecordEvent("prompt_build", ElapsedMs(stage_start), 0,
                            "messages=" + std::to_string(candidate_messages.size()));

        // prompt_text -> token_id
        stage_start = RequestClock::now();
        std::vector<int> tokens = tokenizer->Encode(prompt_text);
        request.tokenize_ms = ElapsedMs(stage_start);
        request.prompt_tokens = static_cast<int>(tokens.size());
        request.RecordEvent("tokenize", request.tokenize_ms, request.prompt_tokens,
                            "prompt");

        // 单次token数量超过最大序列长度 输出错误提示
        if (tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
            request.SetError("prompt uses " + std::to_string(tokens.size()) +
                             " tokens, context window is " +
                             std::to_string(model.config.max_seq_len) +
                             ". Use /clear or a shorter prompt.");
            request.Finish();
            PrintRequestTrace(request);
            term.PrintMessage("Error: " + request.error + "\n");
            continue;
        }
        // 添加了用户输入的信息再此由session接收
        session.messages = candidate_messages;

        // 获取单次最大返回数量
        int max_response = model.config.max_seq_len - static_cast<int>(tokens.size());
        if (max_response > max_response_tokens) {
            max_response = max_response_tokens;
        }
        
        // 单轮输出计时开始
        auto start = std::chrono::steady_clock::now();

        // 通过radix tree获取最大长度
        size_t cached_prefix_len = session.LongestCachedPrefix(tokens);
        // 通过context记录的历史token获取最大长度 代表了实际kv_cache内记录的信息
        size_t context_prefix_len = CommonPrefixLength(ctx.token_history, tokens);
        // 取两者中较小值
        size_t prefix_len = std::min(cached_prefix_len, context_prefix_len);
        if (prefix_len >= tokens.size()) {
            // 本次token完全在历史记录里 会导致没有足够的新token来进行Prefill
            // 重置ctx
            ctx = MiniLlamaContext(&model);
            prefix_len = 0;
        } else if (prefix_len == 0) {
            // 本次token完全没有命中历史记录
            // 重置ctx
            ctx = MiniLlamaContext(&model);
        } else if (prefix_len < ctx.token_history.size()) {
            ctx.token_history.resize(prefix_len);
            ctx.pos = static_cast<int>(prefix_len - 1);
        }
        // 截取真正需要计算的token
        std::vector<int> new_prompt_tokens(
        tokens.begin() + static_cast<std::ptrdiff_t>(prefix_len), tokens.end());
        session.SetTokenHistory(tokens);

        // Prefill阶段
        Tensor logits;
        {
            MiniBatch prefill = MiniBatch::FromTokens(new_prompt_tokens,
                                                static_cast<int>(prefix_len));
            stage_start = RequestClock::now();
            logits = ForwardBatch(ctx, model, prefill);
            request.prefill_ms = ElapsedMs(stage_start);
            request.prefill_tokens = static_cast<int>(new_prompt_tokens.size());
            request.RecordEvent(
                "prefill", request.prefill_ms, request.prefill_tokens,
                "radix_hit=" + std::to_string(cached_prefix_len) +
                    ", prefix_reuse=" + std::to_string(prefix_len));
            ctx.n_prefill_tokens += static_cast<int>(new_prompt_tokens.size());
        }

        // Decode阶段
        term.PrintAssistantPrefix();
        MiniSampler sampler(session.sampling_params);
        int generated_count = 0;
        std::vector<int> generated_ids;
        std::string streamed_reply;
        try {
            for (int i = 0; i < max_response; ++i) {
                // 采样 获取下一个token
                stage_start = RequestClock::now();
                int next_token = sampler.Sample(logits, session.sampling_params);
                request.sample_ms += ElapsedMs(stage_start);

                tokens.push_back(next_token);
                session.AppendToken(next_token);
                ++generated_count;

                // 逐token输出
                if (next_token != tokenizer->eos_id()) {
                    generated_ids.push_back(next_token);
                    std::string current_reply = tokenizer->Decode(generated_ids);
                    if (current_reply.size() > streamed_reply.size()) {
                        term.PrintTokenText(current_reply.substr(streamed_reply.size()));
                        term.Flush();
                        streamed_reply = current_reply;
                    }
                }

                // Decode计算下一个token
                MiniBatch decode_batch =
                    MiniBatch::Single(next_token, static_cast<int>(tokens.size() - 1));
                stage_start = RequestClock::now();
                logits = ForwardBatch(ctx, model, decode_batch);
                double decode_ms = ElapsedMs(stage_start);
                request.decode_ms += decode_ms;
                request.RecordEvent("decode", decode_ms, 1,
                                    "pos=" + std::to_string(tokens.size() - 1));
                ++ctx.n_decode_tokens;
                ++request.decode_tokens;
                if (next_token == tokenizer->eos_id()) {
                    break;
                }
            }
        } catch (const std::exception& e) {
            term.NewLine();
            request.SetError("Inference error: " + std::string(e.what()));
            request.Finish();
            PrintRequestTrace(request);
            term.PrintMessage(request.error + "\n");
            continue;
        }

        std::string assistant_reply = tokenizer->Decode(generated_ids);
        if (assistant_reply.size() > streamed_reply.size()) {
            term.PrintTokenText(assistant_reply.substr(streamed_reply.size()));
        }
        term.NewLine();
        term.NewLine();
        // 记录从前缀处理->Prefill->Decode的总耗时
        auto end = std::chrono::steady_clock::now();
        double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
        // 收集信息 更新session request
        session.AddMessage("assistant", assistant_reply);
        session.RecordTurn(static_cast<int>(new_prompt_tokens.size()),
                        generated_count, elapsed_ms);
        session.RecordPrefix(session.token_history);
        request.generated_tokens = generated_count;
        request.RecordEvent("sample", request.sample_ms, request.generated_tokens,
                            "generated_tokens");
        request.Finish();
        PrintRequestTrace(request);
    }

    // 输出结束信息
    term.PrintMessage("Goodbye.\n");
    return 0;
}

}   // namespace mini_llama