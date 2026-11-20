#include "mini_llama/CLI/generate_mode.h"

#include <string>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>

#include "mini_llama/CLI/parsing_helper.h"
#include "mini_llama/tensor_and_Ops/thread_pool.h"
#include "mini_llama/runtime/context.h"
#include "mini_llama/chat/chat.h"
#include "mini_llama/forward/batch.h"
#include "mini_llama/forward/forward.h"

namespace mini_llama {

namespace {

static void EnsureDumpDirectory(const std::string& path) {
    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (error) {
        throw std::runtime_error("failed to create logits dump directory: " + path +
                                ": " + error.message());
    }
    if (!std::filesystem::is_directory(path)) {
        throw std::runtime_error("logits dump path is not a directory: " + path);
    }
}

static void DumpLogits(const Tensor& logits, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open logits dump file: " + path);
    }
    out.write(reinterpret_cast<const char*>(logits.data.data()),
              static_cast<std::streamsize>(logits.data.size() * sizeof(float)));
    if (!out.good()) {
        throw std::runtime_error("failed to write logits to: " + path);
    }
}

static void DumpGeneratedTokens(const std::vector<int>& generated,
                                const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open generation token dump file: " +
                                path);
    }
    for (size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) {
            out << " ";
        }
        out << generated[i];
    }
    out << "\n";
    if (!out.good()) {
        throw std::runtime_error("failed to write generation tokens to: " + path);
    }
}

}   // namespace

void PrintGenerateUsage(const char* prog) {
    std::cout
        << "Usage: " << prog << " generate [options]\n"
        << "Options:\n"
        << "  --model <path|dir>   Path to model weights binary or model "
            "directory (default: models/tiny/model.bin)\n"
        << "  --config <path>      Path to model config JSON (default: "
            "models/tiny/model.json)\n"
        << "  -p, --prompt <str>   Input prompt text (default: \"hello\")\n"
        << "  -n, --n-predict <n>  Number of tokens to generate (default: 16)\n"
        << "  --temperature <T>    Sampling temperature (default: 0.0 = greedy)\n"
        << "  --top-k <k>          Top-k sampling (default: 0 = disabled)\n"
        << "  --seed <S>           Random seed for reproducible sampling "
            "(default: 0 = random)\n"
        << "  --tokenizer <path>   Path to vocab.json tokenizer file\n"
        << "  --quant q8_0|q4_0    Quantize loaded Linear weights before "
            "generation\n"
        << "  --threads <n>        Number of threads for parallel ops (0 = "
            "auto)\n"
        << "  --backend cpu|cuda   Execution backend (default: cpu; cuda "
            "requires -DMINI_LLAMA_CUDA=ON)\n"
        << "  --device <n>         CUDA device id for --backend cuda (default: "
            "0)\n"
        << "  --dump-logits <dir>  Dump logits for each step to directory\n"
        << "  -h, --help           Show this help\n";
}

int RunGenerate(int argc, char** argv) {
    std::string model_path = "models/tiny/model.bin";
    std::string config_path = "models/tiny/model.json";
    bool config_path_set = false;
    std::string prompt = "hello";
    int n_predict = 16;
    float temperature = 0.0f;
    int top_k = 0;
    unsigned int seed = 0;
    std::string dump_logits_dir;
    std::string tokenizer_path;
    //std::string quant_type;   // 暂不考虑量化
    int n_threads = 0;

    // 收集可能的参数信息
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
            config_path_set = true;
        } else if ((std::strcmp(argv[i], "-p") == 0 ||
                    std::strcmp(argv[i], "--prompt") == 0) &&
                    i + 1 < argc) {
            prompt = argv[++i];
        } else if ((std::strcmp(argv[i], "-n") == 0 ||
                    std::strcmp(argv[i], "--n-predict") == 0) &&
                    i + 1 < argc) {
            if (!ParseIntArg(argv[++i], n_predict)) {
                std::cerr << "Invalid --n-predict value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            if (!ParseFloatArg(argv[++i], temperature)) {
                std::cerr << "Invalid --temperature value. Expected a non-negative float.\n";
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
        } else if (std::strcmp(argv[i], "--dump-logits") == 0 && i + 1 < argc) {
            dump_logits_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
        if (!ParseIntArg(argv[++i], n_threads) || n_threads < 0) {
            std::cerr << "Invalid --threads value.\n";
            return 1;
        }
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            PrintGenerateUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: " << argv[i] << "\n";
            PrintGenerateUsage(argv[0]);
            return 1;
        }
    }

    std::cout << "mini-llama.cpp\n";
    std::cout << "==============\n\n";

    // 确保保存生成结果的路径存在
    if (!dump_logits_dir.empty()) {
        try {
            EnsureDumpDirectory(dump_logits_dir);
        } catch (const std::exception& e) {
            std::cerr << "Logits dump setup failed: " << e.what() << "\n";
            return 1;
        }
    }

    RequestContext request = StartRequest("generate", "cpu", model_path);
    // 加载model 和 tokenizer
    auto stage_start = RequestClock::now();
    LoadedModel lm = LoadModelAndTokenizer(
        model_path, config_path_set ? config_path : "", tokenizer_path);
    request.model_load_ms = ElapsedMs(stage_start);
    request.RecordEvent("model_load", request.model_load_ms, 0, model_path);
    if (!lm.model.loaded) {
        request.SetError("Failed to load model: " + lm.model.load_error);
        request.Finish();
        PrintRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }
    if (!lm.tokenizer) {
        request.SetError("Failed to load tokenizer.");
        request.Finish();
        PrintRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }
    if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
        request.SetError("Model vocab_size must be at least " +
                        std::to_string(lm.tokenizer->vocab_size()) +
                        " for the tokenizer.");
        request.Finish();
        PrintRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }

    MiniLlamaModel& model = lm.model;
    std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;
    SetThreadCount(n_threads);

    // prompt -> tokens
    stage_start = RequestClock::now();
    std::vector<int> tokens = tokenizer->Encode(prompt);
    request.tokenize_ms = ElapsedMs(stage_start);
    request.prompt_tokens = static_cast<int>(tokens.size());
    request.RecordEvent("tokenize", request.tokenize_ms, request.prompt_tokens,
                        "prompt");
    
    std::cout << "prompt: " << prompt << "\n";
    std::cout << "tokens: [";
    // 打印tokens到终端
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
        std::cout << ", ";
        }
        std::cout << tokens[i];
    }
    std::cout << "]\n";
    std::cout << "sampling: temperature=" << temperature << ", top_k=" << top_k
              << ", seed=" << seed << "\n";
    std::cout << "quant: " << "model-native" 
              << "\n";
    std::cout << "threads: " << GetThreadCount() << "\n\n";
    // tokens数量不能超过最大序列长度
    if (tokens.size() > static_cast<size_t>(model.config.max_seq_len)) {
        request.SetError("Prompt is too long for max_seq_len=" +
                        std::to_string(model.config.max_seq_len) + ".");
        request.Finish();
        PrintRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }
    if (tokens.size() + static_cast<size_t>(n_predict) >
        static_cast<size_t>(model.config.max_seq_len)) {
        request.SetError("Requested tokens exceed context window.");
        request.Finish();
        PrintRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }

    // 设置context sampler
    MiniLlamaContext ctx(&model);
    SamplingParams sampling_params;
    sampling_params.temperature = temperature;
    sampling_params.top_k = top_k;
    sampling_params.seed = seed;
    MiniSampler sampler(sampling_params);
    // 模拟单次的prefill和decode
    size_t prompt_len = tokens.size();
    int step = 0;
    try {
        Tensor logits;
        // 模拟prefill阶段
        std::cout << "prefill...\n";
        stage_start = RequestClock::now();
        if (!dump_logits_dir.empty()) {
            for (size_t i = 0; i < prompt_len; ++i) {
                MiniBatch prefill_step =
                    MiniBatch::FromTokens({tokens[i]}, static_cast<int>(i));
                logits = ForwardBatch(ctx, model, prefill_step);
                ++ctx.n_prefill_tokens;
                DumpLogits(logits, dump_logits_dir + "/logits_step" +
                                    std::to_string(step) + ".bin");
                ++step;
            }
        } else {
            MiniBatch prefill = MiniBatch::FromTokens(tokens, 0);
            logits = ForwardBatch(ctx, model, prefill);
            ctx.n_prefill_tokens += static_cast<int>(tokens.size());
        }
        request.prefill_ms = ElapsedMs(stage_start);
        request.prefill_tokens = static_cast<int>(prompt_len);
        request.RecordEvent("prefill", request.prefill_ms, request.prefill_tokens,
                            dump_logits_dir.empty() ? "batch" : "step_dump");
        // 模拟decode阶段
        if (n_predict > 0) {
            stage_start = RequestClock::now();
            int next_token = sampler.Sample(logits, sampling_params);
            request.sample_ms += ElapsedMs(stage_start);
            tokens.push_back(next_token);

            if (next_token != tokenizer->eos_id()) {
                std::cout << "decode loop...\n";
                for (int i = 1; i < n_predict; ++i) {
                    MiniBatch decode_batch = MiniBatch::Single(
                        tokens.back(), static_cast<int>(tokens.size() - 1));
                    stage_start = RequestClock::now();
                    logits = ForwardBatch(ctx, model, decode_batch);
                    double decode_ms = ElapsedMs(stage_start);
                    request.decode_ms += decode_ms;
                    request.RecordEvent("decode", decode_ms, 1,
                                        "pos=" + std::to_string(tokens.size() - 1));
                    ++ctx.n_decode_tokens;
                    ++request.decode_tokens;
                    if (!dump_logits_dir.empty()) {
                        DumpLogits(logits, dump_logits_dir + "/logits_step" +
                                                std::to_string(step) + ".bin");
                        ++step;
                    }
                    stage_start = RequestClock::now();
                    next_token = sampler.Sample(logits, sampling_params);
                    request.sample_ms += ElapsedMs(stage_start);
                    tokens.push_back(next_token);
                    if (next_token == tokenizer->eos_id()) {
                        break;
                    }
                }   
            }

        } else {
            std::cout << "decode loop skipped.\n";
        }
    } catch (const std::exception& e) {
        request.SetError("Inference failed: " + std::string(e.what()));
        request.Finish();
        PrintRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }
    // 记录生成的token
    std::vector<int> generated(tokens.begin() + prompt_len, tokens.end());
    if (!dump_logits_dir.empty()) {
        try {
            DumpGeneratedTokens(generated,
                                dump_logits_dir + "/generation_tokens.txt");
        } catch (const std::exception& e) {
            request.SetError("Logits dump failed: " + std::string(e.what()));
            request.Finish();
            PrintRequestTrace(request, std::cerr);
            std::cerr << request.error << "\n";
            return 1;
        }
    }

    request.generated_tokens = static_cast<int>(generated.size());
    request.RecordEvent("sample", request.sample_ms, request.generated_tokens,
                        "generated_tokens");
    request.Finish();
    // 打印生成token
    std::cout << "\ngenerated tokens: [";
    for (size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) {
        std::cout << ", ";
        }
        std::cout << generated[i];
    }
    std::cout << "]\n";
    // 打印解码文字
    std::string generated_text = tokenizer->Decode(generated);
    std::cout << "generated text: \"" << generated_text << "\"\n";

    PrintRequestTrace(request);
    return 0;
}

}   // namespace mini_llama

