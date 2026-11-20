#include "mini_llama/CLI/parsing_helper.h"

#include <iostream>
#include <cmath>
#include <cstdint>
#include <filesystem>

#include "mini_llama/model/loader.h"
#include "mini_llama/runtime/request_context.h"

namespace mini_llama {

void PrintRequestTrace(const RequestContext& request, std::ostream& out) {
    for (const std::string& line : FormatRequestTraceEvents(request)) {
        out << line << "\n";
    }
    out << FormatRequestTraceSummary(request) << "\n";
}

bool ParseIntArg(const char* text, int& value) {
    char* end = nullptr;
    errno = 0;
    int64_t parsed = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    // 若发生溢出 errno会被设置为ERANGE
    if (errno == ERANGE || parsed < 0 || parsed > 1000000) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool ParseFloatArg(const char* text, float& value) {
    char* end = nullptr;
    errno = 0;
    double parsed = std::strtod(text, &end);
    if (end == text || *end != '\0') {
        return false;
    }
    // 若发生溢出 errno会被设置为ERANGE
    if (errno == ERANGE || !std::isfinite(parsed) || parsed < 0.0 ||
        parsed > 10000.0) {
        return false;
    }
    value = static_cast<float>(parsed);
    return true;
}

bool ParseUintArg(const char* text, unsigned int& value) {
    if (text[0] == '-' || text[0] == '\0') {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    uint64_t parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    if (errno == ERANGE || parsed > std::numeric_limits<unsigned int>::max()) {
        return false;
    }
    value = static_cast<unsigned int>(parsed);
    return true;
}

std::string ResolveManifestTokenizerPath(const std::string& config_path) {
    try {
        ModelManifest manifest = ParseManifest(config_path);
        if (manifest.tokenizer.type != "json_vocab" ||
            manifest.tokenizer.path.empty()) {
            return "";
        }
        std::filesystem::path tokenizer_path(manifest.tokenizer.path);
        if (tokenizer_path.is_relative()) {
            std::filesystem::path config_file(config_path);
            tokenizer_path = config_file.parent_path() / tokenizer_path;
        }
        return tokenizer_path.string();
    } catch (const std::exception&) {
        return "";
    }
}

LoadedModel LoadModelAndTokenizer(
    const std::string& path, const std::string& explicit_config_path,
    const std::string& explicit_tokenizer_path) {
    LoadedModel result;

    std::string model_path = path;
    std::string config_path;
    std::string tokenizer_path;
    
    // 获取 JSON+BIN 格式的配置文件路径
    if (std::filesystem::is_directory(path)) {
        model_path = path + "/model.bin";
        if (!explicit_config_path.empty()) {
            config_path = explicit_config_path;
        } else {
            config_path = path + "/model.json";
        }
    } else {
        model_path = path;
        if (!explicit_config_path.empty()) {
            config_path = explicit_config_path;
        } else {
            config_path =
                (std::filesystem::path(path).parent_path() / "model.json").string();
        }
    }
    // 加载model
    result.model = LoadModel(config_path, model_path);
    if (!result.model.loaded) {
        return result;
    }

    // 尝试从入参处获取tokenizer路径 
    if (!explicit_tokenizer_path.empty()) {
        tokenizer_path = explicit_tokenizer_path;
    }

    // 尝试从JSON文件内获取tokenizer路径 
    if (tokenizer_path.empty()) {
        tokenizer_path = ResolveManifestTokenizerPath(config_path);
    }

    // 尝试从model文件同路径下获取tokenizer路径 
    if (tokenizer_path.empty()) {
        std::filesystem::path dir = std::filesystem::is_directory(path) ? std::filesystem::path(path)
                                                                        : std::filesystem::path(path).parent_path();
        std::filesystem::path auto_vocab = dir / "vocab_py.json";
        if (std::filesystem::exists(auto_vocab)) {
            tokenizer_path = auto_vocab.string();
        }
    }

    // 加载tokenizer
    result.tokenizer = CreateTokenizer(tokenizer_path);

    if (!result.tokenizer) {
        result.model.load_error = "Failed to load tokenizer";
        result.model.loaded = false;
        return result;
    }

    return result;
}

}   // mini_llama

