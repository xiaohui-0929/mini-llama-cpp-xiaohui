#include <fstream>
#include <sstream>
#include <filesystem>

#include "test_main.h"
#include "test_names.h"

#ifndef MINI_LLAMA_SOURCE_DIR
#define MINI_LLAMA_SOURCE_DIR "."
#endif

namespace {

static std::string TestDataPath(const std::string& relative) {
    return std::string(MINI_LLAMA_SOURCE_DIR) + "/" + relative;
}

std::string ReadTextFile(const std::string& path) {
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

/**
 * 生成临时文件
 */
std::filesystem::path TestTempPath(const std::string& name) {
    return std::filesystem::temp_directory_path() / ("mini_llama_" + name);
}

/**
 * 写入文件
 */
void WriteTextFile(const std::filesystem::path& path,
                   const std::string& content) {
    std::ofstream out(path);
    out << content;
}

/**
 * 将text内的from替换为to
 */
bool ReplaceOnce(std::string& text, const std::string& from,
                 const std::string& to) {
    size_t pos = text.find(from);
    if (pos == std::string::npos) {
        return false;
    }
    text.replace(pos, from.size(), to);
    return true;
}

/**
 * 将json内的tokenizerType进行替换
 */
std::string ReplaceTokenizerType(const std::string& json,
                                 const std::string& tokenizer_type) {
    std::string updated = json;
    if (!ReplaceOnce(updated, 
                    "\"type\": \"json_vocab\"",
                    "\"type\": \"" + tokenizer_type + "\"")) {
        throw std::runtime_error("failed to replace tokenizer type");
    }
    return updated;
}

/**
 * 将text内maker位置开始的from替换为to
 */
bool ReplaceAfter(std::string& text, const std::string& marker,
                  const std::string& from, const std::string& to) {
    size_t marker_pos = text.find(marker);
    if (marker_pos == std::string::npos) {
        return false;
    }
    size_t pos = text.find(from, marker_pos);
    if (pos == std::string::npos) {
        return false;
    }
    text.replace(pos, from.size(), to);
    return true;
}

/**
 * 测试错误json信息是否正确抛出error
 */
bool ParseManifestFails(const std::string& json, const std::string& filename) {
    std::filesystem::path path = TestTempPath(filename);
    WriteTextFile(path, json);
    try {
        ParseManifest(path.string());
        std::filesystem::remove(path);
        return false;
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        std::filesystem::remove(path);
        return true;
    }
}

/**
 * 移除Tokenizer
 */
std::string RemoveTokenizerBlock(const std::string& json) {
    const std::string tokenizer_block =
        "  \"tokenizer\": {\n"
        "    \"type\": \"json_vocab\",\n"
        "    \"path\": \"vocab.json\",\n"
        "    \"bos_id\": 1,\n"
        "    \"eos_id\": 2,\n"
        "    \"unk_id\": 0\n"
        "  },\n";
    std::string updated = json;
    if (!ReplaceOnce(updated, tokenizer_block, "")) {
        throw std::runtime_error("failed to remove tokenizer block");
    }
    return updated;
}

}   // namespace

/**
 * 测试基本的Manifest解析是否正确
 */
static bool TestLoaderManifestParsesMetadata() {
    ModelManifest manifest = ParseManifest(TestDataPath("models/tiny/model.json"));
    MINI_LLAMA_ASSERT_EQ(manifest.config.vocab_size, 128);
    MINI_LLAMA_ASSERT_EQ(manifest.config.n_layers, 2);
    MINI_LLAMA_ASSERT_EQ(manifest.tensors.size(), 21);
    MINI_LLAMA_ASSERT_TRUE(manifest.tensors[0].name == "token_embedding");
    MINI_LLAMA_ASSERT_TRUE(manifest.tensors[0].dtype == "float32");
    MINI_LLAMA_ASSERT_EQ(manifest.tensors[0].offset, 0);
    MINI_LLAMA_ASSERT_EQ(manifest.tensors[0].byte_size, 16384);
    MINI_LLAMA_ASSERT_TRUE(manifest.tokenizer.type == "json_vocab");
    MINI_LLAMA_ASSERT_TRUE(manifest.tokenizer.path == "vocab.json");
    MINI_LLAMA_ASSERT_EQ(manifest.tokenizer.bos_id, 1);
    MINI_LLAMA_ASSERT_EQ(manifest.tokenizer.eos_id, 2);
    MINI_LLAMA_ASSERT_EQ(manifest.tokenizer.unk_id, 0);
    return true;
}

/**
 * 测试json文件内没有tokenizer信息的情况下 是否将tokenizer设置为默认的ASCII格式
 */
static bool TestLoaderManifestDefaultsAsciiWithoutTokenizerBlock() {
    // 将json配置内的tokenizer信息删除 并生成临时文件
    std::string json = ReadTextFile(TestDataPath("models/tiny/model.json"));
    std::string without_tokenizer = RemoveTokenizerBlock(json);
    std::filesystem::path path = TestTempPath("tokenizer_manifest.json");
    WriteTextFile(path, without_tokenizer);
    // 解析临时文件
    ModelManifest manifest = ParseManifest(path.string());
    std::filesystem::remove(path);

    MINI_LLAMA_ASSERT_TRUE(manifest.tokenizer.type == "ascii");
    MINI_LLAMA_ASSERT_TRUE(manifest.tokenizer.path.empty());
    MINI_LLAMA_ASSERT_EQ(manifest.tokenizer.bos_id, 1);
    MINI_LLAMA_ASSERT_EQ(manifest.tokenizer.eos_id, 2);
    MINI_LLAMA_ASSERT_EQ(manifest.tokenizer.unk_id, 0);
    return true;
}

/**
 * 测试解析不支持的tokenizerType时 是否正确抛出异常
 */
static bool TestLoaderRejectsInvalidTokenizerType() {
    std::string json = ReadTextFile(TestDataPath("models/tiny/model.json"));
    // 将tokenizerType替换为当前不支持的类型
    std::string with_tokenizer = ReplaceTokenizerType(json, "bpe");
    MINI_LLAMA_ASSERT_TRUE(
        ParseManifestFails(with_tokenizer, "invalid_tokenizer_type.json"));
    return true;
}

/**
 * 测试含有重复tensor名称时 是否正确抛出异常
 */
static bool TestLoaderRejectsDuplicateTensorName() {
    std::string json = ReadTextFile("models/tiny/model.json");
    MINI_LLAMA_ASSERT_TRUE(ReplaceOnce(json, 
                                    "\"name\": \"final_norm\"",
                                    "\"name\": \"token_embedding\""));
    MINI_LLAMA_ASSERT_TRUE(ParseManifestFails(json, "duplicate_tensor.json"));
    return true;
}

/**
 * 测试tensor的shape不匹配时 是否正确抛出异常
 */
static bool TestLoaderRejectsRequiredShapeMismatch() {
    std::string json = ReadTextFile("models/tiny/model.json");
    MINI_LLAMA_ASSERT_TRUE(
        ReplaceAfter(json, 
                    "\"name\": \"lm_head\"",
                    "\"shape\": [\n        128,\n        32\n      ]",
                    "\"shape\": [\n        64,\n        64\n      ]"));
    MINI_LLAMA_ASSERT_TRUE(ParseManifestFails(json, "shape_mismatch.json"));
    return true;
}

/**
 * 测试tensor之间存在重叠时 是否正确抛出异常
 */
static bool TestLoaderRejectsOverlappingTensorRanges() {
    std::string json = ReadTextFile("models/tiny/model.json");
    MINI_LLAMA_ASSERT_TRUE(ReplaceAfter(json, 
                                        "\"name\": \"lm_head\"",
                                        "\"offset\": 115840",
                                        "\"offset\": 115712"));
    MINI_LLAMA_ASSERT_TRUE(ParseManifestFails(json, "overlap.json"));
    return true;
}

/**
 * 测试tensor内丢失dtype时 是否正确抛出异常
 */
static bool TestLoaderRejectsMissingDtype() {
    std::string json = ReadTextFile("models/tiny/model.json");
    MINI_LLAMA_ASSERT_TRUE(ReplaceAfter(json, 
                                        "\"name\": \"lm_head\"",
                                        "\"dtype\": \"float32\",\n",
                                        ""));
    MINI_LLAMA_ASSERT_TRUE(ParseManifestFails(json, "missing_dtype.json"));
    return true;
}

/**
 * 测试Int数值类型不和预期时 是否正确抛出异常
 */
static bool TestLoaderRejectsFractionalIntegerConfig() {
    std::string json = ReadTextFile("models/tiny/model.json");
    MINI_LLAMA_ASSERT_TRUE(
        ReplaceOnce(json, "\"n_heads\": 4", "\"n_heads\": 4.5"));
    MINI_LLAMA_ASSERT_TRUE(
        ParseManifestFails(json, "fractional_integer_config.json"));
    return true;
}

/**
 * 测试float数值类型异常时 是否正确抛出异常
 */
static bool TestLoaderRejectsNonfiniteFloatConfig() {
    std::string json = ReadTextFile("models/tiny/model.json");
    MINI_LLAMA_ASSERT_TRUE(
        ReplaceOnce(json, "\"rope_theta\": 10000.0", "\"rope_theta\": NaN"));
    MINI_LLAMA_ASSERT_TRUE(
        ParseManifestFails(json, "nonfinite_float_config.json"));
    return true;
}

/**
 * 测试bin文件大小和预期大小不一致时 是否正确抛出异常
 */
static bool TestLoaderRejectsTrailingWeightBytes() {
    // 创建一份临时数据文件 并改变其大小
    std::filesystem::path weights_path = TestTempPath("trailing_model.bin");
    {
        std::ifstream in(TestDataPath("models/tiny/model.bin"), std::ios::binary);
        std::ofstream out(weights_path, std::ios::binary);
        out << in.rdbuf();
        char extra = 0;
        out.write(&extra, 1);
    }

    MiniLlamaModel model =
        LoadModel(TestDataPath("models/tiny/model.json"), weights_path.string());
    std::filesystem::remove(weights_path);
    MINI_LLAMA_ASSERT_TRUE(!model.loaded);
    MINI_LLAMA_ASSERT_TRUE(model.load_error.find("trailing bytes") !=
                            std::string::npos);
    return true;
}

static struct LoaderTestRegistrar {
    LoaderTestRegistrar() {
        RegisterTest("loader_manifest_parses_metadata",
                    TestLoaderManifestParsesMetadata);
        RegisterTest("loader_manifest_defaults_ascii_without_tokenizer_block",
                    TestLoaderManifestDefaultsAsciiWithoutTokenizerBlock);
        RegisterTest("loader_rejects_invalid_tokenizer_type",
                    TestLoaderRejectsInvalidTokenizerType);
        RegisterTest("loader_rejects_duplicate_tensor_name",
                    TestLoaderRejectsDuplicateTensorName);
        RegisterTest("loader_rejects_required_shape_mismatch",
                    TestLoaderRejectsRequiredShapeMismatch);
        RegisterTest("loader_rejects_overlapping_tensor_ranges",
                    TestLoaderRejectsOverlappingTensorRanges);
        RegisterTest("loader_rejects_missing_dtype", 
                    TestLoaderRejectsMissingDtype);
        RegisterTest("loader_rejects_fractional_integer_config",
                    TestLoaderRejectsFractionalIntegerConfig);
        RegisterTest("loader_rejects_nonfinite_float_config",
                    TestLoaderRejectsNonfiniteFloatConfig);
        RegisterTest("loader_rejects_trailing_weight_bytes",
                    TestLoaderRejectsTrailingWeightBytes);
    }
    
} loader_test_registrar;