#include <cstring>
#include <string>
#include <vector>

#include "test_main.h"
#include "test_names.h"
#include "mini_llama/gguf/gguf.h"

#ifndef MINI_LLAMA_SOURCE_DIR
#define MINI_LLAMA_SOURCE_DIR "."
#endif

namespace {

std::string FixturePath(const std::string& name) {
    return std::string(MINI_LLAMA_SOURCE_DIR) + "/tests/data/gguf/" + name;
}

bool TestGgufMinimalParses() {
    mini_llama::GgufReader reader;
    std::string error;
    MINI_LLAMA_ASSERT_TRUE(reader.Open(FixturePath("minimal.gguf"), error));

    const mini_llama::GgufHeader& header = reader.Header();
    MINI_LLAMA_ASSERT_EQ(header.tensor_info_count, static_cast<uint64_t>(1));
    MINI_LLAMA_ASSERT_EQ(header.metadata_kv_count, static_cast<uint64_t>(2));
    MINI_LLAMA_ASSERT_EQ(reader.Metadata().size(), static_cast<size_t>(2));
    MINI_LLAMA_ASSERT_EQ(reader.Alignment(), static_cast<uint32_t>(32));
    MINI_LLAMA_ASSERT_EQ(reader.DataOffset(), static_cast<uint64_t>(160));
    MINI_LLAMA_ASSERT_EQ(reader.TensorInfo().size(), static_cast<size_t>(1));

    const mini_llama::GgufMetadataKV* arch =
        reader.GetMetadataByKey("general.architecture");
    MINI_LLAMA_ASSERT_TRUE(arch != nullptr);
    MINI_LLAMA_ASSERT_TRUE(
        arch->value.value_type == mini_llama::GgufMetadataValueType::kgguf_string);
    MINI_LLAMA_ASSERT_TRUE(arch->value.string == "qwen2");

    const mini_llama::GgufMetadataKV* alignment =
        reader.GetMetadataByKey("general.alignment");
    MINI_LLAMA_ASSERT_TRUE(alignment != nullptr);
    MINI_LLAMA_ASSERT_TRUE(
        alignment->value.value_type ==
        mini_llama::GgufMetadataValueType::kgguf_uint32);
    MINI_LLAMA_ASSERT_EQ(alignment->value.u64, static_cast<uint64_t>(32));

    const mini_llama::GgufTensorInfo* info =
        reader.GetTensorInfo("token_embd.weight");
    MINI_LLAMA_ASSERT_TRUE(info != nullptr);
    MINI_LLAMA_ASSERT_EQ(info->dims.size(), static_cast<size_t>(1));
    MINI_LLAMA_ASSERT_EQ(info->dims[0], static_cast<uint64_t>(4));
    MINI_LLAMA_ASSERT_TRUE(info->ggml_type == mini_llama::GgufGgmlType::kggml_F32);
    MINI_LLAMA_ASSERT_EQ(info->offset, static_cast<uint64_t>(0));
    MINI_LLAMA_ASSERT_EQ(info->byte_size, static_cast<uint64_t>(16));

    std::vector<uint8_t> data;
    MINI_LLAMA_ASSERT_TRUE(reader.ReadTensorData(info->name, data, error));
    MINI_LLAMA_ASSERT_EQ(data.size(), static_cast<size_t>(16));

    float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    std::memcpy(values, data.data(), sizeof(values));
    MINI_LLAMA_ASSERT_TRUE(values[0] == 1.0f);
    MINI_LLAMA_ASSERT_TRUE(values[1] == 2.0f);
    MINI_LLAMA_ASSERT_TRUE(values[2] == 3.0f);
    MINI_LLAMA_ASSERT_TRUE(values[3] == 4.0f);
    return true;
}

bool TestGgufRejectsBadMagic() {
    mini_llama::GgufReader reader;
    std::string error;
    MINI_LLAMA_ASSERT_TRUE(!reader.Open(FixturePath("bad_magic.gguf"), error));
    MINI_LLAMA_ASSERT_TRUE(!reader.Loaded());
    MINI_LLAMA_ASSERT_TRUE(reader.Metadata().empty());
    MINI_LLAMA_ASSERT_TRUE(error.find("magic") != std::string::npos);
    return true;
}

bool TestGgufRejectsBadVersion() {
    mini_llama::GgufReader reader;
    std::string error;
    MINI_LLAMA_ASSERT_TRUE(!reader.Open(FixturePath("bad_version.gguf"), error));
    MINI_LLAMA_ASSERT_TRUE(!reader.Loaded());
    MINI_LLAMA_ASSERT_TRUE(error.find("version") != std::string::npos);
    return true;
}

bool TestGgufRejectsTruncatedString() {
    mini_llama::GgufReader reader;
    std::string error;
    MINI_LLAMA_ASSERT_TRUE(
        !reader.Open(FixturePath("truncated_string.gguf"), error));
    MINI_LLAMA_ASSERT_TRUE(!reader.Loaded());
    MINI_LLAMA_ASSERT_TRUE(!error.empty());
    MINI_LLAMA_ASSERT_TRUE(error.find("string") != std::string::npos);
    return true;
}

bool TestGgufRejectsUnalignedTensorOffset() {
    mini_llama::GgufReader reader;
    std::string error;
    MINI_LLAMA_ASSERT_TRUE(
        !reader.Open(FixturePath("bad_tensor_offset.gguf"), error));
    MINI_LLAMA_ASSERT_TRUE(!reader.Loaded());
    MINI_LLAMA_ASSERT_TRUE(error.find("align") != std::string::npos);
    return true;
}

bool TestGgufParsesNestedArray() {
    mini_llama::GgufReader reader;
    std::string error;
    MINI_LLAMA_ASSERT_TRUE(reader.Open(FixturePath("nested_array.gguf"), error));
    MINI_LLAMA_ASSERT_EQ(reader.Metadata().size(), static_cast<size_t>(1));

    const mini_llama::GgufMetadataKV* kv = reader.GetMetadataByKey("test.nested");
    MINI_LLAMA_ASSERT_TRUE(kv != nullptr);
    MINI_LLAMA_ASSERT_TRUE(
        kv->value.value_type == mini_llama::GgufMetadataValueType::kgguf_array);
    MINI_LLAMA_ASSERT_EQ(kv->value.array.size(), static_cast<size_t>(2));

    for (size_t i = 0; i < kv->value.array.size(); ++i) {
        const mini_llama::GgufMetadataValue& inner = kv->value.array[i];
        MINI_LLAMA_ASSERT_TRUE(
            inner.value_type == mini_llama::GgufMetadataValueType::kgguf_array);
        MINI_LLAMA_ASSERT_EQ(inner.array.size(), static_cast<size_t>(2));
        MINI_LLAMA_ASSERT_TRUE(
            inner.array[0].value_type ==
            mini_llama::GgufMetadataValueType::kgguf_uint32);
        if (i == 0) {
            MINI_LLAMA_ASSERT_EQ(inner.array[0].u64, static_cast<uint64_t>(7));
            MINI_LLAMA_ASSERT_EQ(inner.array[1].u64, static_cast<uint64_t>(8));
        } else {
            MINI_LLAMA_ASSERT_EQ(inner.array[0].u64, static_cast<uint64_t>(9));
            MINI_LLAMA_ASSERT_EQ(inner.array[1].u64, static_cast<uint64_t>(10));
        }
    }
    return true;
}

bool TestGgufReadUnknownTensorFails() {
    mini_llama::GgufReader reader;
    std::string error;
    MINI_LLAMA_ASSERT_TRUE(reader.Open(FixturePath("minimal.gguf"), error));

    std::vector<uint8_t> data;
    MINI_LLAMA_ASSERT_TRUE(!reader.ReadTensorData("no_such_tensor", data, error));
    MINI_LLAMA_ASSERT_TRUE(!error.empty());
    return true;
}

bool TestGgufRepeatedLoadResetsState() {
    mini_llama::GgufReader reader;
    std::string error;
    MINI_LLAMA_ASSERT_TRUE(reader.Open(FixturePath("minimal.gguf"), error));
    MINI_LLAMA_ASSERT_TRUE(reader.Loaded());

    MINI_LLAMA_ASSERT_TRUE(!reader.Open(FixturePath("bad_magic.gguf"), error));
    MINI_LLAMA_ASSERT_TRUE(!reader.Loaded());
    MINI_LLAMA_ASSERT_EQ(reader.Header().version, static_cast<uint32_t>(0));
    MINI_LLAMA_ASSERT_EQ(reader.Header().tensor_info_count,
                         static_cast<uint64_t>(0));
    MINI_LLAMA_ASSERT_EQ(reader.Header().metadata_kv_count,
                         static_cast<uint64_t>(0));
    MINI_LLAMA_ASSERT_TRUE(reader.Metadata().empty());
    MINI_LLAMA_ASSERT_TRUE(reader.TensorInfo().empty());
    MINI_LLAMA_ASSERT_EQ(reader.DataOffset(), static_cast<uint64_t>(0));
    MINI_LLAMA_ASSERT_EQ(reader.Alignment(), static_cast<uint32_t>(0));
    MINI_LLAMA_ASSERT_TRUE(!error.empty());
    return true;
}

bool TestGgufMissingFile() {
    mini_llama::GgufReader reader;
    std::string error;
    MINI_LLAMA_ASSERT_TRUE(
        !reader.Open(FixturePath("does_not_exist.gguf"), error));
    MINI_LLAMA_ASSERT_TRUE(!reader.Loaded());
    MINI_LLAMA_ASSERT_TRUE(!error.empty());
    return true;
}

bool TestGgufDuplicateMetadataKey() {
    mini_llama::GgufReader reader;
    std::string error;
    MINI_LLAMA_ASSERT_TRUE(
        !reader.Open(FixturePath("duplicate_metadata.gguf"), error));
    MINI_LLAMA_ASSERT_TRUE(!reader.Loaded());
    MINI_LLAMA_ASSERT_TRUE(error.find("duplicate") != std::string::npos);
    return true;
}

bool TestGgufZeroTensorDimension() {
    mini_llama::GgufReader reader;
    std::string error;
    MINI_LLAMA_ASSERT_TRUE(
        !reader.Open(FixturePath("zero_dimension.gguf"), error));
    MINI_LLAMA_ASSERT_TRUE(!reader.Loaded());
    MINI_LLAMA_ASSERT_TRUE(error.find("dimension") != std::string::npos);
    return true;
}

bool TestGgufTensorDataOutOfBounds() {
    mini_llama::GgufReader reader;
    std::string error;
    MINI_LLAMA_ASSERT_TRUE(
        !reader.Open(FixturePath("tensor_out_of_bounds.gguf"), error));
    MINI_LLAMA_ASSERT_TRUE(!reader.Loaded());
    MINI_LLAMA_ASSERT_TRUE(error.find("range") != std::string::npos);
    return true;
}

bool TestGgufTruncatedFile() {
    mini_llama::GgufReader reader;
    std::string error;
    MINI_LLAMA_ASSERT_TRUE(
        !reader.Open(FixturePath("truncated_file.gguf"), error));
    MINI_LLAMA_ASSERT_TRUE(!reader.Loaded());
    MINI_LLAMA_ASSERT_TRUE(!error.empty());
    return true;
}

bool TestGgufInvalidAlignment() {
    mini_llama::GgufReader reader;
    std::string error;
    MINI_LLAMA_ASSERT_TRUE(
        !reader.Open(FixturePath("bad_alignment.gguf"), error));
    MINI_LLAMA_ASSERT_TRUE(!reader.Loaded());
    MINI_LLAMA_ASSERT_TRUE(error.find("alignment") != std::string::npos);
    return true;
}

bool TestGgufTypeNames() {
    MINI_LLAMA_ASSERT_TRUE(
        std::string(mini_llama::GgufReader::GgmlTypeName(0)) == "F32");
    MINI_LLAMA_ASSERT_TRUE(
        std::string(mini_llama::GgufReader::GgmlTypeName(1)) == "F16");
    MINI_LLAMA_ASSERT_TRUE(
        std::string(mini_llama::GgufReader::GgmlTypeName(8)) == "UNSUPPORTED");     // Q8_0暂不支持
    MINI_LLAMA_ASSERT_TRUE(
        std::string(mini_llama::GgufReader::GgmlTypeName(12)) == "UNSUPPORTED");    // Q4_K暂不支持

    MINI_LLAMA_ASSERT_TRUE(
        std::string(mini_llama::GgufReader::GgufMetadataValueTypeName(4)) ==
        "uint32");
    MINI_LLAMA_ASSERT_TRUE(
        std::string(mini_llama::GgufReader::GgufMetadataValueTypeName(8)) ==
        "string");
    MINI_LLAMA_ASSERT_TRUE(
        std::string(mini_llama::GgufReader::GgufMetadataValueTypeName(9)) ==
        "array");
    return true;
}

static struct GgufReaderTestRegistrar {
    GgufReaderTestRegistrar() {
        RegisterTest("gguf_minimal_parses", TestGgufMinimalParses);
        RegisterTest("gguf_rejects_bad_magic", TestGgufRejectsBadMagic);
        RegisterTest("gguf_rejects_bad_version", TestGgufRejectsBadVersion);
        RegisterTest("gguf_rejects_truncated_string",
                     TestGgufRejectsTruncatedString);
        RegisterTest("gguf_rejects_unaligned_tensor_offset",
                     TestGgufRejectsUnalignedTensorOffset);
        RegisterTest("gguf_parses_nested_array", TestGgufParsesNestedArray);
        RegisterTest("gguf_read_unknown_tensor_fails",
                     TestGgufReadUnknownTensorFails);
        RegisterTest("gguf_repeated_load_resets_state",
                     TestGgufRepeatedLoadResetsState);
        RegisterTest("gguf_missing_file", TestGgufMissingFile);
        RegisterTest("gguf_duplicate_metadata_key",
                     TestGgufDuplicateMetadataKey);
        RegisterTest("gguf_zero_tensor_dimension",
                     TestGgufZeroTensorDimension);
        RegisterTest("gguf_tensor_data_out_of_bounds",
                     TestGgufTensorDataOutOfBounds);
        RegisterTest("gguf_truncated_file", TestGgufTruncatedFile);
        RegisterTest("gguf_invalid_alignment", TestGgufInvalidAlignment);
        RegisterTest("gguf_type_names", TestGgufTypeNames);
    }
} gguf_reader_test_registrar;

}  // namespace
