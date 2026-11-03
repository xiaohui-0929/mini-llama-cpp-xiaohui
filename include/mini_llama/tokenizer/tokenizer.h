#ifndef INCLUDE_MINI_LLAMA_TOKENIZER_H_
#define INCLUDE_MINI_LLAMA_TOKENIZER_H_

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <cstdint>
#include <memory>

namespace mini_llama {

/**
 * Tokenizer 抽象基类
 */
class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    // Encode 文字转对应tokenid
    virtual std::vector<int> Encode(const std::string& text) const = 0;

    // DecodeToken 解码单个token值
    virtual std::string DecodeToken(int token) const = 0;

    // Decode 解码序列token值
    virtual std::string Decode(const std::vector<int>& tokens) const =0;

    // vocab_size 词表大小
    virtual int vocab_size() const = 0;

    // bos_id 序列开始标志
    virtual int bos_id() const = 0;

    // eos_id 序列结束标志
    virtual int eos_id() const = 0;

    // unk_id 未知标志
    virtual int unk_id() const = 0;
};

/**
 * ASCII tokenizer 用于验证的ASCII分词器 0至127的字符级映射
 */
class AsciiTokenizer : public ITokenizer {
public:
    static constexpr int kVocabSize = 128;

    AsciiTokenizer() = default;

    std::vector<int> Encode(const std::string& text) const override;
    std::string DecodeToken(int token) const override;
    std::string Decode(const std::vector<int>& tokens) const override;

    int vocab_size() const override { return kVocabSize; }
    int bos_id() const override { return 1; }
    int eos_id() const override { return 2; }
    int unk_id() const override { return 0; }
};

/**
 * JSON词表分词器 词表信息通过JSON文件记录
 * 词表格式 [{"id": 0, "content": "<unk>", "special": true}, ...]
 * 内部采用“最长匹配”算法
 */
class JsonVocabTokenizer : public ITokenizer {
public:
    explicit JsonVocabTokenizer(const std::string& vocab_path);

    std::vector<int> Encode(const std::string& text) const override;
    std::string DecodeToken(int token) const override;
    std::string Decode(const std::vector<int>& tokens) const override;

    int vocab_size() const override { return vocab_size_; }
    int bos_id() const override { return bos_id_; }
    int eos_id() const override { return eos_id_; }
    int unk_id() const override { return unk_id_; }

private:
    // 词表单元
    struct VocabEntry {
        int id = 0;
        std::string content;
        bool special = false;
    };

    int vocab_size_ = 0;
    int bos_id_ = 1;
    int eos_id_ = 2;
    int unk_id_ = 0;

    // id -> entry 映射 数组下标为id 对应元素为词信息
    std::vector<VocabEntry> id_to_entry_;

    // content -> id 映射 词内容 -> 对应id
    std::vector<std::pair<std::string, int>> content_to_id_;
};

/**
 * BPE分词器 (GPT-2 style)
 */
class BpeTokenizer : public ITokenizer {
public:
    BpeTokenizer() = default;
    ~BpeTokenizer() override = default;

    bool Load(const std::string& vocab_path, const std::string& merges_path,
              const std::string& special_path);
    
    std::vector<int> Encode(const std::string& text) const override;
    std::string DecodeToken(int token) const override;
    std::string Decode(const std::vector<int>& tokens) const override;

    int vocab_size() const override { 
        return static_cast<int>(id_to_token_.size()); 
    }
    int bos_id() const override { return bos_id_; }
    int eos_id() const override { return eos_id_; }
    int unk_id() const override { return unk_id_; }

private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_token_;
    std::map<std::pair<std::string, std::string>, int> merge_ranks_;
    int bos_id_ = -1;
    int eos_id_ = -1;
    int unk_id_ = -1;

    std::vector<std::pair<std::string, int>> special_tokens_;

    std::vector<std::string> b2u_;
    std::unordered_map<std::string, uint8_t> u2b_;

private:
    void BuildByteMappings();
};

/**
 * 工厂函数
 */
std::unique_ptr<ITokenizer> CreateTokenizer(const std::string& vocab_path);

std::unique_ptr<ITokenizer> CreateBpeTokenizer(const std::string& vocab_path,
                                               const std::string& merges_path,
                                               const std::string& special_path);

}   // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_TOKENIZER_H_