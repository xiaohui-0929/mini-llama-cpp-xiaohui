#include "mini_llama/tokenizer/tokenizer.h"

namespace mini_llama {

std::vector<int> AsciiTokenizer::Encode(const std::string& text) const {
    std::vector<int> tokens;
    // 设置序列开始标志
    tokens.push_back(bos_id());
    // 遍历字符 识别对应ASCII码
    for (unsigned char c : text) {
        int id = static_cast<int>(c);
        if (id >= kVocabSize) {
            tokens.push_back(unk_id());
        } else {
            tokens.push_back(id);
        }
    }
    return tokens;
}

std::string AsciiTokenizer::DecodeToken(int token) const {
    if (token == bos_id()) {
        return "<bos>";
    }
    if (token == eos_id()) {
        return "<eos>";
    }
    if (token == unk_id()) {
        return "<unk>";
    }
    if (token >= kVocabSize) {
        return "<unk>";
    }
    if (token < 32 || token == 127) {
        // 控制字符无需解码   
        return "";
    }
    return std::string(1, static_cast<char>(token));
}

std::string AsciiTokenizer::Decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int token : tokens) {
        result += DecodeToken(token);
    }
    return result;
}

}   // namespace mini_llama