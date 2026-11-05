#include "mini_llama/tokenizer/tokenizer.h"
#include "utf8_util.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include <limits>

namespace mini_llama {

/**
 * UTF-8 helpers
 */

static std::string NextUtf8Codepoint(const std::string& s, size_t& pos) {
    if (pos >= s.size()) {
        return "";
    }
    unsigned char first = s[pos];
    size_t len = 1;
    if ((first & 0x80) == 0) {
        len = 1;
    } else if ((first & 0xE0) == 0xC0) {
        len = 2;
    } else if ((first & 0xF0) == 0xE0) {
        len = 3;
    } else if ((first & 0xF8) == 0xF0) {
        len = 4;
    } else {
        // Invalid UTF-8, skip one byte
        ++pos;
        return "";
    }
    if (pos + len > s.size()) {
        ++pos;
        return "";
    }
    std::string result = s.substr(pos, len);
    pos += len;
    return result;
}

/**
 * GPT-2 预切分
 *
 * 复现正则:
 *   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
 *
 * 字符分类做了简化 (没有 Unicode 表):
 *   letter: ASCII 字母, 以及所有非 ASCII 码点
 *   number: ASCII 数字
 *   space : ASCII 空白 (' ' \t \n \r \v \f)
 *   other : 其余字符; 非法 UTF-8 字节按单字节处理
 * 已知差异: U+00A0、U+3000 等 Unicode 空白与非 ASCII 标点会被当作 letter。
 */

enum class CharClass { kLetter, kNumber, kSpace, kOther };

static CharClass ClassifyAt(const std::string& s, size_t pos, size_t& len) {
    unsigned char c = static_cast<unsigned char>(s[pos]);
    len = 1;
    if (c < 0x80) {
        if (std::isalpha(c)) {
            return CharClass::kLetter;
        }
        if (std::isdigit(c)) {
            return CharClass::kNumber;
        }
        if (c == ' ' || (c >= '\t' && c <= '\r')) {
            return CharClass::kSpace;
        }
        return CharClass::kOther;
    }
    size_t seq = 0;
    if ((c & 0xE0) == 0xC0) {
        seq = 2;
    } else if ((c & 0xF0) == 0xE0) {
        seq = 3;
    } else if ((c & 0xF8) == 0xF0) {
        seq = 4;
    }
    if (seq == 0 || pos + seq > s.size()) {
        return CharClass::kOther;
    }
    len = seq;
    return CharClass::kLetter;
}

// s[pos] == '\'' 时, 返回缩写 ('s 't 're 've 'm 'll 'd) 的总长度, 不匹配返回 0
static size_t MatchContraction(const std::string& s, size_t pos) {
    static const char* const kSuffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    for (const char* suffix : kSuffixes) {
        size_t n = std::strlen(suffix);
        if (s.compare(pos + 1, n, suffix) == 0 && pos + 1 + n <= s.size()) {
            return n + 1;
        }
    }
    return 0;
}

static std::vector<std::string> PreTokenize(const std::string& s) {
    std::vector<std::string> pieces;
    const size_t n = s.size();
    size_t i = 0;
    while (i < n) {
        size_t len = 0;
        CharClass cls = ClassifyAt(s, i, len);

        // 1. 缩写
        if (s[i] == '\'') {
            size_t m = MatchContraction(s, i);
            if (m > 0) {
                pieces.push_back(s.substr(i, m));
                i += m;
                continue;
            }
        }

        // 2~4. 可选的一个前导空格 + 同类字符连续段
        size_t p = i;
        if (cls == CharClass::kSpace && s[i] == ' ' && i + 1 < n) {
            size_t next_len = 0;
            CharClass next_cls = ClassifyAt(s, i + 1, next_len);
            if (next_cls != CharClass::kSpace) {
                p = i + 1;
                cls = next_cls;
                len = next_len;
            }
        }
        if (cls != CharClass::kSpace) {
            p += len;
            while (p < n) {
                size_t l = 0;
                if (ClassifyAt(s, p, l) != cls) {
                    break;
                }
                p += l;
            }
            pieces.push_back(s.substr(i, p - i));
            i = p;
            continue;
        }

        // 5~6. 空白段: 后面还有非空白时, 留最后一个空白给下一个 piece
        size_t j = i;
        size_t last = i;
        while (j < n) {
            size_t l = 0;
            if (ClassifyAt(s, j, l) != CharClass::kSpace) {
                break;
            }
            last = j;
            j += l;
        }
        size_t end = (j < n && last > i) ? last : j;
        pieces.push_back(s.substr(i, end - i));
        i = end;
    }
    return pieces;
}

/**
 * Simple JSON number parser helper
 */

static bool ParseJsonInt(const std::string& s, size_t& pos, int& out) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
        ++pos;
    }
    if (pos >= s.size()) {
        return false;
    }
    size_t start = pos;
    if (s[pos] == '-') {
        ++pos;
    }
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
        ++pos;
    }
    if (start == pos) {
        return false;
    }
    try {
        out = std::stoi(s.substr(start, pos - start));
        return true;
    } catch (...) {
        return false;
    }
}

static bool ParseJsonString(const std::string& s, size_t& pos,
                            std::string& out) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
        ++pos;
    }
    if (pos >= s.size() || s[pos] != '"') {
        return false;
    }
    ++pos;  // skip opening quote
    out.clear();
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            char esc = s[pos + 1];
            switch (esc) {
                case '"':
                    out += '"';
                    break;
                case '\\':
                    out += '\\';
                    break;
                case '/':
                    out += '/';
                    break;
                case 'b':
                    out += '\b';
                    break;
                case 'f':
                    out += '\f';
                    break;
                case 'n':
                    out += '\n';
                    break;
                case 'r':
                    out += '\r';
                    break;
                case 't':
                    out += '\t';
                    break;
                case 'u': {
                    if (pos + 5 < s.size()) {
                        std::string hex = s.substr(pos + 2, 4);
                        try {
                            char32_t cp = static_cast<char32_t>(std::stoul(hex, nullptr, 16));
                            out += CodepointToUtf8(cp);
                        } catch (...) {
                            out += '?';
                        }
                        pos += 4;
                    }
                    break;
                }
                default:
                    out += esc;
                    break;
            }
            pos += 2;
        } else {
            out += s[pos];
            ++pos;
        }
    }
    if (pos < s.size() && s[pos] == '"') {
        ++pos;
    }
    return true;
}

/**
 * GPT-2 字节转unicode 映射
 */

static std::vector<std::string> BuildBytesToUnicode() {
    std::vector<std::string> map(256);

    std::vector<int> bs;
    for (int c = '!'; c <= '~'; ++c) {
        bs.push_back(c);
    }
    for (int c = 0xA1; c <= 0xAC; ++c) {
        bs.push_back(c);
    }
    for (int c = 0xAE; c <= 0xFF; ++c) {
        bs.push_back(c);
    }

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        map[bs[i]] = CodepointToUtf8(static_cast<char32_t>(cs[i]));
    }
    return map;
}

/**
 * BpeTokenizer
 */

void BpeTokenizer::BuildByteMappings() {
    b2u_ = BuildBytesToUnicode();
    u2b_.clear();
    for (int b = 0; b < 256; ++b) {
        u2b_[b2u_[b]] = static_cast<uint8_t>(b);
    }
}

bool BpeTokenizer::Load(const std::string& vocab_path, 
                        const std::string& merges_path,
                        const std::string& special_path) {
    BuildByteMappings();
    
    auto skipSpace = [](size_t& pos, const std::string& content) {
        while (pos < content.size() &&
            std::isspace(static_cast<unsigned char>(content[pos]))) {
            ++pos;
        }
    };

    // Load vocab.json: { "token": id, ... }
    {
        std::ifstream f(vocab_path);
        if (!f.is_open()) {
            std::cerr << "Failed to open vocab: " << vocab_path << std::endl;
            return false;
        }
        std::stringstream buf;
        buf << f.rdbuf();
        std::string content = buf.str();

        size_t pos = 0;
        // 跳过空格
        skipSpace(pos, content);
        // 起始位置 '{' 跳过
        if (pos < content.size() && content[pos] == '{') {
            ++pos;
        }
        // 循环记录vocab_
        while (pos < content.size()) {
            skipSpace(pos, content);
            // 终止位置 '}' 结束
            if (pos < content.size() && content[pos] == '}') {
                break;
            }
            // 解析Key
            std::string key;
            if (!ParseJsonString(content, pos, key)) {
                break;
            }
            skipSpace(pos, content);
            // 非分割位置 ':' 结束
            if (pos >= content.size() || content[pos] != ':') {
                break;
            }
            ++pos;
            // 解析Val
            int val = 0;
            if (!ParseJsonInt(content, pos, val)) {
                break;
            }
            // 如果val解析为负数 说明解析有问题 应该返回失败
            if (val < 0) {
                std::cerr << "Invalid negative token id " << val
                << " for token: " << key << std::endl;
                return false;
            }
            // 记录vocab_
            vocab_[key] = val;

            skipSpace(pos, content);
            // 分割位置 ',' 跳过
            if (pos < content.size() && content[pos] == ',') {
                ++pos;
            }
        }
        if (vocab_.empty()) {
            std::cerr << "Vocab is empty" << std::endl;
            return false;
        }
        // 寻找最大id
        int max_id = 0;
        for (const auto& kv : vocab_) {
            if (kv.second > max_id) {
                max_id = kv.second;
            }
        }
        // 记录id_to_token_映射
        id_to_token_.resize(max_id + 1);
        for (const auto& kv : vocab_) {
            id_to_token_[kv.second] = kv.first;
        }
        // 收集特殊token 如<|im_start|>
        for (const auto& kv : vocab_) {
            if (kv.first.size() >= 4 && kv.first.front() == '<' &&
                kv.first[1] == '|' && kv.first[kv.first.size() - 2] == '|' &&
                kv.first.back() == '>') {
                special_tokens_.push_back({kv.first, kv.second});
            }
        }
        // 对特殊token进行排序
        std::sort(special_tokens_.begin(), special_tokens_.end(),
                [](const auto& a, const auto& b) {
                    return a.first.size() > b.first.size();
                });
    }

    // Load merges.txt: "token1 token2" per line
    {
        std::ifstream f(merges_path);
        if (!f.is_open()) {
            std::cerr << "Failed to open merges: " << merges_path << std::endl;
            return false;
        }
        std::string line;
        int rank = 0;
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            std::string a, b;
            if (ss >> a >> b) {
                merge_ranks_[{a, b}] = rank++;
            }
        }
    }

    // Load special_tokens.json
    {
        std::ifstream f(special_path);
        if (!f.is_open()) {
            std::cerr << "Failed to open special: " << special_path << std::endl;
            return false;
        }
        std::stringstream buf;
        buf << f.rdbuf();
        std::string content = buf.str();

        auto find_int = [&](const std::string& key) -> int {
            size_t p = content.find('"' + key + '"');
            if (p == std::string::npos) {
                return -1;
            }
            p = content.find(':', p);
            if (p == std::string::npos) {
                return -1;
            }
            size_t start = p + 1;
            while (start < content.size() &&
                std::isspace(static_cast<unsigned char>(content[start]))) {
                ++start;
            }
            size_t end = start;
            while (end < content.size() &&
                (std::isdigit(static_cast<unsigned char>(content[end])) ||
                    content[end] == '-')) {
                ++end;
            }
            if (start == end) {
                return -1;
            }
            try {
                return std::stoi(content.substr(start, end - start));
            } catch (...) {
                return -1;
            }
        };
        bos_id_ = find_int("bos_id");
        eos_id_ = find_int("eos_id");
        unk_id_ = find_int("unk_id");
    }

    std::cout << "BPE tokenizer loaded: vocab=" << vocab_.size()
              << " merges=" << merge_ranks_.size() << " bos=" << bos_id_
              << " eos=" << eos_id_ << " unk=" << unk_id_ << std::endl;
    return true;
}

void BpeTokenizer::EncodePiece(const std::string& piece,
                               std::vector<int>& out) const {
    // 1.将 piece 按照字节切分
    // "hello" -> "h" "e" "l" "l" "o"     (ASCII, 5 字节, 5 个元素)
    // "世界"  -> "ä" "¸" "ĸ" "ç" "ķ" "Į" (6 字节, 6 个元素, 非 2 个)
    std::vector<std::string> word;
    word.reserve(piece.size());
    for (unsigned char c : piece) {
        word.push_back(b2u_[c]);
    }

    // 2.应用 BPE 合并
    // 如
    // "h" "e" "l" "l" "o" -> "he" "l" "l" "o"
    // "he" "l" "l" "o" -> "hel" "l" "o"
    // "hel" "l" "o" -> "hell" "o"
    // "hell" "o" -> break 跳出merge循环
    while (word.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_idx = word.size();

        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto it = merge_ranks_.find({word[i], word[i + 1]});
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }

        if (best_idx >= word.size()) {
            break;
        }

        word[best_idx] += word[best_idx + 1];
        word.erase(word.begin() + static_cast<std::ptrdiff_t>(best_idx) + 1);
    }

    // 3.获取对应的token_id
    for (const auto& w : word) {
        auto it = vocab_.find(w);
        out.push_back(it != vocab_.end() ? it->second : unk_id_);
    }
}

std::vector<int> BpeTokenizer::Encode(const std::string& text) const {
    if (text.empty()) {
        return {};
    }

    std::vector<int> result;
    size_t pos = 0;

    while (pos < text.size()) {
        // 先尝试 匹配特殊token
        // 匹配策略 最长匹配
        bool matched = false;
        for (const auto& st : special_tokens_) {
            const std::string& token_str = st.first;
            // 判断1 防止匹配当前token_str时越界
            // 判断2 从当前pos位置开始匹配当前token_str
            if (pos + token_str.size() <= text.size() &&
                std::memcmp(text.data() + pos, token_str.data(), token_str.size()) == 0) {
                result.push_back(st.second);
                pos += token_str.size();
                matched = true;
                break;
            }
        }
        if (matched) {
            continue;
        }

        // 找到下一个特殊token或终止位置
        size_t end = text.size();
        for (const auto& st : special_tokens_) {
            size_t p = text.find(st.first, pos);
            if (p != std::string::npos && p < end) {
                end = p;
            }
        }

        // 截取字段
        std::string segment = text.substr(pos, end - pos);
        pos = end;

        // 预切分后, 每个 piece 独立做 BPE, 合并不会跨越 piece 边界
        for (const std::string& piece : PreTokenize(segment)) {
            EncodePiece(piece, result);
        }
    }

    return result;
}

std::string BpeTokenizer::DecodeToken(int token) const {
    if (token >= 0 && token < static_cast<int>(id_to_token_.size())) {
        return id_to_token_[token];
    }
    return "";
}

std::string BpeTokenizer::Decode(const std::vector<int>& tokens) const {
    // 连接token字符串
    std::string text;
    for (int id : tokens) {
        text += DecodeToken(id);
    }

    std::string result;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t start = pos;
        std::string cp = NextUtf8Codepoint(text, pos);
        if (cp.empty()) {
            if (pos == start) {
                ++pos;
            }
            continue;
        }
        auto it = u2b_.find(cp);
        if (it != u2b_.end()) {
            result += static_cast<char>(it->second);
        } else {
            result += cp;
        }
    }

    return result;
}

std::unique_ptr<ITokenizer> CreateBpeTokenizer(
    const std::string& vocab_path,
    const std::string& merges_path,
    const std::string& special_path) {
    auto tok = std::make_unique<BpeTokenizer>();
    if (tok->Load(vocab_path, merges_path, special_path)) {
        return tok;
    }
    return nullptr;
}

}   // namespace mini_llama