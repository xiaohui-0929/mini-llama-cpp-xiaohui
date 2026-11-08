#include "mini_llama/runtime/radix_tree.h"

#include <stdexcept>

namespace mini_llama {

void RadixTree::Clear() {
    root_.key.clear();
    root_.terminal = false;
    root_.children.clear();
    terminal_count_ = 0;
}

void RadixTree::Insert(const std::vector<int>& tokens) {
    if (tokens.empty()) {
        throw std::runtime_error("insert empty tokens into radix tree");
    }
    InsertInto(&root_, tokens);
}

size_t RadixTree::LongestPrefix(const std::vector<int>& tokens) const {
    if (tokens.empty()) {
        return 0;
    }

    const Node* node = &root_;
    size_t pos = 0;
    while (pos < tokens.size()) {
        auto child_it = node->children.find(tokens[pos]);
        if (child_it == node->children.end()) {
            return pos;
        }

        const Node* child = child_it->second.get();
        size_t common = CommonPrefixLength(tokens, pos, child->key);
        pos += common;
        if (common < child->key.size()) {
            return pos;
        }
        node = child;
    }
    
    return pos;
}

void RadixTree::InsertInto(Node* parent, std::vector<int> suffix) {
    auto child_it = parent->children.find(suffix[0]);
    // 插入序列没有匹配前缀的情况
    if (child_it == parent->children.end()) {
        auto child = std::make_unique<Node>();
        child->key = std::move(suffix);
        child->terminal = true;
        parent->children[child->key[0]] = std::move(child);
        ++terminal_count_;
        return;
    }
    // 存在匹配前缀的情况
    std::unique_ptr<Node>& child = child_it->second;
    // 判断插入序列和匹配序列之间的关系
    size_t common = CommonPrefixLength(suffix, 0, child->key);
    // 如果和匹配序列 完全匹配
    if (common == child->key.size()) {
        // 情况1 插入序列和匹配序列完全一致
        if (common == suffix.size()) {
            if (!child->terminal) {
                child->terminal = true;
                ++terminal_count_;
            }
            return;
        }
        // 情况2 插入序列比匹配序列长 截取剩下的序列插入
        std::vector<int> remaining(suffix.begin() + static_cast<long>(common),
                                   suffix.end());
        InsertInto(child.get(), std::move(remaining));
        return;
    }
    // 如果和匹配序列 不完全匹配 将匹配序列进行切分
    auto split = std::make_unique<Node>();
    split->key.assign(child->key.begin(),
                      child->key.begin() + static_cast<long>(common));
    child->key.erase(child->key.begin(),
                     child->key.begin() + static_cast<long>(common));
    int old_child_first = child->key[0];
    split->children[old_child_first] = std::move(child);
    // 情况3 插入序列比匹配序列短 匹配序列被切分
    if (common == suffix.size()) {
        split->terminal = true;
        ++terminal_count_;
    } else {
    // 情况4 插入序列和匹配序列之间共享split前缀 将插入序列的后半段作为子序列插入
        auto child_from_suffix = std::make_unique<Node>();
        child_from_suffix->key.assign(suffix.begin() + static_cast<long>(common),
                                      suffix.end());
        child_from_suffix->terminal = true;
        ++terminal_count_;
        int new_child_first = child_from_suffix->key[0];
        split->children[new_child_first] = std::move(child_from_suffix);
    }

    child_it->second = std::move(split);
}

size_t RadixTree::CommonPrefixLength(const std::vector<int>& a, size_t a_offset,
                                 const std::vector<int>& b) {
    size_t result = 0;
    if (a.empty() || b.empty()) {
        return result;
    }

    while (result < b.size() && result + a_offset < a.size()) {
        if (a[result + a_offset] != b[result]) {
            break;
        }
        ++result;
    }

    return result;
}

}   // namespace mini_llama