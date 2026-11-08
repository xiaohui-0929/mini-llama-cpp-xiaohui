#ifndef INCLUDE_MINI_LLAMA_RADIX_TREE_H_
#define INCLUDE_MINI_LLAMA_RADIX_TREE_H_

#include <vector>
#include <map>
#include <memory>

namespace mini_llama {

class RadixTree {
public:
    RadixTree() = default;
    ~RadixTree() = default;

    RadixTree(const RadixTree&) = delete;
    RadixTree& operator=(const RadixTree&) = delete;
    RadixTree(RadixTree&&) noexcept = default;
    RadixTree& operator=(RadixTree&&) noexcept = default;

    void Clear();
    bool empty() const { return terminal_count_ == 0; }
    size_t size() const { return terminal_count_; }

    // 将传入的token序列插入到redix tree
    void Insert(const std::vector<int>& tokens);
    // 获取传入的token序列和当前redix tree的最长匹配前缀
    size_t LongestPrefix(const std::vector<int>& tokens) const;

private:
    struct Node {
        std::vector<int> key;                           // 当前节点代表的一段token
        bool terminal = false;                          // 是否为一段完整序列的结尾
        std::map<int, std::unique_ptr<Node>> children;  // key: 下一个token value: 对应节点
    };

    Node root_;
    size_t terminal_count_ = 0;

private:
    // 插入节点的实际实现
    void InsertInto(Node* parent, std::vector<int> suffix);
    // 在序列a中偏移a_offset的位置开始 与 序列b 相同的个数
    static size_t CommonPrefixLength(const std::vector<int>& a, size_t a_offset,
                                     const std::vector<int>& b);

};

}   // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_RADIX_TREE_H_