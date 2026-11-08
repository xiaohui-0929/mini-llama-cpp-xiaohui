#include "test_main.h"
#include "test_names.h"

static bool TestRadixTreeEmptyMiss() {
    RadixTree tree;
    MINI_LLAMA_ASSERT_TRUE(tree.empty());
    MINI_LLAMA_ASSERT_EQ(tree.size(), 0);
    MINI_LLAMA_ASSERT_EQ(tree.LongestPrefix({1, 2, 3}), 0);
    return true;
}

static bool TestRadixTreeExactMatch() {
    RadixTree tree;
    tree.Insert({1, 2, 3});
    MINI_LLAMA_ASSERT_TRUE(!tree.empty());
    MINI_LLAMA_ASSERT_EQ(tree.size(), 1);
    MINI_LLAMA_ASSERT_EQ(tree.LongestPrefix({1, 2, 3}), 3);
    return true;
}

static bool TestRadixTreeQueryIsStoredPrefix() {
    RadixTree tree;
    tree.Insert({1, 2, 3, 4});
    MINI_LLAMA_ASSERT_EQ(tree.LongestPrefix({1, 2}), 2);
    return true;  
}

static bool TestRadixTreeStoredSequenceIsQueryPrefix() {
    RadixTree tree;
    tree.Insert({1, 2});
    MINI_LLAMA_ASSERT_EQ(tree.LongestPrefix({1, 2, 3, 4}), 2);
    return true;
}

static bool TestRadixTreeSplitOnBranch() {
    RadixTree tree;
    tree.Insert({1, 2, 3, 4});
    tree.Insert({1, 2, 9});
    MINI_LLAMA_ASSERT_EQ(tree.size(), 2);
    MINI_LLAMA_ASSERT_EQ(tree.LongestPrefix({1, 8}), 1);
    MINI_LLAMA_ASSERT_EQ(tree.LongestPrefix({1, 2, 3}), 3);
    MINI_LLAMA_ASSERT_EQ(tree.LongestPrefix({1, 2, 9, 8}), 3);
    MINI_LLAMA_ASSERT_EQ(tree.LongestPrefix({1, 2, 8}), 2);
    return true;
}

static bool TestRadixTreeSplitWithInsertedMiddlePrefix() {
    RadixTree tree;
    tree.Insert({1, 2, 3, 4, 5});
    tree.Insert({1, 2, 3});
    tree.Insert({1, 2, 3, 9});
    MINI_LLAMA_ASSERT_EQ(tree.size(), 3);
    MINI_LLAMA_ASSERT_EQ(tree.LongestPrefix({1, 2, 3, 4}), 4);
    MINI_LLAMA_ASSERT_EQ(tree.LongestPrefix({1, 2, 3, 9, 8}), 4);
    MINI_LLAMA_ASSERT_EQ(tree.LongestPrefix({1, 2, 3, 8}), 3);
    MINI_LLAMA_ASSERT_EQ(tree.LongestPrefix({1, 2, 8}), 2);
    return true;
}

static bool TestRadixTreeDuplicateInsertStable() {
    RadixTree tree;
    tree.Insert({4, 5, 6});
    tree.Insert({4, 5, 6});
    MINI_LLAMA_ASSERT_EQ(tree.size(), 1);
    MINI_LLAMA_ASSERT_EQ(tree.LongestPrefix({4, 5, 6, 7}), 3);
    return true;
}

static bool TestRadixTreeClear() {
    RadixTree tree;
    tree.Insert({1, 2, 3});
    tree.Clear();
    MINI_LLAMA_ASSERT_TRUE(tree.empty());
    MINI_LLAMA_ASSERT_EQ(tree.size(), 0);
    MINI_LLAMA_ASSERT_EQ(tree.LongestPrefix({1, 2, 3}), 0);
    return true;
}

static bool TestRadixTreeEmptyInsert() {
    RadixTree tree;
    try {
        tree.Insert({});
        MINI_LLAMA_ASSERT_FAIL("expected exception for radix tree insert empty");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

static struct RadixTreeTestRegistrar {
    RadixTreeTestRegistrar() {
        RegisterTest("radix_tree_empty_miss", 
                    TestRadixTreeEmptyMiss);
        RegisterTest("radix_tree_exact_match", 
                    TestRadixTreeExactMatch);
        RegisterTest("radix_tree_query_is_stored_prefix",
                    TestRadixTreeQueryIsStoredPrefix);
        RegisterTest("radix_tree_stored_sequence_is_query_prefix",
                    TestRadixTreeStoredSequenceIsQueryPrefix);
        RegisterTest("radix_tree_split_on_branch", 
                    TestRadixTreeSplitOnBranch);
        RegisterTest("radix_tree_split_with_inserted_middle_prefix", 
                    TestRadixTreeSplitWithInsertedMiddlePrefix);
        RegisterTest("radix_tree_duplicate_insert_stable",
                    TestRadixTreeDuplicateInsertStable);
        RegisterTest("radix_tree_clear", 
                    TestRadixTreeClear);
        RegisterTest("radix_tree_empty_insert", 
                    TestRadixTreeEmptyInsert);
    }
} radix_tree_test_registrar;