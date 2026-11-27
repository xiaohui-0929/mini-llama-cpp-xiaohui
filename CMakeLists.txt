cmake_minimum_required(VERSION 3.17)
project(mini-llama CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# clang 默认省略标准库类型的完整调试信息，会导致 gdb 里 std::string 显示为
# <incomplete type>。仅在 Debug 构建下开启完整调试信息。
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options($<$<CONFIG:Debug>:-fstandalone-debug>)
endif()

# --- 编译警告 ---
# -Wmissing-declarations: 定义在源文件里的非 static 函数必须有对应声明，
#                         可提前暴露头文件声明与定义的签名/名字不一致（如拼写错误）。
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(MINI_LLAMA_WARNINGS
        -Wall
        -Wextra
        -Wmissing-declarations
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wnull-dereference
        -Wdouble-promotion
    )
endif()

# 生成 compile_commands.json，供 clangd / IDE 补全使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- include ---
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/tests)

# --- Core library ---
set(MINI_LLAMA_CORE_SOURCES
    src/tensor_and_Ops/tensor.cc
    src/tensor_and_Ops/ops.cc
    src/tensor_and_Ops/thread_pool.cc
    src/tensor_and_Ops/matmul_optimized.cc
    src/tokenizer/tokenizer.cc
    src/tokenizer/json_vocab_tokenizer.cc
    src/tokenizer/bpe_tokenizer.cc
    src/runtime/kv_cache.cc
    src/runtime/context.cc
    src/runtime/radix_tree.cc
    src/runtime/request_context.cc
    src/model/loader.cc
    src/forward/forward.cc
    src/forward/batch.cc
    src/sampler/sampler.cc
    src/chat/chat.cc
    src/chat/prompt_builder.cc
    src/chat/terminal.cc
    src/CLI/parsing_helper.cc
    src/CLI/run_mode.cc
    src/CLI/generate_mode.cc
    src/CLI/inspect_gguf_mode.cc
    src/gguf/gguf.cc
)

# --- Main executable ---
set(MAIN_SOURCES
    ${MINI_LLAMA_CORE_SOURCES}
    src/main.cc
)

add_executable(mini-llama ${MAIN_SOURCES})
set_target_properties(mini-llama PROPERTIES OUTPUT_NAME mini-llama)
target_compile_options(mini-llama PRIVATE ${MINI_LLAMA_WARNINGS})

# --- Test executable ---
set(TEST_SOURCES
    ${MINI_LLAMA_CORE_SOURCES}
    tests/test_main.cc
    tests/test_tensor.cc
    tests/test_ops.cc
    tests/test_tokenizer.cc
    tests/test_kv_cache.cc
    tests/test_radix_tree.cc
    tests/test_request_context.cc
    tests/test_loader.cc
    tests/test_batch.cc
    tests/test_forward.cc
    tests/test_sampler.cc
    tests/test_chat.cc
    tests/test_gguf.cc
)

add_executable(mini-llama-tests ${TEST_SOURCES})
set_target_properties(mini-llama-tests PROPERTIES OUTPUT_NAME mini-llama-tests)
target_compile_options(mini-llama-tests PRIVATE ${MINI_LLAMA_WARNINGS})

# 测试需要读取 models/ 下的资源文件。把源码根目录注入为宏，
# 避免用例依赖运行测试时的当前工作目录（从 build/ 下跑也能找到文件）。
target_compile_definitions(mini-llama-tests PRIVATE
    MINI_LLAMA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# 支持 ctest 运行，工作目录固定为源码根
enable_testing()
add_test(NAME mini_llama_tests
         COMMAND mini-llama-tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
