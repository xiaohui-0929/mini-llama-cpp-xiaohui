cmake_minimum_required(VERSION 3.17)
project(mini-llama CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
)

# --- Main executable ---
set(MAIN_SOURCES
    ${MINI_LLAMA_CORE_SOURCES}
    src/main.cc
)

add_executable(mini-llama ${MAIN_SOURCES})
set_target_properties(mini-llama PROPERTIES OUTPUT_NAME mini-llama)

# --- Test executable ---
set(TEST_SOURCES
    ${MINI_LLAMA_CORE_SOURCES}
    tests/test_main.cc
    tests/test_tensor.cc
    tests/test_ops.cc
)

add_executable(mini-llama-tests ${TEST_SOURCES})
set_target_properties(mini-llama-tests PROPERTIES OUTPUT_NAME mini-llama-tests)
