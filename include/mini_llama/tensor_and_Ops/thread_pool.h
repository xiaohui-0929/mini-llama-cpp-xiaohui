#ifndef INCLUDE_MINI_LLAMA_THREAD_POOL_H_
#define INCLUDE_MINI_LLAMA_THREAD_POOL_H_

#include <functional>

namespace mini_llama {

// 获取并行操作的线程数
// 默认为std::thread::hardware_concurrency()
int GetThreadCount();

// 设置并行操作的线程数
void SetThreadCount(int n);

// 简易并行操作
// 将 [0, n) 区间均分为大致相等的组块
void ParallelFor(int n, const std::function<void(int begin, int end)>& fn);

}   // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_THREAD_POOL_H_