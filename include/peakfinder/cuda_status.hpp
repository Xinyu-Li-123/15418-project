#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace peakfinder {

inline void check_cuda(cudaError_t status, const char *expr, const char *file,
                       int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") +
                             cudaGetErrorString(status) + " | expr=" + expr +
                             " | file=" + file +
                             " | line=" + std::to_string(line));
  }
}

} // namespace peakfinder

#define PF_CUDA_CHECK(expr)                                                    \
  ::peakfinder::check_cuda((expr), #expr, __FILE__, __LINE__)
