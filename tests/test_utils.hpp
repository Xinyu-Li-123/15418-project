#pragma once

#include "gpjson/cuda/cuda.hpp"

#include <gtest/gtest.h>

namespace gpjson::test {

inline bool cuda_device_available() {
  static const bool available = gpjson::cuda::device_available();
  return available;
}

} // namespace gpjson::test

#define GPJSON_SKIP_IF_CUDA_UNAVAILABLE()                                      \
  do {                                                                         \
    if (!::gpjson::test::cuda_device_available()) {                            \
      GTEST_SKIP() << "CUDA device unavailable";                               \
    }                                                                          \
  } while (false)
