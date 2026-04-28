#include <cstddef>

namespace gpjson::index::kernels::orig {

__global__ void newline_index(const char *file, size_t fileSize,
                              int *newlineCountIndex, long *newlineIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int offset = newlineCountIndex[index];

  size_t charsPerThread =
      (fileSize + static_cast<size_t>(stride) - 1) / stride;
  size_t start = static_cast<size_t>(index) * charsPerThread;
  size_t end = start + charsPerThread;

  for (size_t i = start; i < end && i < fileSize; i += 1) {
    if (file[i] == '\n') {
      newlineIndex[offset++] = static_cast<long>(i);
    }
  }
}

} // namespace gpjson::index::kernels::orig
