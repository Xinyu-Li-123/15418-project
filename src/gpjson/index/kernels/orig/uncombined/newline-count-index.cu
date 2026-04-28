#include <cstddef>

namespace gpjson::index::kernels::orig {

__global__ void newline_count_index(const char *file, size_t fileSize,
                                    int *newlineCountIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  size_t charsPerThread =
      (fileSize + static_cast<size_t>(stride) - 1) / stride;
  size_t start = static_cast<size_t>(index) * charsPerThread;
  size_t end = start + charsPerThread;

  int count = 0;
  for (size_t i = start; i < end && i < fileSize; i += 1) {
    if (file[i] == '\n') {
      count += 1;
    }
  }

  newlineCountIndex[index] = count;
}
} // namespace gpjson::index::kernels::orig
