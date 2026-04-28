#include <cstddef>

namespace gpjson::index::kernels::orig {

__global__ void escape_carry_index(const char *file, size_t fileSize,
                                   char *escapeCarryIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  size_t charsPerThread =
      (fileSize + static_cast<size_t>(stride) - 1) / stride;
  size_t bitmapAlignedCharsPerThread =
      ((charsPerThread + 64 - 1) / 64) * 64;
  size_t start = static_cast<size_t>(index) * bitmapAlignedCharsPerThread;
  size_t end = start + bitmapAlignedCharsPerThread;

  char carry = 0;

  for (size_t i = start; i < end && i < fileSize; i += 1) {
    if (file[i] == '\\') {
      carry = 1 ^ carry;
    } else {
      carry = 0;
    }
  }

  escapeCarryIndex[index] = carry;
}

} // namespace gpjson::index::kernels::orig
