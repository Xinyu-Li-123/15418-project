#include <cassert>
#include <cstddef>

namespace gpjson::index::kernels::orig {

__global__ void escape_index(const char *file, size_t fileSize,
                             char *escapeCarryIndex, long *escapeIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  size_t charsPerThread =
      (fileSize + static_cast<size_t>(stride) - 1) / stride;
  size_t bitmapAlignedCharsPerThread =
      ((charsPerThread + 64 - 1) / 64) * 64;
  size_t start = static_cast<size_t>(index) * bitmapAlignedCharsPerThread;
  size_t end = start + bitmapAlignedCharsPerThread;

  bool carry = index == 0 ? false : escapeCarryIndex[index - 1];

  long escape = 0;

  int escapeCount = 0;
  size_t totalCount = end - start;

  for (size_t i = start; i < end && i < fileSize; i += 1) {
    const int offsetInBlock = static_cast<int>(i % 64);
    if (carry == 1) {
      escape = escape | (1L << offsetInBlock);
    }

    if (file[i] == '\\') {
      escapeCount++;
      carry = carry ^ 1;
    } else {
      carry = 0;
    }

    if (offsetInBlock == 63) {
      escapeIndex[i / 64] = escape;
      escape = 0;
    }
  }

  if (fileSize > 0 && start < fileSize && fileSize <= end &&
      (fileSize - 1) % 64 != 63L) {
    escapeIndex[(fileSize - 1) / 64] = escape;
  }

  assert(static_cast<size_t>(escapeCount) != totalCount);
}

} // namespace gpjson::index::kernels::orig
