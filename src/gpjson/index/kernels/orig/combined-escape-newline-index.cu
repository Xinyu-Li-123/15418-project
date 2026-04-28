#include <cassert>
#include <cstddef>

namespace gpjson::index::kernels::orig {

__global__ void combined_escape_newline_index(const char *file, size_t fileSize,
                                              char *escapeCarryIndex,
                                              int *newlineCountIndex,
                                              long *escapeIndex,
                                              long *newlineIndex) {
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

  int newlineOffset = newlineCountIndex[index];

  for (size_t i = start; i < end && i < fileSize; i += 1) {
    char value = file[i];
    const int offsetInBlock = static_cast<int>(i % 64);

    if (carry == 1) {
      escape |= (1L << offsetInBlock);
    }

    if (value == '\\') {
      escapeCount++;
      carry = carry ^ 1;
    } else {
      carry = 0;
    }

    if (value == '\n') {
      newlineIndex[newlineOffset++] = static_cast<long>(i);
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
