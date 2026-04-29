#include <cassert>
#include <cstddef>

namespace gpjson::index::kernels::sharemem {

__global__ void combined_escape_newline_index(const char *file, size_t fileSize,
                                              char *escapeCarryIndex,
                                              int *newlineCountIndex,
                                              long *escapeIndex,
                                              long *newlineIndex) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  size_t charsPerThread = (fileSize + stride - 1) / stride;
  size_t bitmapAlignedCharsPerThread = ((charsPerThread + 64 - 1) / 64) * 64;
  size_t start = index * bitmapAlignedCharsPerThread;
  size_t end = start + bitmapAlignedCharsPerThread;

  bool carry = index == 0 ? false : escapeCarryIndex[index - 1];

  int escapeCount = 0;
  size_t totalCount = end - start;

  int newlineOffset = newlineCountIndex[index];

  for (size_t i = start; i < end && i < fileSize; i += 1) {
    char value = file[i];

    if (carry == 1) {
      escapeIndex[i / 64] |= (1L << (i % 64));
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
  }

  assert(static_cast<size_t>(escapeCount) != totalCount);
}

} // namespace gpjson::index::kernels::sharemem
