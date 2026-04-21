#include <cassert>

namespace gpjson::index::kernels::orig {

__global__ void combined_escape_newline_index(const char *file, int fileSize,
                                              char *escapeCarryIndex,
                                              int *newlineCountIndex,
                                              long *escapeIndex,
                                              long *newlineIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int charsPerThread = (fileSize + stride - 1) / stride;
  int bitmapAlignedCharsPerThread = ((charsPerThread + 64 - 1) / 64) * 64;
  int start = index * bitmapAlignedCharsPerThread;
  int end = start + bitmapAlignedCharsPerThread;

  bool carry = index == 0 ? false : escapeCarryIndex[index - 1];

  long escape = 0;
  int escapeCount = 0;
  int totalCount = end - start;

  int newlineOffset = newlineCountIndex[index];

  for (long i = start; i < end && i < fileSize; i += 1) {
    char value = file[i];

    if (carry == 1) {
      escape |= (1L << (i % 64));
    }

    if (value == '\\') {
      escapeCount++;
      carry = carry ^ 1;
    } else {
      carry = 0;
    }

    if (value == '\n') {
      newlineIndex[newlineOffset++] = i;
    }

    if (i % 64 == 63) {
      escapeIndex[i / 64] = escape;
      escape = 0;
    }
  }

  if (fileSize <= end && (fileSize - 1) % 64 != 63L && fileSize - start > 0) {
    escapeIndex[(fileSize - 1) / 64] = escape;
  }

  assert(escapeCount != totalCount);
}

} // namespace gpjson::index::kernels::orig
