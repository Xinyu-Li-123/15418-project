#include <cstddef>

namespace gpjson::index::kernels::orig {

__global__ void leveled_bitmaps_carry_index(const char *file, size_t fileSize,
                                            const long *stringIndex,
                                            char *leveledBitmapsAuxIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  size_t charsPerThread =
      (fileSize + static_cast<size_t>(stride) - 1) / stride;
  size_t bitmapAlignedCharsPerThread =
      ((charsPerThread + 64 - 1) / 64) * 64;
  size_t start = static_cast<size_t>(index) * bitmapAlignedCharsPerThread;
  size_t end = start + bitmapAlignedCharsPerThread;

  long string = 0;
  signed char level = 0;

  for (size_t i = start; i < end && i < fileSize; i += 1) {
    int offsetInBlock = static_cast<int>(i % 64);

    if (offsetInBlock == 0) {
      string = stringIndex[i / 64];
    }

    if ((string & (1L << offsetInBlock)) != 0) {
      continue;
    }

    char value = file[i];

    if (value == '{' || value == '[') {
      level++;
    } else if (value == '}' || value == ']') {
      level--;
    }
  }

  leveledBitmapsAuxIndex[index] = level;
}

} // namespace gpjson::index::kernels::orig
