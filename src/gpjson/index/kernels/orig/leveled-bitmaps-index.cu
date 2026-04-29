#include <cassert>
#include <cstddef>

namespace gpjson::index::kernels::orig {
namespace {
constexpr int kMaxNumLevels = 22;
}

__global__ void leveled_bitmaps_index(const char *file, size_t fileSize,
                                      const long *stringIndex,
                                      char *leveledBitmapsAuxIndex,
                                      long *leveledBitmapsIndex, int levelSize,
                                      int numLevels) {
  assert(numLevels <= kMaxNumLevels);

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  size_t charsPerThread =
      (fileSize + static_cast<size_t>(stride) - 1) / stride;
  size_t bitmapAlignedCharsPerThread =
      ((charsPerThread + 64 - 1) / 64) * 64;
  size_t start = static_cast<size_t>(index) * bitmapAlignedCharsPerThread;
  size_t end = start + bitmapAlignedCharsPerThread;

  signed char level = leveledBitmapsAuxIndex[index];

  for (size_t blockStart = start; blockStart < end && blockStart < fileSize;
       blockStart += 64) {
    const size_t wordIndex = blockStart / 64;
    const long string = stringIndex[wordIndex];
    // Accumulate a full 64-bit output word locally, then write each level once.
    long structuralBitmaps[kMaxNumLevels];
    for (int bitmapLevel = 0; bitmapLevel < numLevels; bitmapLevel += 1) {
      structuralBitmaps[bitmapLevel] = 0;
    }

    const size_t blockEnd =
        blockStart + 64 < fileSize ? blockStart + 64 : fileSize;
    for (size_t i = blockStart; i < blockEnd; i += 1) {
      assert(level >= -1);

      const int offsetInBlock = static_cast<int>(i % 64);
      const long bit = 1L << offsetInBlock;
      if ((string & bit) != 0) {
        continue;
      }

      const char value = file[i];
      if (value == '{' || value == '[') {
        level++;
        if (level >= 0 && level < numLevels) {
          structuralBitmaps[level] |= bit;
        }
      } else if (value == '}' || value == ']') {
        if (level >= 0 && level < numLevels) {
          structuralBitmaps[level] |= bit;
        }
        level--;
      } else if (value == ':' || value == ',') {
        if (level >= 0 && level < numLevels) {
          structuralBitmaps[level] |= bit;
        }
      }
    }

    for (int bitmapLevel = 0; bitmapLevel < numLevels; bitmapLevel += 1) {
      leveledBitmapsIndex[static_cast<size_t>(levelSize) * bitmapLevel +
                          wordIndex] =
          structuralBitmaps[bitmapLevel];
    }
  }
}

} // namespace gpjson::index::kernels::orig
