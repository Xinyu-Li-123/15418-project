#include <cassert>

namespace gpjson::index::kernels::sharemem {
namespace {
constexpr int kMaxNumLevels = 22;
}

__global__ void leveled_bitmaps_index(const char *file, int fileSize,
                                      const long *stringIndex,
                                      char *leveledBitmapsAuxIndex,
                                      long *leveledBitmapsIndex, int levelSize,
                                      int numLevels) {
  assert(numLevels <= kMaxNumLevels);

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int charsPerThread = (fileSize + stride - 1) / stride;
  int bitmapAlignedCharsPerThread = ((charsPerThread + 64 - 1) / 64) * 64;
  int start = index * bitmapAlignedCharsPerThread;
  int end = start + bitmapAlignedCharsPerThread;

  for (int i = start; i < end && i < levelSize * numLevels; i += 1) {
    for (int level = 0; level < numLevels; level += 1) {
      leveledBitmapsIndex[levelSize * level + i / 64] = 0;
    }
  }

  long string = 0;
  signed char level = leveledBitmapsAuxIndex[index];

  for (int i = start; i < end && i < fileSize; i += 1) {
    assert(level >= -1);

    long offsetInBlock = i % 64;

    if (offsetInBlock == 0) {
      string = stringIndex[i / 64];
    }

    if ((string & (1L << offsetInBlock)) != 0) {
      continue;
    }

    char value = file[i];

    if (value == '{' || value == '[') {
      level++;
      if (level < numLevels) {
        leveledBitmapsIndex[levelSize * level + i / 64] |=
            (1L << offsetInBlock);
      }
    } else if (value == '}' || value == ']') {
      if (level < numLevels) {
        leveledBitmapsIndex[levelSize * level + i / 64] |=
            (1L << offsetInBlock);
      }
      level--;
    } else if (value == ':' || value == ',') {
      if (level >= 0 && level < numLevels) {
        leveledBitmapsIndex[levelSize * level + i / 64] |=
            (1L << offsetInBlock);
      }
    }
  }
}

} // namespace gpjson::index::kernels::sharemem
