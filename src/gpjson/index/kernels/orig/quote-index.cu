#include <cstddef>

namespace gpjson::index::kernels::orig {

__global__ void quote_index(const char *file, size_t fileSize,
                            long *escapeIndex,
                            long *quoteIndex, char *quoteCarryIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  size_t charsPerThread =
      (fileSize + static_cast<size_t>(stride) - 1) / stride;
  size_t bitmapAlignedCharsPerThread =
      ((charsPerThread + 64 - 1) / 64) * 64;
  size_t start = static_cast<size_t>(index) * bitmapAlignedCharsPerThread;
  size_t end = start + bitmapAlignedCharsPerThread;

  long escaped = 0;
  long quote = 0;
  int quoteCount = 0;

  for (size_t i = start; i < end && i < fileSize; i += 1) {
    int offsetInBlock = static_cast<int>(i % 64);

    if (offsetInBlock == 0) {
      escaped = escapeIndex[i / 64];
    }

    if (file[i] == '"') {
      if ((escaped & (1L << offsetInBlock)) == 0) {
        quote = quote | (1L << offsetInBlock);
        quoteCount++;
      }
    }

    if (offsetInBlock == 63) {
      quoteIndex[i / 64] = quote;
      quote = 0;
    }
  }

  if (fileSize > 0 && start < fileSize && fileSize <= end &&
      (fileSize - 1) % 64 != 63L) {
    quoteIndex[(fileSize - 1) / 64] = quote;
  }

  quoteCarryIndex[index] = quoteCount & 1;
}

} // namespace gpjson::index::kernels::orig
