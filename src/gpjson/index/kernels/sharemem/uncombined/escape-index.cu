#include "gpjson/log/log.hpp"
#include <cassert>

namespace gpjson::index::kernels::sharemem {

__global__ void escape_index(const char *file, long fileSize,
                             char *escapeCarryIndex, long *escapeIndex) {
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

  for (long i = start; i < end && i < fileSize; i += 1) {
    if (carry == 1) {
      escape = escape | (1L << (i % 64));
    }

    if (file[i] == '\\') {
      escapeCount++;
      carry = carry ^ 1;
    } else {
      carry = 0;
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

__global__ void escape_index_sharemem(const char *file, long fileSize,
                                      long *escapeIndex) {
  constexpr int B = 1024;
  // constexpr int B = 16384;
  constexpr int HALO = 15;
  constexpr int WORD_BITS = 64;
  constexpr int WORDS_PER_BLOCK = B / WORD_BITS;

  Check(B % WORD_BITS == 0, "B % WORD_BITS != 0");
  Check(blockDim.x == B, "blockDim.x=%d, B=%d, blockDim.x != B", blockDim.x, B);

  if (blockDim.x != B) {
    printf("WTF?\n");
    return;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("blockDim.x=%d, B=%d, blockDim.x != B\n", blockDim.x, B);
  }

  __shared__ char file_chunk[B + HALO];
  __shared__ unsigned long long block_words[WORDS_PER_BLOCK];

  const int tid = threadIdx.x;
  const long numWords = (fileSize + WORD_BITS - 1) / WORD_BITS;
  auto *escapeIndex64 = reinterpret_cast<unsigned long long *>(escapeIndex);

  // Grid-stride over 1024-byte tiles.
  for (long base = static_cast<long>(blockIdx.x) * B; base < fileSize;
       base += static_cast<long>(gridDim.x) * B) {
    // Zero the per-block bitmap words.
    if (tid < WORDS_PER_BLOCK) {
      block_words[tid] = 0ULL;
    }

    // Load 15-byte left halo:
    // file_chunk[k] corresponds to global byte (base - HALO + k), for k in
    // [0,14].
    if (tid < HALO) {
      long g = base - HALO + tid;
      file_chunk[tid] = (g >= 0 && g < fileSize) ? file[g] : 0;
    }

    // Load the main 1024-byte tile:
    // file_chunk[HALO + tid] corresponds to global byte (base + tid).
    long i = base + tid;
    file_chunk[HALO + tid] = (i < fileSize) ? file[i] : 0;

    __syncthreads();

    if (i < fileSize) {
      bool carry = false;

      // Scan the 15 bytes immediately preceding file[i]:
      // file_chunk[tid + 0]  ... file_chunk[tid + 14]
      // correspond to file[i-15] ... file[i-1].
#pragma unroll
      for (int j = 0; j < HALO; ++j) {
        if (file_chunk[tid + j] == '\\') {
          carry ^= 1;
        } else {
          carry = 0;
        }
      }

      if (carry) {
        int localWord = tid >> 6; // tid / 64
        int localBit = tid & 63;  // tid % 64
        atomicOr(&block_words[localWord], 1ULL << localBit);
      }
    }

    __syncthreads();

    // Write out the 16 64-bit words for this 1024-byte tile.
    if (tid < WORDS_PER_BLOCK) {
      long wordIndex = (base >> 6) + tid; // base / 64 + tid
      if (wordIndex < numWords) {
        escapeIndex64[wordIndex] = block_words[tid];
      }
    }

    __syncthreads();
  }
}

} // namespace gpjson::index::kernels::sharemem
