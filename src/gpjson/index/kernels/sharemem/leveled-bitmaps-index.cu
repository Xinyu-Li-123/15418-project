#include "gpjson/log/log.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace gpjson::index::kernels::sharemem {
namespace {
constexpr int kMaxNumLevels = 22;
}

__device__ void leveled_bitmaps_index_orig(const char *file, size_t fileSize,
                                           const long *stringIndex,
                                           char *leveledBitmapsAuxIndex,
                                           long *leveledBitmapsIndex,
                                           int levelSize, int numLevels) {
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

__device__ void leveled_bitmaps_index_sharemem_transposed_packed(
    const char *file, size_t fileSize, const long *stringIndex,
    const char *leveledBitmapsAuxIndex, long *leveledBitmapsIndex,
    int levelSize, int numLevels) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = THREADS_PER_BLOCK * BYTES_PER_THREAD;

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int PACK_BYTES = 8;
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;
  constexpr int PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / PACK_BYTES;

  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % PACK_BYTES == 0);
  static_assert(sizeof(uint2) == PACK_BYTES);

  const int tid = threadIdx.x;
  const int tile_idx = blockIdx.x;
  const size_t block_start = static_cast<size_t>(tile_idx) * CHUNK_SIZE;
  const size_t global_thread_id =
      static_cast<size_t>(tile_idx) * THREADS_PER_BLOCK + tid;
  const size_t thread_global_base = global_thread_id * BYTES_PER_THREAD;
  const size_t word_idx = thread_global_base / BYTES_PER_THREAD;

  __shared__ uint2
      smem_packed_bytes[PACKED_GROUPS_PER_THREAD * THREADS_PER_BLOCK];

  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  for (int p = tid; p < PACKED_ELEMS_PER_BLOCK; p += blockDim.x) {
    const size_t global_byte_idx =
        block_start + static_cast<size_t>(p) * PACK_BYTES;

    uint2 packed_bytes = make_uint2(0u, 0u);

    if (global_byte_idx + PACK_BYTES <= fileSize) {
      const size_t global_packed_idx = global_byte_idx / PACK_BYTES;
      packed_bytes = file_packed_gmem[global_packed_idx];
    } else if (global_byte_idx < fileSize) {
      unsigned char tmp[PACK_BYTES] = {0};

      // #pragma unroll
      for (int i = 0; i < PACK_BYTES; ++i) {
        const size_t global_idx = global_byte_idx + i;
        if (global_idx < fileSize) {
          tmp[i] = static_cast<unsigned char>(file[global_idx]);
        }
      }

      unsigned char *packed_chars =
          reinterpret_cast<unsigned char *>(&packed_bytes);

      // #pragma unroll
      for (int i = 0; i < PACK_BYTES; ++i) {
        packed_chars[i] = tmp[i];
      }
    }

    const int owner_tid = p / PACKED_GROUPS_PER_THREAD;
    const int packed_group = p % PACKED_GROUPS_PER_THREAD;
    const int smem_idx = packed_group * THREADS_PER_BLOCK + owner_tid;

    smem_packed_bytes[smem_idx] = packed_bytes;
  }

  __syncthreads();

  if (word_idx < static_cast<size_t>(levelSize)) {
    for (int level = 0; level < numLevels; level += 1) {
      leveledBitmapsIndex[static_cast<size_t>(levelSize) * level + word_idx] =
          0;
    }
  }

  if (thread_global_base >= fileSize) {
    return;
  }

  long bitmap_words[kMaxNumLevels];

  // #pragma unroll
  for (int level = 0; level < kMaxNumLevels; level += 1) {
    bitmap_words[level] = 0;
  }

  const long string = stringIndex[word_idx];
  signed char level = leveledBitmapsAuxIndex[global_thread_id];

  for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
    const int smem_idx = group * THREADS_PER_BLOCK + tid;
    const uint2 packed_bytes_word = smem_packed_bytes[smem_idx];
    const unsigned char *packed_bytes =
        reinterpret_cast<const unsigned char *>(&packed_bytes_word);

    // #pragma unroll
    for (int i = 0; i < PACK_BYTES; ++i) {
      const int byte_offset = group * PACK_BYTES + i;
      const size_t global_idx = thread_global_base + byte_offset;

      if (global_idx >= fileSize) {
        break;
      }

      assert(level >= -1);

      if ((string & (1L << byte_offset)) != 0) {
        continue;
      }

      const unsigned char value = packed_bytes[i];

      if (value == static_cast<unsigned char>('{') ||
          value == static_cast<unsigned char>('[')) {
        level++;
        if (level >= 0 && level < numLevels) {
          bitmap_words[level] |= 1L << byte_offset;
        }
      } else if (value == static_cast<unsigned char>('}') ||
                 value == static_cast<unsigned char>(']')) {
        if (level >= 0 && level < numLevels) {
          bitmap_words[level] |= 1L << byte_offset;
        }
        level--;
      } else if (value == static_cast<unsigned char>(':') ||
                 value == static_cast<unsigned char>(',')) {
        if (level >= 0 && level < numLevels) {
          bitmap_words[level] |= 1L << byte_offset;
        }
      }
    }
  }

  if (word_idx < static_cast<size_t>(levelSize)) {
    for (int level = 0; level < numLevels; level += 1) {
      leveledBitmapsIndex[static_cast<size_t>(levelSize) * level + word_idx] =
          bitmap_words[level];
    }
  }
}

__device__ void leveled_bitmaps_index_sharemem_2(
    const char *file, size_t fileSize, const long *stringIndex,
    const char *leveledBitmapsAuxIndex, long *leveledBitmapsIndex,
    int levelSize, int numLevels) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = THREADS_PER_BLOCK * BYTES_PER_THREAD;

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int PACK_BYTES = 8;
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;
  constexpr int PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / PACK_BYTES;

  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % PACK_BYTES == 0);
  static_assert(sizeof(uint2) == PACK_BYTES);

  const int tid = threadIdx.x;
  const int tile_idx = blockIdx.x;
  const size_t block_start = static_cast<size_t>(tile_idx) * CHUNK_SIZE;
  const size_t global_thread_id =
      static_cast<size_t>(tile_idx) * THREADS_PER_BLOCK + tid;
  const size_t thread_global_base = global_thread_id * BYTES_PER_THREAD;
  const size_t word_idx = thread_global_base / BYTES_PER_THREAD;

  __shared__ uint2
      smem_packed_bytes[PACKED_GROUPS_PER_THREAD * THREADS_PER_BLOCK];

  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  for (int p = tid; p < PACKED_ELEMS_PER_BLOCK; p += blockDim.x) {
    const size_t global_byte_idx =
        block_start + static_cast<size_t>(p) * PACK_BYTES;

    uint2 packed_bytes = make_uint2(0u, 0u);

    if (global_byte_idx + PACK_BYTES <= fileSize) {
      const size_t global_packed_idx = global_byte_idx / PACK_BYTES;
      packed_bytes = file_packed_gmem[global_packed_idx];
    } else if (global_byte_idx < fileSize) {
      unsigned char tmp[PACK_BYTES] = {0};

#pragma unroll
      for (int i = 0; i < PACK_BYTES; ++i) {
        const size_t global_idx = global_byte_idx + i;
        if (global_idx < fileSize) {
          tmp[i] = static_cast<unsigned char>(file[global_idx]);
        }
      }

      unsigned char *packed_chars =
          reinterpret_cast<unsigned char *>(&packed_bytes);

#pragma unroll
      for (int i = 0; i < PACK_BYTES; ++i) {
        packed_chars[i] = tmp[i];
      }
    }

    const int owner_tid = p / PACKED_GROUPS_PER_THREAD;
    const int packed_group = p % PACKED_GROUPS_PER_THREAD;
    const int smem_idx = packed_group * THREADS_PER_BLOCK + owner_tid;

    smem_packed_bytes[smem_idx] = packed_bytes;
  }

  __syncthreads();

  if (word_idx < static_cast<size_t>(levelSize)) {
    if (numLevels > 0) {
      leveledBitmapsIndex[static_cast<size_t>(levelSize) * 0 + word_idx] = 0;
    }
    if (numLevels > 1) {
      leveledBitmapsIndex[static_cast<size_t>(levelSize) * 1 + word_idx] = 0;
    }
  }

  if (thread_global_base >= fileSize) {
    return;
  }

  long bitmap0 = 0;
  long bitmap1 = 0;

  const long string = stringIndex[word_idx];
  signed char level = leveledBitmapsAuxIndex[global_thread_id];

#pragma unroll
  for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
    const int smem_idx = group * THREADS_PER_BLOCK + tid;
    const uint2 packed_bytes_word = smem_packed_bytes[smem_idx];
    const unsigned char *packed_bytes =
        reinterpret_cast<const unsigned char *>(&packed_bytes_word);

#pragma unroll
    for (int i = 0; i < PACK_BYTES; ++i) {
      const int byte_offset = group * PACK_BYTES + i;
      const size_t global_idx = thread_global_base + byte_offset;

      if (global_idx >= fileSize) {
        break;
      }

      assert(level >= -1);

      if ((string & (1L << byte_offset)) != 0) {
        continue;
      }

      const unsigned char value = packed_bytes[i];
      const long bit = 1L << byte_offset;

      if (value == static_cast<unsigned char>('{') ||
          value == static_cast<unsigned char>('[')) {
        level++;
        if (level >= 0 && level < numLevels) {
          switch (level) {
          case 0:
            bitmap0 |= bit;
            break;
          case 1:
            bitmap1 |= bit;
            break;
          default:
            break;
          }
        }
      } else if (value == static_cast<unsigned char>('}') ||
                 value == static_cast<unsigned char>(']')) {
        if (level >= 0 && level < numLevels) {
          switch (level) {
          case 0:
            bitmap0 |= bit;
            break;
          case 1:
            bitmap1 |= bit;
            break;
          default:
            break;
          }
        }
        level--;
      } else if (value == static_cast<unsigned char>(':') ||
                 value == static_cast<unsigned char>(',')) {
        if (level >= 0 && level < numLevels) {
          switch (level) {
          case 0:
            bitmap0 |= bit;
            break;
          case 1:
            bitmap1 |= bit;
            break;
          default:
            break;
          }
        }
      }
    }
  }

  if (word_idx < static_cast<size_t>(levelSize)) {
    if (numLevels > 0) {
      leveledBitmapsIndex[static_cast<size_t>(levelSize) * 0 + word_idx] =
          bitmap0;
    }
    if (numLevels > 1) {
      leveledBitmapsIndex[static_cast<size_t>(levelSize) * 1 + word_idx] =
          bitmap1;
    }
  }
}

__device__ void leveled_bitmaps_index_sharemem_1(
    const char *file, size_t fileSize, const long *stringIndex,
    const char *leveledBitmapsAuxIndex, long *leveledBitmapsIndex,
    int levelSize, int numLevels) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = THREADS_PER_BLOCK * BYTES_PER_THREAD;

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int PACK_BYTES = 8;
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;
  constexpr int PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / PACK_BYTES;

  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % PACK_BYTES == 0);
  static_assert(sizeof(uint2) == PACK_BYTES);

  const int tid = threadIdx.x;
  const int tile_idx = blockIdx.x;
  const size_t block_start = static_cast<size_t>(tile_idx) * CHUNK_SIZE;
  const size_t global_thread_id =
      static_cast<size_t>(tile_idx) * THREADS_PER_BLOCK + tid;
  const size_t thread_global_base = global_thread_id * BYTES_PER_THREAD;
  const size_t word_idx = thread_global_base / BYTES_PER_THREAD;

  __shared__ uint2
      smem_packed_bytes[PACKED_GROUPS_PER_THREAD * THREADS_PER_BLOCK];

  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  for (int p = tid; p < PACKED_ELEMS_PER_BLOCK; p += blockDim.x) {
    const size_t global_byte_idx =
        block_start + static_cast<size_t>(p) * PACK_BYTES;

    uint2 packed_bytes = make_uint2(0u, 0u);

    if (global_byte_idx + PACK_BYTES <= fileSize) {
      const size_t global_packed_idx = global_byte_idx / PACK_BYTES;
      packed_bytes = file_packed_gmem[global_packed_idx];
    } else if (global_byte_idx < fileSize) {
      unsigned char tmp[PACK_BYTES] = {0};

#pragma unroll
      for (int i = 0; i < PACK_BYTES; ++i) {
        const size_t global_idx = global_byte_idx + i;
        if (global_idx < fileSize) {
          tmp[i] = static_cast<unsigned char>(file[global_idx]);
        }
      }

      unsigned char *packed_chars =
          reinterpret_cast<unsigned char *>(&packed_bytes);

#pragma unroll
      for (int i = 0; i < PACK_BYTES; ++i) {
        packed_chars[i] = tmp[i];
      }
    }

    const int owner_tid = p / PACKED_GROUPS_PER_THREAD;
    const int packed_group = p % PACKED_GROUPS_PER_THREAD;
    const int smem_idx = packed_group * THREADS_PER_BLOCK + owner_tid;

    smem_packed_bytes[smem_idx] = packed_bytes;
  }

  __syncthreads();

  if (word_idx < static_cast<size_t>(levelSize) && numLevels > 0) {
    leveledBitmapsIndex[word_idx] = 0;
  }

  if (thread_global_base >= fileSize) {
    return;
  }

  long bitmap0 = 0;

  const long string = stringIndex[word_idx];
  signed char level = leveledBitmapsAuxIndex[global_thread_id];

#pragma unroll
  for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
    const int smem_idx = group * THREADS_PER_BLOCK + tid;
    const uint2 packed_bytes_word = smem_packed_bytes[smem_idx];
    const unsigned char *packed_bytes =
        reinterpret_cast<const unsigned char *>(&packed_bytes_word);

#pragma unroll
    for (int i = 0; i < PACK_BYTES; ++i) {
      const int byte_offset = group * PACK_BYTES + i;
      const size_t global_idx = thread_global_base + byte_offset;

      if (global_idx >= fileSize) {
        break;
      }

      assert(level >= -1);

      if ((string & (1L << byte_offset)) != 0) {
        continue;
      }

      const unsigned char value = packed_bytes[i];
      const long bit = 1L << byte_offset;

      if (value == static_cast<unsigned char>('{') ||
          value == static_cast<unsigned char>('[')) {
        level++;
        if (level == 0 && numLevels > 0) {
          bitmap0 |= bit;
        }
      } else if (value == static_cast<unsigned char>('}') ||
                 value == static_cast<unsigned char>(']')) {
        if (level == 0 && numLevels > 0) {
          bitmap0 |= bit;
        }
        level--;
      } else if (value == static_cast<unsigned char>(':') ||
                 value == static_cast<unsigned char>(',')) {
        if (level == 0 && numLevels > 0) {
          bitmap0 |= bit;
        }
      }
    }
  }

  if (word_idx < static_cast<size_t>(levelSize) && numLevels > 0) {
    leveledBitmapsIndex[word_idx] = bitmap0;
  }
}

__global__ void leveled_bitmaps_index(const char *file, size_t fileSize,
                                      const long *stringIndex,
                                      char *leveledBitmapsAuxIndex,
                                      long *leveledBitmapsIndex, int levelSize,
                                      int numLevels) {
  assert(numLevels <= kMaxNumLevels);

  leveled_bitmaps_index_orig(file, fileSize, stringIndex,
                             leveledBitmapsAuxIndex, leveledBitmapsIndex,
                             levelSize, numLevels);

  // leveled_bitmaps_index_sharemem_transposed_packed(
  //     file, fileSize, stringIndex, leveledBitmapsAuxIndex,
  //     leveledBitmapsIndex, levelSize, numLevels);

  // leveled_bitmaps_index_sharemem_2(file, fileSize, stringIndex,
  //                                  leveledBitmapsAuxIndex,
  //                                  leveledBitmapsIndex, levelSize,
  //                                  numLevels);

  // leveled_bitmaps_index_sharemem_1(file, fileSize, stringIndex,
  //                                  leveledBitmapsAuxIndex,
  //                                  leveledBitmapsIndex, levelSize,
  //                                  numLevels);
}

} // namespace gpjson::index::kernels::sharemem
