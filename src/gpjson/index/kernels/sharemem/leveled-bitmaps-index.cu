#include "gpjson/index/kernels/sharemem/packed-bytes.cuh"
#include "gpjson/log/log.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace gpjson::index::kernels::sharemem {
namespace {

constexpr int kMaxNumLevels = 22;

#ifndef LBM_INDEX_GMEM_PACK_TYPE
#define LBM_INDEX_GMEM_PACK_TYPE uint4
#endif

#ifndef LBM_INDEX_SMEM_PACK_TYPE
#define LBM_INDEX_SMEM_PACK_TYPE uint2
#endif

using LbmIndexGmemPackT = LBM_INDEX_GMEM_PACK_TYPE;
using LbmIndexSmemPackT = LBM_INDEX_SMEM_PACK_TYPE;

template <bool TRANSPOSED, typename SmemPackT>
__device__ __forceinline__ void compute_leveled_bitmaps_from_smem(
    size_t fileSize, const long *stringIndex,
    const char *leveledBitmapsAuxIndex, long *leveledBitmapsIndex,
    int levelSize, int numLevels, const SmemPackT *smem_packed_bytes) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int SMEM_PACK_BYTES = static_cast<int>(sizeof(SmemPackT));
  constexpr int SMEM_GROUPS_PER_THREAD = BYTES_PER_THREAD / SMEM_PACK_BYTES;

  static_assert(SMEM_PACK_BYTES == 2 || SMEM_PACK_BYTES == 4 ||
                    SMEM_PACK_BYTES == 8 || SMEM_PACK_BYTES == 16,
                "SmemPackT must be 2, 4, 8, or 16 bytes.");

  const int tid = threadIdx.x;
  const int tile_idx = blockIdx.x;

  const size_t global_thread_id =
      static_cast<size_t>(tile_idx) * THREADS_PER_BLOCK + tid;
  const size_t thread_global_base = global_thread_id * BYTES_PER_THREAD;
  const size_t word_idx = thread_global_base / BYTES_PER_THREAD;

  if (thread_global_base >= fileSize) {
    return;
  }

  long bitmap_words[kMaxNumLevels];

#pragma unroll
  for (int bitmap_level = 0; bitmap_level < kMaxNumLevels; ++bitmap_level) {
    bitmap_words[bitmap_level] = 0;
  }

  const uint64_t string = static_cast<uint64_t>(stringIndex[word_idx]);
  signed char cur_level = leveledBitmapsAuxIndex[global_thread_id];

#pragma unroll
  for (int group = 0; group < SMEM_GROUPS_PER_THREAD; ++group) {
    int smem_idx;

    if constexpr (TRANSPOSED) {
      smem_idx = group * THREADS_PER_BLOCK + tid;
    } else {
      smem_idx = tid * SMEM_GROUPS_PER_THREAD + group;
    }

    const SmemPackT packed_bytes_word = smem_packed_bytes[smem_idx];

    const unsigned char *packed_bytes =
        reinterpret_cast<const unsigned char *>(&packed_bytes_word);

    const int byte_offset_base = group * SMEM_PACK_BYTES;
    constexpr unsigned int PACK_STRING_MASK =
        packed_bytes::pack_bit_mask<SMEM_PACK_BYTES>();

    const unsigned int string_pack_mask = static_cast<unsigned int>(
        (string >> byte_offset_base) & PACK_STRING_MASK);

#pragma unroll
    for (int i = 0; i < SMEM_PACK_BYTES; ++i) {
      const int byte_offset = byte_offset_base + i;
      const size_t global_idx = thread_global_base + byte_offset;

      if (global_idx >= fileSize) {
        break;
      }

      assert(cur_level >= -1);

      const long bit = 1L << byte_offset;

      if ((string_pack_mask & (1u << i)) != 0u) {
        continue;
      }

      const unsigned char value = packed_bytes[i];

      if (value == static_cast<unsigned char>('{') ||
          value == static_cast<unsigned char>('[')) {
        cur_level++;

        if (cur_level >= 0 && cur_level < numLevels) {
          bitmap_words[cur_level] |= bit;
        }
      } else if (value == static_cast<unsigned char>('}') ||
                 value == static_cast<unsigned char>(']')) {
        if (cur_level >= 0 && cur_level < numLevels) {
          bitmap_words[cur_level] |= bit;
        }

        cur_level--;
      } else if (value == static_cast<unsigned char>(':') ||
                 value == static_cast<unsigned char>(',')) {
        if (cur_level >= 0 && cur_level < numLevels) {
          bitmap_words[cur_level] |= bit;
        }
      }
    }
  }

  if (word_idx < static_cast<size_t>(levelSize)) {
    for (int bitmap_level = 0; bitmap_level < numLevels; ++bitmap_level) {
      leveledBitmapsIndex[static_cast<size_t>(levelSize) * bitmap_level +
                          word_idx] = bitmap_words[bitmap_level];
    }
  }
}

} // namespace

__device__ void leveled_bitmaps_index_orig(const char *file, size_t fileSize,
                                           const long *stringIndex,
                                           char *leveledBitmapsAuxIndex,
                                           long *leveledBitmapsIndex,
                                           int levelSize, int numLevels) {
  assert(numLevels <= kMaxNumLevels);

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  size_t charsPerThread = (fileSize + static_cast<size_t>(stride) - 1) / stride;
  size_t bitmapAlignedCharsPerThread = ((charsPerThread + 64 - 1) / 64) * 64;
  size_t start = static_cast<size_t>(index) * bitmapAlignedCharsPerThread;
  size_t end = start + bitmapAlignedCharsPerThread;

  signed char level = leveledBitmapsAuxIndex[index];

  for (size_t blockStart = start; blockStart < end && blockStart < fileSize;
       blockStart += 64) {
    const size_t wordIndex = blockStart / 64;
    const long string = stringIndex[wordIndex];

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
                          wordIndex] = structuralBitmaps[bitmapLevel];
    }
  }
}

__device__ void leveled_bitmaps_index_packed(const char *file, size_t fileSize,
                                             const long *stringIndex,
                                             const char *leveledBitmapsAuxIndex,
                                             long *leveledBitmapsIndex,
                                             int levelSize, int numLevels) {
  using PackT = LbmIndexGmemPackT;

  assert(numLevels <= kMaxNumLevels);

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;

  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int PACK_BYTES = static_cast<int>(sizeof(PackT));
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;

  static_assert(PACK_BYTES == 2 || PACK_BYTES == 4 || PACK_BYTES == 8 ||
                    PACK_BYTES == 16,
                "LBM_INDEX_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);

  const int tid = threadIdx.x;

  const size_t global_thread_id =
      static_cast<size_t>(blockIdx.x) * THREADS_PER_BLOCK + tid;

  const size_t thread_global_base = global_thread_id * BYTES_PER_THREAD;

  signed char level = leveledBitmapsAuxIndex[global_thread_id];

  long structuralBitmaps[kMaxNumLevels];

#pragma unroll
  for (int bitmapLevel = 0; bitmapLevel < kMaxNumLevels; ++bitmapLevel) {
    if (bitmapLevel < numLevels) {
      structuralBitmaps[bitmapLevel] = 0;
    }
  }

  if (thread_global_base < fileSize) {
    const size_t wordIndex = thread_global_base / BYTES_PER_THREAD;
    const uint64_t string = static_cast<uint64_t>(stringIndex[wordIndex]);

#pragma unroll
    for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
      const int byte_offset_base = group * PACK_BYTES;
      const size_t group_global_base =
          thread_global_base + static_cast<size_t>(byte_offset_base);

      if (group_global_base >= fileSize) {
        break;
      }

      const PackT packed_bytes_word =
          packed_bytes::load_gmem_pack_or_tail<PackT>(file, fileSize,
                                                      group_global_base);
      const unsigned char *packed_bytes =
          packed_bytes::bytes(packed_bytes_word);

#pragma unroll
      for (int i = 0; i < PACK_BYTES; ++i) {
        const int byte_offset = byte_offset_base + i;
        const size_t global_idx = thread_global_base + byte_offset;

        if (global_idx >= fileSize) {
          break;
        }

        const long bit = 1L << byte_offset;

        if ((string & static_cast<uint64_t>(bit)) != 0) {
          continue;
        }

        const unsigned char value = packed_bytes[i];

        if (value == static_cast<unsigned char>('{') ||
            value == static_cast<unsigned char>('[')) {
          level++;

          if (level >= 0 && level < numLevels) {
            structuralBitmaps[level] |= bit;
          }
        } else if (value == static_cast<unsigned char>('}') ||
                   value == static_cast<unsigned char>(']')) {
          if (level >= 0 && level < numLevels) {
            structuralBitmaps[level] |= bit;
          }

          level--;
        } else if (value == static_cast<unsigned char>(':') ||
                   value == static_cast<unsigned char>(',')) {
          if (level >= 0 && level < numLevels) {
            structuralBitmaps[level] |= bit;
          }
        }
      }
    }

    for (int bitmapLevel = 0; bitmapLevel < numLevels; ++bitmapLevel) {
      leveledBitmapsIndex[static_cast<size_t>(levelSize) * bitmapLevel +
                          wordIndex] = structuralBitmaps[bitmapLevel];
    }
  }
}

__device__ void leveled_bitmaps_index_packed_skip_str(
    const char *file, size_t fileSize, const long *stringIndex,
    const char *leveledBitmapsAuxIndex, long *leveledBitmapsIndex,
    int levelSize, int numLevels) {
  using PackT = LbmIndexGmemPackT;

  assert(numLevels <= kMaxNumLevels);

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;

  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int PACK_BYTES = static_cast<int>(sizeof(PackT));
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;

  static_assert(PACK_BYTES == 2 || PACK_BYTES == 4 || PACK_BYTES == 8 ||
                    PACK_BYTES == 16,
                "LBM_INDEX_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);

  const int tid = threadIdx.x;

  const size_t global_thread_id =
      static_cast<size_t>(blockIdx.x) * THREADS_PER_BLOCK + tid;

  const size_t thread_global_base = global_thread_id * BYTES_PER_THREAD;

  signed char level = leveledBitmapsAuxIndex[global_thread_id];

  long structuralBitmaps[kMaxNumLevels];

#pragma unroll
  for (int bitmapLevel = 0; bitmapLevel < kMaxNumLevels; ++bitmapLevel) {
    if (bitmapLevel < numLevels) {
      structuralBitmaps[bitmapLevel] = 0;
    }
  }

  if (thread_global_base < fileSize) {
    const size_t wordIndex = thread_global_base / BYTES_PER_THREAD;
    const uint64_t string = static_cast<uint64_t>(stringIndex[wordIndex]);

#pragma unroll
    for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
      const int byte_offset_base = group * PACK_BYTES;
      const size_t group_global_base =
          thread_global_base + static_cast<size_t>(byte_offset_base);

      if (group_global_base >= fileSize) {
        break;
      }

      constexpr unsigned int PACK_STRING_MASK =
          packed_bytes::pack_bit_mask<PACK_BYTES>();

      const unsigned int string_pack_mask = static_cast<unsigned int>(
          (string >> byte_offset_base) & PACK_STRING_MASK);

      const unsigned int outside_string_mask =
          (~string_pack_mask) & PACK_STRING_MASK;

      if (outside_string_mask == 0u) {
        continue;
      }

      const PackT packed_bytes_word =
          packed_bytes::load_gmem_pack_or_tail<PackT>(file, fileSize,
                                                      group_global_base);
      const unsigned char *packed_bytes =
          packed_bytes::bytes(packed_bytes_word);

#pragma unroll
      for (int i = 0; i < PACK_BYTES; ++i) {
        if ((outside_string_mask & (1u << i)) == 0u) {
          continue;
        }

        const int byte_offset = byte_offset_base + i;
        const size_t global_idx = thread_global_base + byte_offset;

        if (global_idx >= fileSize) {
          break;
        }

        const long bit = 1L << byte_offset;
        const unsigned char value = packed_bytes[i];

        if (value == static_cast<unsigned char>('{') ||
            value == static_cast<unsigned char>('[')) {
          level++;

          if (level >= 0 && level < numLevels) {
            structuralBitmaps[level] |= bit;
          }
        } else if (value == static_cast<unsigned char>('}') ||
                   value == static_cast<unsigned char>(']')) {
          if (level >= 0 && level < numLevels) {
            structuralBitmaps[level] |= bit;
          }

          level--;
        } else if (value == static_cast<unsigned char>(':') ||
                   value == static_cast<unsigned char>(',')) {
          if (level >= 0 && level < numLevels) {
            structuralBitmaps[level] |= bit;
          }
        }
      }
    }

    for (int bitmapLevel = 0; bitmapLevel < numLevels; ++bitmapLevel) {
      leveledBitmapsIndex[static_cast<size_t>(levelSize) * bitmapLevel +
                          wordIndex] = structuralBitmaps[bitmapLevel];
    }
  }
}

__device__ void leveled_bitmaps_index_sharemem_packed(
    const char *file, size_t fileSize, const long *stringIndex,
    const char *leveledBitmapsAuxIndex, long *leveledBitmapsIndex,
    int levelSize, int numLevels) {
  using GmemPackT = LbmIndexGmemPackT;
  using SmemPackT = LbmIndexSmemPackT;

  assert(numLevels <= kMaxNumLevels);

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = THREADS_PER_BLOCK * BYTES_PER_THREAD;

  constexpr int GMEM_PACK_BYTES = static_cast<int>(sizeof(GmemPackT));
  constexpr int SMEM_PACK_BYTES = static_cast<int>(sizeof(SmemPackT));
  constexpr int SMEM_PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / SMEM_PACK_BYTES;

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  static_assert(GMEM_PACK_BYTES == 2 || GMEM_PACK_BYTES == 4 ||
                    GMEM_PACK_BYTES == 8 || GMEM_PACK_BYTES == 16,
                "LBM_INDEX_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(SMEM_PACK_BYTES == 2 || SMEM_PACK_BYTES == 4 ||
                    SMEM_PACK_BYTES == 8 || SMEM_PACK_BYTES == 16,
                "LBM_INDEX_SMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(BYTES_PER_THREAD % GMEM_PACK_BYTES == 0);
  static_assert(BYTES_PER_THREAD % SMEM_PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % GMEM_PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % SMEM_PACK_BYTES == 0);

  const int tile_idx = blockIdx.x;
  const size_t block_start = static_cast<size_t>(tile_idx) * CHUNK_SIZE;

  __shared__ SmemPackT smem_packed_bytes[SMEM_PACKED_ELEMS_PER_BLOCK];

  packed_bytes::stage_file_to_smem</*TRANSPOSED=*/false, GmemPackT, SmemPackT,
                                   BYTES_PER_THREAD, THREADS_PER_BLOCK>(
      file, fileSize, block_start, smem_packed_bytes);

  __syncthreads();

  compute_leveled_bitmaps_from_smem</*TRANSPOSED=*/false, SmemPackT>(
      fileSize, stringIndex, leveledBitmapsAuxIndex, leveledBitmapsIndex,
      levelSize, numLevels, smem_packed_bytes);
}

__device__ void leveled_bitmaps_index_sharemem_transposed_packed(
    const char *file, size_t fileSize, const long *stringIndex,
    const char *leveledBitmapsAuxIndex, long *leveledBitmapsIndex,
    int levelSize, int numLevels) {
  using GmemPackT = LbmIndexGmemPackT;
  using SmemPackT = LbmIndexSmemPackT;

  assert(numLevels <= kMaxNumLevels);

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = THREADS_PER_BLOCK * BYTES_PER_THREAD;

  constexpr int GMEM_PACK_BYTES = static_cast<int>(sizeof(GmemPackT));
  constexpr int SMEM_PACK_BYTES = static_cast<int>(sizeof(SmemPackT));
  constexpr int SMEM_PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / SMEM_PACK_BYTES;

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  static_assert(GMEM_PACK_BYTES == 2 || GMEM_PACK_BYTES == 4 ||
                    GMEM_PACK_BYTES == 8 || GMEM_PACK_BYTES == 16,
                "LBM_INDEX_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(SMEM_PACK_BYTES == 2 || SMEM_PACK_BYTES == 4 ||
                    SMEM_PACK_BYTES == 8 || SMEM_PACK_BYTES == 16,
                "LBM_INDEX_SMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(BYTES_PER_THREAD % GMEM_PACK_BYTES == 0);
  static_assert(BYTES_PER_THREAD % SMEM_PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % GMEM_PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % SMEM_PACK_BYTES == 0);

  const int tile_idx = blockIdx.x;
  const size_t block_start = static_cast<size_t>(tile_idx) * CHUNK_SIZE;

  __shared__ SmemPackT smem_packed_bytes[SMEM_PACKED_ELEMS_PER_BLOCK];

  packed_bytes::stage_file_to_smem</*TRANSPOSED=*/true, GmemPackT, SmemPackT,
                                   BYTES_PER_THREAD, THREADS_PER_BLOCK>(
      file, fileSize, block_start, smem_packed_bytes);

  __syncthreads();

  compute_leveled_bitmaps_from_smem</*TRANSPOSED=*/true, SmemPackT>(
      fileSize, stringIndex, leveledBitmapsAuxIndex, leveledBitmapsIndex,
      levelSize, numLevels, smem_packed_bytes);
}

__global__ void leveled_bitmaps_index(const char *file, size_t fileSize,
                                      const long *stringIndex,
                                      char *leveledBitmapsAuxIndex,
                                      long *leveledBitmapsIndex, int levelSize,
                                      int numLevels) {
  assert(numLevels <= kMaxNumLevels);

  // Original byte-by-byte version.
  //
  // leveled_bitmaps_index_orig(file, fileSize, stringIndex,
  //                            leveledBitmapsAuxIndex, leveledBitmapsIndex,
  //                            levelSize, numLevels);

  // Direct packed.
  //
  leveled_bitmaps_index_packed(file, fileSize, stringIndex,
                               leveledBitmapsAuxIndex, leveledBitmapsIndex,
                               levelSize, numLevels);

  // Direct packed with string-pack load skip.
  //
  // leveled_bitmaps_index_packed_skip_str(
  //     file, fileSize, stringIndex, leveledBitmapsAuxIndex,
  //     leveledBitmapsIndex, levelSize, numLevels);

  // Shared-memory staged, non-transposed.
  //
  // leveled_bitmaps_index_sharemem_packed(
  //     file, fileSize, stringIndex, leveledBitmapsAuxIndex,
  //     leveledBitmapsIndex, levelSize, numLevels);

  // Shared-memory staged, transposed.
  //
  // leveled_bitmaps_index_sharemem_transposed_packed(
  //     file, fileSize, stringIndex, leveledBitmapsAuxIndex,
  //     leveledBitmapsIndex, levelSize, numLevels);
}

} // namespace gpjson::index::kernels::sharemem
