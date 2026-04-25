#include "gpjson/log/log.hpp"

#include <cassert>
#include <cstdint>

namespace gpjson::index::kernels::sharemem {
namespace {

constexpr int kMaxNumLevels = 22;
constexpr int MAX_NUM_LEVEL_SMEM = 8;

static_assert(MAX_NUM_LEVEL_SMEM <= kMaxNumLevels);

template <int NumLevels> struct LevelBitmapAccum;

#define GPJSON_CAT_IMPL(a, b) a##b
#define GPJSON_CAT(a, b) GPJSON_CAT_IMPL(a, b)

#define GPJSON_LEVELS_1(M) M(0)
#define GPJSON_LEVELS_2(M) GPJSON_LEVELS_1(M) M(1)
#define GPJSON_LEVELS_3(M) GPJSON_LEVELS_2(M) M(2)
#define GPJSON_LEVELS_4(M) GPJSON_LEVELS_3(M) M(3)
#define GPJSON_LEVELS_5(M) GPJSON_LEVELS_4(M) M(4)
#define GPJSON_LEVELS_6(M) GPJSON_LEVELS_5(M) M(5)
#define GPJSON_LEVELS_7(M) GPJSON_LEVELS_6(M) M(6)
#define GPJSON_LEVELS_8(M) GPJSON_LEVELS_7(M) M(7)

#define GPJSON_LEVELS_FOR(N, M) GPJSON_CAT(GPJSON_LEVELS_, N)(M)

#define GPJSON_ACCUM_FIELD(level_id) long bitmap##level_id = 0;

#define GPJSON_ACCUM_RECORD_CASE(level_id)                                     \
  case level_id:                                                               \
    bitmap##level_id |= bit;                                                   \
    break;

#define GPJSON_ACCUM_WRITE(level_id)                                           \
  out[levelSize * (level_id) + wordIdx] = bitmap##level_id;

#define GPJSON_DEFINE_ACCUM(NUM_LEVELS)                                        \
  template <> struct LevelBitmapAccum<NUM_LEVELS> {                            \
    GPJSON_LEVELS_FOR(NUM_LEVELS, GPJSON_ACCUM_FIELD)                          \
                                                                               \
    __device__ __forceinline__ void record(int level, long bit) {              \
      switch (level) {                                                         \
        GPJSON_LEVELS_FOR(NUM_LEVELS, GPJSON_ACCUM_RECORD_CASE)                \
      default:                                                                 \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
                                                                               \
    __device__ __forceinline__ void write(long *out, int levelSize,            \
                                          int wordIdx) const {                 \
      GPJSON_LEVELS_FOR(NUM_LEVELS, GPJSON_ACCUM_WRITE)                        \
    }                                                                          \
  };

GPJSON_DEFINE_ACCUM(1)
GPJSON_DEFINE_ACCUM(2)
GPJSON_DEFINE_ACCUM(3)
GPJSON_DEFINE_ACCUM(4)
GPJSON_DEFINE_ACCUM(5)
GPJSON_DEFINE_ACCUM(6)
GPJSON_DEFINE_ACCUM(7)
GPJSON_DEFINE_ACCUM(8)

template <int NumLevels>
__device__ void leveled_bitmaps_index_smem_impl(const char *file, int fileSize,
                                                const long *stringIndex,
                                                char *leveledBitmapsAuxIndex,
                                                long *leveledBitmapsIndex,
                                                int levelSize) {
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

  static_assert(NumLevels >= 1);
  static_assert(NumLevels <= MAX_NUM_LEVEL_SMEM);
  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % PACK_BYTES == 0);
  static_assert(sizeof(uint2) == PACK_BYTES);

  const int tid = threadIdx.x;
  const int tile_idx = blockIdx.x;
  const int block_start = tile_idx * CHUNK_SIZE;
  const int global_thread_id = tile_idx * THREADS_PER_BLOCK + tid;
  const int thread_global_base = global_thread_id * BYTES_PER_THREAD;
  const int word_idx = thread_global_base / BYTES_PER_THREAD;

  __shared__ uint2
      smem_packed_bytes[PACKED_GROUPS_PER_THREAD * THREADS_PER_BLOCK];

  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  for (int p = tid; p < PACKED_ELEMS_PER_BLOCK; p += blockDim.x) {
    const int global_byte_idx = block_start + p * PACK_BYTES;

    uint2 packed_bytes = make_uint2(0u, 0u);

    if (global_byte_idx + PACK_BYTES - 1 < fileSize) {
      const int global_packed_idx = global_byte_idx / PACK_BYTES;
      packed_bytes = file_packed_gmem[global_packed_idx];
    } else if (global_byte_idx < fileSize) {
      unsigned char tmp[PACK_BYTES] = {0};

#pragma unroll
      for (int i = 0; i < PACK_BYTES; ++i) {
        const int global_idx = global_byte_idx + i;
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

  if (word_idx < levelSize) {
#pragma unroll
    for (int level = 0; level < NumLevels; level += 1) {
      leveledBitmapsIndex[levelSize * level + word_idx] = 0;
    }
  }

  if (thread_global_base >= fileSize) {
    return;
  }

  LevelBitmapAccum<NumLevels> accum;

  const long string = stringIndex[word_idx];
  signed char level = leveledBitmapsAuxIndex[global_thread_id];

  for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
    const int smem_idx = group * THREADS_PER_BLOCK + tid;
    const uint2 packed_bytes_word = smem_packed_bytes[smem_idx];
    const unsigned char *packed_bytes =
        reinterpret_cast<const unsigned char *>(&packed_bytes_word);

#pragma unroll
    for (int i = 0; i < PACK_BYTES; ++i) {
      const int byte_offset = group * PACK_BYTES + i;
      const int global_idx = thread_global_base + byte_offset;

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
        if (level >= 0 && level < NumLevels) {
          accum.record(level, bit);
        }
      } else if (value == static_cast<unsigned char>('}') ||
                 value == static_cast<unsigned char>(']')) {
        if (level >= 0 && level < NumLevels) {
          accum.record(level, bit);
        }
        level--;
      } else if (value == static_cast<unsigned char>(':') ||
                 value == static_cast<unsigned char>(',')) {
        if (level >= 0 && level < NumLevels) {
          accum.record(level, bit);
        }
      }
    }
  }

  if (word_idx < levelSize) {
    accum.write(leveledBitmapsIndex, levelSize, word_idx);
  }
}

__device__ void leveled_bitmaps_index_smem_dispatch(
    const char *file, int fileSize, const long *stringIndex,
    char *leveledBitmapsAuxIndex, long *leveledBitmapsIndex, int levelSize,
    int numLevels) {
  switch (numLevels) {
  case 1:
    leveled_bitmaps_index_smem_impl<1>(file, fileSize, stringIndex,
                                       leveledBitmapsAuxIndex,
                                       leveledBitmapsIndex, levelSize);
    break;
  case 2:
    leveled_bitmaps_index_smem_impl<2>(file, fileSize, stringIndex,
                                       leveledBitmapsAuxIndex,
                                       leveledBitmapsIndex, levelSize);
    break;
  case 3:
    leveled_bitmaps_index_smem_impl<3>(file, fileSize, stringIndex,
                                       leveledBitmapsAuxIndex,
                                       leveledBitmapsIndex, levelSize);
    break;
  case 4:
    leveled_bitmaps_index_smem_impl<4>(file, fileSize, stringIndex,
                                       leveledBitmapsAuxIndex,
                                       leveledBitmapsIndex, levelSize);
    break;
  case 5:
    leveled_bitmaps_index_smem_impl<5>(file, fileSize, stringIndex,
                                       leveledBitmapsAuxIndex,
                                       leveledBitmapsIndex, levelSize);
    break;
  case 6:
    leveled_bitmaps_index_smem_impl<6>(file, fileSize, stringIndex,
                                       leveledBitmapsAuxIndex,
                                       leveledBitmapsIndex, levelSize);
    break;
  case 7:
    leveled_bitmaps_index_smem_impl<7>(file, fileSize, stringIndex,
                                       leveledBitmapsAuxIndex,
                                       leveledBitmapsIndex, levelSize);
    break;
  case 8:
    leveled_bitmaps_index_smem_impl<8>(file, fileSize, stringIndex,
                                       leveledBitmapsAuxIndex,
                                       leveledBitmapsIndex, levelSize);
    break;
  default:
    break;
  }
}

} // namespace

__device__ void leveled_bitmaps_index_default(const char *file, int fileSize,
                                              const long *stringIndex,
                                              char *leveledBitmapsAuxIndex,
                                              long *leveledBitmapsIndex,
                                              int levelSize, int numLevels) {
  assert(numLevels <= kMaxNumLevels);

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int charsPerThread = (fileSize + stride - 1) / stride;
  int bitmapAlignedCharsPerThread = ((charsPerThread + 64 - 1) / 64) * 64;
  int start = index * bitmapAlignedCharsPerThread;
  int end = start + bitmapAlignedCharsPerThread;

  signed char level = leveledBitmapsAuxIndex[index];

  for (int blockStart = start; blockStart < end && blockStart < fileSize;
       blockStart += 64) {
    const int wordIndex = blockStart / 64;
    const long string = stringIndex[wordIndex];
    // Accumulate a full 64-bit output word locally, then write each level once.
    long structuralBitmaps[kMaxNumLevels];
    for (int bitmapLevel = 0; bitmapLevel < numLevels; bitmapLevel += 1) {
      structuralBitmaps[bitmapLevel] = 0;
    }

    const int blockEnd =
        blockStart + 64 < fileSize ? blockStart + 64 : fileSize;
    for (int i = blockStart; i < blockEnd; i += 1) {
      assert(level >= -1);

      const long offsetInBlock = i % 64;
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
      leveledBitmapsIndex[levelSize * bitmapLevel + wordIndex] =
          structuralBitmaps[bitmapLevel];
    }
  }
}

__global__ void leveled_bitmaps_index(const char *file, int fileSize,
                                      const long *stringIndex,
                                      char *leveledBitmapsAuxIndex,
                                      long *leveledBitmapsIndex, int levelSize,
                                      int numLevels) {
  assert(numLevels <= kMaxNumLevels);

  if (numLevels >= 1 && numLevels <= MAX_NUM_LEVEL_SMEM) {
    leveled_bitmaps_index_smem_dispatch(
        file, fileSize, stringIndex, leveledBitmapsAuxIndex,
        leveledBitmapsIndex, levelSize, numLevels);
  } else {
    leveled_bitmaps_index_default(file, fileSize, stringIndex,
                                  leveledBitmapsAuxIndex, leveledBitmapsIndex,
                                  levelSize, numLevels);
  }
}

#undef GPJSON_DEFINE_ACCUM
#undef GPJSON_ACCUM_WRITE
#undef GPJSON_ACCUM_RECORD_CASE
#undef GPJSON_ACCUM_FIELD
#undef GPJSON_LEVELS_FOR
#undef GPJSON_LEVELS_8
#undef GPJSON_LEVELS_7
#undef GPJSON_LEVELS_6
#undef GPJSON_LEVELS_5
#undef GPJSON_LEVELS_4
#undef GPJSON_LEVELS_3
#undef GPJSON_LEVELS_2
#undef GPJSON_LEVELS_1
#undef GPJSON_CAT
#undef GPJSON_CAT_IMPL

} // namespace gpjson::index::kernels::sharemem
