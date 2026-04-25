#include "gpjson/index/kernels/sharemem.cuh"
#include "gpjson/log/log.hpp"

#include <cassert>
#include <cstdio>

namespace gpjson::index::kernels::sharemem {

__device__ void leveled_bitmaps_carry_index_sharemem_transposed_packed(
    const char *file, int fileSize, const long *stringIndex,
    char *leveledBitmapsAuxIndex) {
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

  const int block_start = tile_idx * CHUNK_SIZE;
  const int global_thread_id = tile_idx * THREADS_PER_BLOCK + tid;
  const int thread_global_base = global_thread_id * BYTES_PER_THREAD;

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

  signed char level = 0;

  if (thread_global_base < fileSize) {
    const long string = stringIndex[thread_global_base / BYTES_PER_THREAD];

#pragma unroll
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

        if ((string & (1L << byte_offset)) != 0) {
          continue;
        }

        const unsigned char value = packed_bytes[i];

        if (value == static_cast<unsigned char>('{') ||
            value == static_cast<unsigned char>('[')) {
          level++;
        } else if (value == static_cast<unsigned char>('}') ||
                   value == static_cast<unsigned char>(']')) {
          level--;
        }
      }
    }
  }

  leveledBitmapsAuxIndex[global_thread_id] = level;
}

__global__ void leveled_bitmaps_carry_index(const char *file, int fileSize,
                                            const long *stringIndex,
                                            char *leveledBitmapsAuxIndex) {
  leveled_bitmaps_carry_index_sharemem_transposed_packed(
      file, fileSize, stringIndex, leveledBitmapsAuxIndex);
}
} // namespace gpjson::index::kernels::sharemem
