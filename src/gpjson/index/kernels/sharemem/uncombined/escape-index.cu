#include "gpjson/log/log.hpp"

#include <cstdint>

namespace gpjson::index::kernels::sharemem {

/**
 * Transposed packed shared-memory escape-index kernel.
 *
 * This kernel uses the same combo of optimization as the packed newline-index
 * kernel: coalesced, packed read from gmem, transposed write to smem to enable
 * coalesced, smem read from smem.
 */
__device__ void escape_index_sharemem_transposed_packed(const char *file,
                                                        long fileSize,
                                                        char *escapeCarryIndex,
                                                        long *escapeIndex) {
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

  const long block_start =
      static_cast<long>(tile_idx) * static_cast<long>(CHUNK_SIZE);

  const long global_thread_id =
      static_cast<long>(blockIdx.x) * static_cast<long>(blockDim.x) +
      static_cast<long>(tid);

  const long thread_global_base =
      global_thread_id * static_cast<long>(BYTES_PER_THREAD);

  __shared__ uint2
      smem_packed_bytes[PACKED_GROUPS_PER_THREAD * THREADS_PER_BLOCK];

  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  for (int p = tid; p < PACKED_ELEMS_PER_BLOCK; p += blockDim.x) {
    const long global_byte_idx =
        block_start + static_cast<long>(p) * static_cast<long>(PACK_BYTES);

    uint2 packed_bytes = make_uint2(0u, 0u);

    if (global_byte_idx + PACK_BYTES - 1 < fileSize) {
      const long global_packed_idx = global_byte_idx / PACK_BYTES;
      packed_bytes = file_packed_gmem[global_packed_idx];
    } else if (global_byte_idx < fileSize) {
      unsigned char tmp[PACK_BYTES] = {0};

#pragma unroll
      for (int i = 0; i < PACK_BYTES; ++i) {
        const long global_idx = global_byte_idx + i;
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

  bool carry = global_thread_id == 0
                   ? false
                   : static_cast<bool>(escapeCarryIndex[global_thread_id - 1]);

  uint64_t escape_word = 0;

#pragma unroll
  for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
    const int smem_idx = group * THREADS_PER_BLOCK + tid;
    const uint2 packed_bytes_word = smem_packed_bytes[smem_idx];

    const unsigned char *packed_bytes =
        reinterpret_cast<const unsigned char *>(&packed_bytes_word);

#pragma unroll
    for (int i = 0; i < PACK_BYTES; ++i) {
      const int byte_offset = group * PACK_BYTES + i;
      const long global_idx = thread_global_base + byte_offset;

      if (global_idx >= fileSize) {
        break;
      }

      if (carry) {
        escape_word |= uint64_t{1} << byte_offset;
      }

      if (packed_bytes[i] == static_cast<unsigned char>('\\')) {
        carry = !carry;
      } else {
        carry = false;
      }
    }
  }

  if (thread_global_base < fileSize) {
    escapeIndex[thread_global_base / BYTES_PER_THREAD] =
        static_cast<long>(escape_word);
  }
}

__global__ void escape_index(const char *file, long fileSize,
                             char *escapeCarryIndex, long *escapeIndex) {
  escape_index_sharemem_transposed_packed(file, fileSize, escapeCarryIndex,
                                          escapeIndex);
}

} // namespace gpjson::index::kernels::sharemem
