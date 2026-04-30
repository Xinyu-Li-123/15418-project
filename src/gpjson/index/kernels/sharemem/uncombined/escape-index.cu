#include "gpjson/index/kernels/sharemem/packed-bytes.cuh"
#include "gpjson/log/log.hpp"

#include <cstddef>
#include <cstdint>

namespace gpjson::index::kernels::sharemem {

#ifndef ESCAPE_GMEM_PACK_TYPE
#define ESCAPE_GMEM_PACK_TYPE uint4
#endif

#ifndef ESCAPE_SMEM_PACK_TYPE
#define ESCAPE_SMEM_PACK_TYPE uint2
#endif

#ifdef FORCED_GMEM_PACK_TYPE
using EscapeGmemPackT = FORCED_GMEM_PACK_TYPE;
#else
using EscapeGmemPackT = ESCAPE_GMEM_PACK_TYPE;
#endif

#ifdef FORCED_SMEM_PACK_TYPE
using EscapeSmemPackT = FORCED_SMEM_PACK_TYPE;
#else
using EscapeSmemPackT = ESCAPE_SMEM_PACK_TYPE;
#endif

__device__ void escape_index_packed(const char *file, size_t fileSize,
                                    char *escapeCarryIndex, long *escapeIndex) {
  using PackT = EscapeGmemPackT;

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;

  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int PACK_BYTES = static_cast<int>(sizeof(PackT));
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;

  static_assert(PACK_BYTES == 2 || PACK_BYTES == 4 || PACK_BYTES == 8 ||
                    PACK_BYTES == 16,
                "ESCAPE_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);

  const int tid = threadIdx.x;

  const size_t global_thread_id =
      static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

  const size_t thread_global_base = global_thread_id * BYTES_PER_THREAD;

  if (thread_global_base >= fileSize) {
    return;
  }

  bool carry = global_thread_id == 0
                   ? false
                   : static_cast<bool>(escapeCarryIndex[global_thread_id - 1]);

  uint64_t escape_word = 0;

#pragma unroll
  for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
    const size_t global_byte_idx =
        thread_global_base + static_cast<size_t>(group) * PACK_BYTES;

    const PackT packed_bytes_word = packed_bytes::load_gmem_pack_or_tail<PackT>(
        file, fileSize, global_byte_idx);
    const unsigned char *packed_bytes = packed_bytes::bytes(packed_bytes_word);

#pragma unroll
    for (int i = 0; i < PACK_BYTES; ++i) {
      const int byte_offset = group * PACK_BYTES + i;
      const size_t global_idx = thread_global_base + byte_offset;

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

  escapeIndex[thread_global_base / BYTES_PER_THREAD] =
      static_cast<long>(escape_word);
}

__device__ void escape_index_sharemem_packed(const char *file, size_t fileSize,
                                             char *escapeCarryIndex,
                                             long *escapeIndex) {
  using GmemPackT = EscapeGmemPackT;
  using SmemPackT = EscapeSmemPackT;

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = THREADS_PER_BLOCK * BYTES_PER_THREAD;

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int GMEM_PACK_BYTES = static_cast<int>(sizeof(GmemPackT));
  constexpr int SMEM_PACK_BYTES = static_cast<int>(sizeof(SmemPackT));
  constexpr int SMEM_GROUPS_PER_THREAD = BYTES_PER_THREAD / SMEM_PACK_BYTES;
  constexpr int SMEM_PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / SMEM_PACK_BYTES;

  static_assert(GMEM_PACK_BYTES == 2 || GMEM_PACK_BYTES == 4 ||
                    GMEM_PACK_BYTES == 8 || GMEM_PACK_BYTES == 16,
                "ESCAPE_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(SMEM_PACK_BYTES == 2 || SMEM_PACK_BYTES == 4 ||
                    SMEM_PACK_BYTES == 8 || SMEM_PACK_BYTES == 16,
                "ESCAPE_SMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(BYTES_PER_THREAD % GMEM_PACK_BYTES == 0);
  static_assert(BYTES_PER_THREAD % SMEM_PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % GMEM_PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % SMEM_PACK_BYTES == 0);

  const int tid = threadIdx.x;
  const int tile_idx = blockIdx.x;

  const size_t block_start = static_cast<size_t>(tile_idx) * CHUNK_SIZE;

  const size_t global_thread_id =
      static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

  const size_t thread_global_base = global_thread_id * BYTES_PER_THREAD;

  // Non-transposed packed layout:
  //
  //   smem_packed_bytes[p]
  //
  // where p follows file order. Equivalently:
  //
  //   smem_packed_bytes[owner_tid * PACKED_GROUPS_PER_THREAD + group]
  //
  // This makes the shared-memory write contiguous, but the later per-group
  // warp read is strided by PACKED_GROUPS_PER_THREAD.
  __shared__ SmemPackT smem_packed_bytes[SMEM_PACKED_ELEMS_PER_BLOCK];

  packed_bytes::stage_file_to_smem</*TRANSPOSED=*/false, GmemPackT, SmemPackT,
                                   BYTES_PER_THREAD, THREADS_PER_BLOCK>(
      file, fileSize, block_start, smem_packed_bytes);

  __syncthreads();

  bool carry = global_thread_id == 0
                   ? false
                   : static_cast<bool>(escapeCarryIndex[global_thread_id - 1]);

  uint64_t escape_word = 0;

#pragma unroll
  for (int group = 0; group < SMEM_GROUPS_PER_THREAD; ++group) {
    const int smem_idx = tid * SMEM_GROUPS_PER_THREAD + group;
    const SmemPackT packed_bytes_word = smem_packed_bytes[smem_idx];

    const unsigned char *packed_bytes = packed_bytes::bytes(packed_bytes_word);

#pragma unroll
    for (int i = 0; i < SMEM_PACK_BYTES; ++i) {
      const int byte_offset = group * SMEM_PACK_BYTES + i;
      const size_t global_idx = thread_global_base + byte_offset;

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

/**
 * Transposed packed shared-memory escape-index kernel.
 *
 * This kernel uses the same combo of optimization as the packed newline-index
 * kernel: coalesced, packed read from gmem, transposed write to smem to enable
 * coalesced, smem read from smem.
 */
__device__ void escape_index_sharemem_transposed_packed(const char *file,
                                                        size_t fileSize,
                                                        char *escapeCarryIndex,
                                                        long *escapeIndex) {
  using GmemPackT = EscapeGmemPackT;
  using SmemPackT = EscapeSmemPackT;

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = THREADS_PER_BLOCK * BYTES_PER_THREAD;

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int GMEM_PACK_BYTES = static_cast<int>(sizeof(GmemPackT));
  constexpr int SMEM_PACK_BYTES = static_cast<int>(sizeof(SmemPackT));
  constexpr int SMEM_GROUPS_PER_THREAD = BYTES_PER_THREAD / SMEM_PACK_BYTES;
  constexpr int SMEM_PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / SMEM_PACK_BYTES;

  static_assert(GMEM_PACK_BYTES == 2 || GMEM_PACK_BYTES == 4 ||
                    GMEM_PACK_BYTES == 8 || GMEM_PACK_BYTES == 16,
                "ESCAPE_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(SMEM_PACK_BYTES == 2 || SMEM_PACK_BYTES == 4 ||
                    SMEM_PACK_BYTES == 8 || SMEM_PACK_BYTES == 16,
                "ESCAPE_SMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(BYTES_PER_THREAD % GMEM_PACK_BYTES == 0);
  static_assert(BYTES_PER_THREAD % SMEM_PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % GMEM_PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % SMEM_PACK_BYTES == 0);

  const int tid = threadIdx.x;
  const int tile_idx = blockIdx.x;

  const size_t block_start = static_cast<size_t>(tile_idx) * CHUNK_SIZE;

  const size_t global_thread_id =
      static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

  const size_t thread_global_base = global_thread_id * BYTES_PER_THREAD;

  __shared__ SmemPackT smem_packed_bytes[SMEM_PACKED_ELEMS_PER_BLOCK];

  packed_bytes::stage_file_to_smem</*TRANSPOSED=*/true, GmemPackT, SmemPackT,
                                   BYTES_PER_THREAD, THREADS_PER_BLOCK>(
      file, fileSize, block_start, smem_packed_bytes);

  __syncthreads();

  bool carry = global_thread_id == 0
                   ? false
                   : static_cast<bool>(escapeCarryIndex[global_thread_id - 1]);

  uint64_t escape_word = 0;

#pragma unroll
  for (int group = 0; group < SMEM_GROUPS_PER_THREAD; ++group) {
    const int smem_idx = group * THREADS_PER_BLOCK + tid;
    const SmemPackT packed_bytes_word = smem_packed_bytes[smem_idx];

    const unsigned char *packed_bytes = packed_bytes::bytes(packed_bytes_word);

#pragma unroll
    for (int i = 0; i < SMEM_PACK_BYTES; ++i) {
      const int byte_offset = group * SMEM_PACK_BYTES + i;
      const size_t global_idx = thread_global_base + byte_offset;

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

__global__ void escape_index(const char *file, size_t fileSize,
                             char *escapeCarryIndex, long *escapeIndex) {

  escape_index_packed(file, fileSize, escapeCarryIndex, escapeIndex);

  // escape_index_sharemem_packed(file, fileSize, escapeCarryIndex,
  // escapeIndex);

  // escape_index_sharemem_transposed_packed(file, fileSize, escapeCarryIndex,
  //                                         escapeIndex);
}

} // namespace gpjson::index::kernels::sharemem
