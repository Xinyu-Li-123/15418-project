#include "gpjson/index/kernels/sharemem/packed-bytes.cuh"
#include "gpjson/log/log.hpp"

#include <cstddef>
#include <cstdint>

namespace gpjson::index::kernels::sharemem {

#ifndef QUOTE_GMEM_PACK_TYPE
#define QUOTE_GMEM_PACK_TYPE uint2
#endif

#ifndef QUOTE_SMEM_PACK_TYPE
#define QUOTE_SMEM_PACK_TYPE uint2
#endif

using QuoteGmemPackT = QUOTE_GMEM_PACK_TYPE;
using QuoteSmemPackT = QUOTE_SMEM_PACK_TYPE;

__device__ void quote_index_packed(const char *file, size_t fileSize,
                                   long *escapeIndex, long *quoteIndex,
                                   char *quoteCarryIndex) {
  using PackT = QuoteGmemPackT;

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;

  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int PACK_BYTES = static_cast<int>(sizeof(PackT));
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;

  static_assert(PACK_BYTES == 2 || PACK_BYTES == 4 || PACK_BYTES == 8 ||
                    PACK_BYTES == 16,
                "QUOTE_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);

  const int tid = threadIdx.x;

  const size_t global_thread_id =
      static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

  const size_t thread_global_base = global_thread_id * BYTES_PER_THREAD;

  uint64_t quote_word = 0;

  if (thread_global_base < fileSize) {
    const uint64_t escaped_word = static_cast<uint64_t>(
        escapeIndex[thread_global_base / BYTES_PER_THREAD]);

#pragma unroll
    for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
      const size_t group_global_base =
          thread_global_base + static_cast<size_t>(group) * PACK_BYTES;

      const PackT packed_bytes_word =
          packed_bytes::load_gmem_pack_or_tail<PackT>(file, fileSize,
                                                      group_global_base);
      const unsigned char *packed_bytes =
          packed_bytes::bytes(packed_bytes_word);

#pragma unroll
      for (int i = 0; i < PACK_BYTES; ++i) {
        const int byte_offset = group * PACK_BYTES + i;
        const size_t global_idx = thread_global_base + byte_offset;

        if (global_idx >= fileSize) {
          break;
        }

        const uint64_t bit = uint64_t{1} << byte_offset;
        if (packed_bytes[i] == static_cast<unsigned char>('"') &&
            (escaped_word & bit) == 0) {
          quote_word |= bit;
        }
      }
    }

    quoteIndex[thread_global_base / BYTES_PER_THREAD] =
        static_cast<long>(quote_word);
  }

  quoteCarryIndex[global_thread_id] = static_cast<char>(
      __popcll(static_cast<unsigned long long>(quote_word)) & 1);
}

__device__ void quote_index_sharemem_packed(const char *file, size_t fileSize,
                                            long *escapeIndex, long *quoteIndex,
                                            char *quoteCarryIndex) {
  using GmemPackT = QuoteGmemPackT;
  using SmemPackT = QuoteSmemPackT;

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
                "QUOTE_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(SMEM_PACK_BYTES == 2 || SMEM_PACK_BYTES == 4 ||
                    SMEM_PACK_BYTES == 8 || SMEM_PACK_BYTES == 16,
                "QUOTE_SMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
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
  //   smem_packed_bytes[owner_tid][packed_group]
  //
  // flattened as:
  //
  //   smem_packed_bytes[owner_tid * PACKED_GROUPS_PER_THREAD + packed_group]
  //
  // This preserves the natural file order in shared memory.
  __shared__ SmemPackT smem_packed_bytes[SMEM_PACKED_ELEMS_PER_BLOCK];

  packed_bytes::stage_file_to_smem</*TRANSPOSED=*/false, GmemPackT, SmemPackT,
                                   BYTES_PER_THREAD, THREADS_PER_BLOCK>(
      file, fileSize, block_start, smem_packed_bytes);

  __syncthreads();

  uint64_t quote_word = 0;

  if (thread_global_base < fileSize) {
    const uint64_t escaped_word = static_cast<uint64_t>(
        escapeIndex[thread_global_base / BYTES_PER_THREAD]);

#pragma unroll
    for (int group = 0; group < SMEM_GROUPS_PER_THREAD; ++group) {
      const int smem_idx = tid * SMEM_GROUPS_PER_THREAD + group;
      const SmemPackT packed_bytes_word = smem_packed_bytes[smem_idx];

      const unsigned char *packed_bytes =
          packed_bytes::bytes(packed_bytes_word);

#pragma unroll
      for (int i = 0; i < SMEM_PACK_BYTES; ++i) {
        const int byte_offset = group * SMEM_PACK_BYTES + i;
        const size_t global_idx = thread_global_base + byte_offset;

        if (global_idx >= fileSize) {
          break;
        }

        const uint64_t bit = uint64_t{1} << byte_offset;

        if (packed_bytes[i] == static_cast<unsigned char>('"') &&
            (escaped_word & bit) == 0) {
          quote_word |= bit;
        }
      }
    }

    quoteIndex[thread_global_base / BYTES_PER_THREAD] =
        static_cast<long>(quote_word);
  }

  quoteCarryIndex[global_thread_id] = static_cast<char>(
      __popcll(static_cast<unsigned long long>(quote_word)) & 1);
}

__device__ void quote_index_sharemem_transposed_packed(const char *file,
                                                       size_t fileSize,
                                                       long *escapeIndex,
                                                       long *quoteIndex,
                                                       char *quoteCarryIndex) {
  using GmemPackT = QuoteGmemPackT;
  using SmemPackT = QuoteSmemPackT;

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
                "QUOTE_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(SMEM_PACK_BYTES == 2 || SMEM_PACK_BYTES == 4 ||
                    SMEM_PACK_BYTES == 8 || SMEM_PACK_BYTES == 16,
                "QUOTE_SMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
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

  uint64_t quote_word = 0;

  if (thread_global_base < fileSize) {
    const uint64_t escaped_word = static_cast<uint64_t>(
        escapeIndex[thread_global_base / BYTES_PER_THREAD]);

#pragma unroll
    for (int group = 0; group < SMEM_GROUPS_PER_THREAD; ++group) {
      const int smem_idx = group * THREADS_PER_BLOCK + tid;
      const SmemPackT packed_bytes_word = smem_packed_bytes[smem_idx];

      const unsigned char *packed_bytes =
          packed_bytes::bytes(packed_bytes_word);

#pragma unroll
      for (int i = 0; i < SMEM_PACK_BYTES; ++i) {
        const int byte_offset = group * SMEM_PACK_BYTES + i;
        const size_t global_idx = thread_global_base + byte_offset;

        if (global_idx >= fileSize) {
          break;
        }

        const uint64_t bit = uint64_t{1} << byte_offset;

        if (packed_bytes[i] == static_cast<unsigned char>('"') &&
            (escaped_word & bit) == 0) {
          quote_word |= bit;
        }
      }
    }

    quoteIndex[thread_global_base / BYTES_PER_THREAD] =
        static_cast<long>(quote_word);
  }

  quoteCarryIndex[global_thread_id] = static_cast<char>(
      __popcll(static_cast<unsigned long long>(quote_word)) & 1);
}

__global__ void quote_index(const char *file, size_t fileSize,
                            long *escapeIndex, long *quoteIndex,
                            char *quoteCarryIndex) {

  quote_index_packed(file, fileSize, escapeIndex, quoteIndex, quoteCarryIndex);

  // quote_index_sharemem_packed(file, fileSize, escapeIndex, quoteIndex,
  //                             quoteCarryIndex);

  // quote_index_sharemem_transposed_packed(file, fileSize, escapeIndex,
  //                                        quoteIndex, quoteCarryIndex);
}

} // namespace gpjson::index::kernels::sharemem
