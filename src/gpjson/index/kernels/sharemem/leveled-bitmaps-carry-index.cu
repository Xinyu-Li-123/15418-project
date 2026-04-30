#include "gpjson/index/kernels/sharemem.cuh"
#include "gpjson/index/kernels/sharemem/packed-bytes.cuh"
#include "gpjson/log/log.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace gpjson::index::kernels::sharemem {

#ifndef LBM_GMEM_PACK_TYPE
#define LBM_GMEM_PACK_TYPE uint4
#endif

#ifndef LBM_SMEM_PACK_TYPE
#define LBM_SMEM_PACK_TYPE uint2
#endif

#ifdef FORCED_GMEM_PACK_TYPE
using LbmGmemPackT = FORCED_GMEM_PACK_TYPE;
#else
using LbmGmemPackT = LBM_GMEM_PACK_TYPE;
#endif

#ifdef FORCED_SMEM_PACK_TYPE
using LbmSmemPackT = FORCED_SMEM_PACK_TYPE;
#else
using LbmSmemPackT = LBM_SMEM_PACK_TYPE;
#endif

namespace {

template <bool TRANSPOSED, typename SmemPackT>
__device__ __forceinline__ void
compute_level_from_smem(size_t fileSize, const long *stringIndex,
                        char *leveledBitmapsAuxIndex,
                        const SmemPackT *smem_packed_bytes) {
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

  signed char level = 0;

  if (thread_global_base < fileSize) {
    const uint64_t string = static_cast<uint64_t>(
        stringIndex[thread_global_base / BYTES_PER_THREAD]);

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

        if ((string_pack_mask & (1u << i)) != 0u) {
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

} // namespace

__device__ void
leveled_bitmaps_carry_index_packed_skip_str(const char *file, size_t fileSize,
                                            const long *stringIndex,
                                            char *leveledBitmapsAuxIndex) {
  using PackT = LbmGmemPackT;

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;

  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int PACK_BYTES = static_cast<int>(sizeof(PackT));
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;

  static_assert(PACK_BYTES == 2 || PACK_BYTES == 4 || PACK_BYTES == 8 ||
                    PACK_BYTES == 16,
                "LBM_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);

  const int tid = threadIdx.x;

  const size_t global_thread_id =
      static_cast<size_t>(blockIdx.x) * THREADS_PER_BLOCK + tid;

  const size_t thread_global_base = global_thread_id * BYTES_PER_THREAD;

  signed char level = 0;

  if (thread_global_base < fileSize) {
    const uint64_t string = static_cast<uint64_t>(
        stringIndex[thread_global_base / BYTES_PER_THREAD]);

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

      if (string_pack_mask == PACK_STRING_MASK) {
        continue;
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

        if ((string_pack_mask & (1u << i)) != 0u) {
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

__device__ void
leveled_bitmaps_carry_index_packed(const char *file, size_t fileSize,
                                   const long *stringIndex,
                                   char *leveledBitmapsAuxIndex) {
  using PackT = LbmGmemPackT;

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;

  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int PACK_BYTES = static_cast<int>(sizeof(PackT));
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;

  static_assert(PACK_BYTES == 2 || PACK_BYTES == 4 || PACK_BYTES == 8 ||
                    PACK_BYTES == 16,
                "LBM_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);

  const int tid = threadIdx.x;

  const size_t global_thread_id =
      static_cast<size_t>(blockIdx.x) * THREADS_PER_BLOCK + tid;

  const size_t thread_global_base = global_thread_id * BYTES_PER_THREAD;

  signed char level = 0;

  if (thread_global_base < fileSize) {
    const uint64_t string = static_cast<uint64_t>(
        stringIndex[thread_global_base / BYTES_PER_THREAD]);

#pragma unroll
    for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
      const size_t group_global_base =
          thread_global_base + static_cast<size_t>(group) * PACK_BYTES;

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
        const int byte_offset = group * PACK_BYTES + i;
        const size_t global_idx = thread_global_base + byte_offset;

        if (global_idx >= fileSize) {
          break;
        }

        if ((string & (uint64_t{1} << byte_offset)) != 0) {
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

__device__ void
leveled_bitmaps_carry_index_sharemem_packed(const char *file, size_t fileSize,
                                            const long *stringIndex,
                                            char *leveledBitmapsAuxIndex) {
  using GmemPackT = LbmGmemPackT;
  using SmemPackT = LbmSmemPackT;

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
                "LBM_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(SMEM_PACK_BYTES == 2 || SMEM_PACK_BYTES == 4 ||
                    SMEM_PACK_BYTES == 8 || SMEM_PACK_BYTES == 16,
                "LBM_SMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
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

  compute_level_from_smem</*TRANSPOSED=*/false, SmemPackT>(
      fileSize, stringIndex, leveledBitmapsAuxIndex, smem_packed_bytes);
}

__device__ void leveled_bitmaps_carry_index_sharemem_transposed_packed(
    const char *file, size_t fileSize, const long *stringIndex,
    char *leveledBitmapsAuxIndex) {
  using GmemPackT = LbmGmemPackT;
  using SmemPackT = LbmSmemPackT;

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
                "LBM_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(SMEM_PACK_BYTES == 2 || SMEM_PACK_BYTES == 4 ||
                    SMEM_PACK_BYTES == 8 || SMEM_PACK_BYTES == 16,
                "LBM_SMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
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

  compute_level_from_smem</*TRANSPOSED=*/true, SmemPackT>(
      fileSize, stringIndex, leveledBitmapsAuxIndex, smem_packed_bytes);
}

__global__ void leveled_bitmaps_carry_index(const char *file, size_t fileSize,
                                            const long *stringIndex,
                                            char *leveledBitmapsAuxIndex) {
  // Direct packed, with skip-string group load elision.
  //
  // leveled_bitmaps_carry_index_packed_skip_str(file, fileSize, stringIndex,
  //                                             leveledBitmapsAuxIndex);

  // Direct packed.
  //
  leveled_bitmaps_carry_index_packed(file, fileSize, stringIndex,
                                     leveledBitmapsAuxIndex);

  // Shared-memory staged, non-transposed.
  //
  // leveled_bitmaps_carry_index_sharemem_packed(file, fileSize, stringIndex,
  //                                             leveledBitmapsAuxIndex);

  // Shared-memory staged, transposed.
  //
  // leveled_bitmaps_carry_index_sharemem_transposed_packed(
  //     file, fileSize, stringIndex, leveledBitmapsAuxIndex);
}

} // namespace gpjson::index::kernels::sharemem
