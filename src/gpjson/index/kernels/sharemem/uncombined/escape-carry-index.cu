#include "gpjson/index/kernels/sharemem.cuh"
#include "gpjson/index/kernels/sharemem/packed-bytes.cuh"
#include "gpjson/log/log.hpp"
#include "gpjson/profiler/profiler.hpp"

#include <cassert>
#include <cstddef>
#include <cstdio>
namespace gpjson::index::kernels::sharemem {

#ifndef ESCAPE_CARRY_GMEM_PACK_TYPE
#define ESCAPE_CARRY_GMEM_PACK_TYPE uint2
#endif

#ifndef ESCAPE_CARRY_SMEM_PACK_TYPE
#define ESCAPE_CARRY_SMEM_PACK_TYPE uint2
#endif

using EscapeCarryGmemPackT = ESCAPE_CARRY_GMEM_PACK_TYPE;
using EscapeCarrySmemPackT = ESCAPE_CARRY_SMEM_PACK_TYPE;

__device__ void escape_carry_index_packed_forward_read(const char *file,
                                                       size_t fileSize,
                                                       char *escapeCarryIndex) {
  using PackT = EscapeCarryGmemPackT;

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;

  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int PACK_BYTES = static_cast<int>(sizeof(PackT));
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;

  static_assert(PACK_BYTES == 2 || PACK_BYTES == 4 || PACK_BYTES == 8 ||
                    PACK_BYTES == 16,
                "ESCAPE_CARRY_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);

  const int tid = threadIdx.x;

  const size_t global_thread_id =
      static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

  const size_t thread_global_base = global_thread_id * BYTES_PER_THREAD;

  char carry = 0;

#pragma unroll
  for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
    const size_t group_global_base =
        thread_global_base + static_cast<size_t>(group) * PACK_BYTES;

    if (group_global_base >= fileSize) {
      break;
    }

    const PackT packed_bytes_word = packed_bytes::load_gmem_pack_or_tail<PackT>(
        file, fileSize, group_global_base);
    const unsigned char *packed_bytes = packed_bytes::bytes(packed_bytes_word);

#pragma unroll
    for (int b = 0; b < PACK_BYTES; ++b) {
      const size_t global_idx = group_global_base + b;

      if (global_idx >= fileSize) {
        break;
      }

      if (packed_bytes[b] == static_cast<unsigned char>('\\')) {
        carry ^= 1;
      } else {
        carry = 0;
      }
    }
  }

  const int index = blockIdx.x * blockDim.x + tid;

  if (thread_global_base < fileSize) {
    escapeCarryIndex[index] = carry;
  }
}

__device__ void
escape_carry_index_packed_backward_read(const char *file, size_t fileSize,
                                        char *escapeCarryIndex) {
  using PackT = EscapeCarryGmemPackT;

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;

  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int PACK_BYTES = static_cast<int>(sizeof(PackT));
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;

  static_assert(PACK_BYTES == 2 || PACK_BYTES == 4 || PACK_BYTES == 8 ||
                    PACK_BYTES == 16,
                "ESCAPE_CARRY_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);

  const int tid = threadIdx.x;

  const size_t global_thread_id =
      static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

  const size_t thread_global_base = global_thread_id * BYTES_PER_THREAD;

  char carry = 0;
  bool done = false;

#pragma unroll
  for (int group = PACKED_GROUPS_PER_THREAD - 1; group >= 0; --group) {
    if (done) {
      break;
    }

    const size_t group_global_base =
        thread_global_base + static_cast<size_t>(group) * PACK_BYTES;

    // Last partial tile: higher groups of some threads may be past EOF.
    if (group_global_base >= fileSize) {
      continue;
    }

    const PackT packed_bytes_word = packed_bytes::load_gmem_pack_or_tail<PackT>(
        file, fileSize, group_global_base);
    const unsigned char *packed_bytes = packed_bytes::bytes(packed_bytes_word);

#pragma unroll
    for (int b = PACK_BYTES - 1; b >= 0; --b) {
      const size_t global_idx = group_global_base + b;

      if (global_idx >= fileSize) {
        continue;
      }

      if (packed_bytes[b] == static_cast<unsigned char>('\\')) {
        carry ^= 1;
      } else {
        done = true;
        break;
      }
    }
  }

  const int index = blockIdx.x * blockDim.x + tid;

  if (thread_global_base < fileSize) {
    escapeCarryIndex[index] = carry;
  }
}

__device__ void escape_carry_index_sharemem_packed_forward_read(
    const char *file, size_t fileSize, char *escapeCarryIndex) {
  using GmemPackT = EscapeCarryGmemPackT;
  using SmemPackT = EscapeCarrySmemPackT;

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = THREADS_PER_BLOCK * BYTES_PER_THREAD;
  constexpr int GMEM_PACK_BYTES = static_cast<int>(sizeof(GmemPackT));
  constexpr int SMEM_PACK_BYTES = static_cast<int>(sizeof(SmemPackT));
  constexpr int SMEM_GROUPS_PER_THREAD = BYTES_PER_THREAD / SMEM_PACK_BYTES;
  constexpr int SMEM_PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / SMEM_PACK_BYTES;

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  static_assert(GMEM_PACK_BYTES == 2 || GMEM_PACK_BYTES == 4 ||
                    GMEM_PACK_BYTES == 8 || GMEM_PACK_BYTES == 16,
                "ESCAPE_CARRY_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(SMEM_PACK_BYTES == 2 || SMEM_PACK_BYTES == 4 ||
                    SMEM_PACK_BYTES == 8 || SMEM_PACK_BYTES == 16,
                "ESCAPE_CARRY_SMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");

  const int tid = threadIdx.x;
  const size_t block_start = static_cast<size_t>(blockIdx.x) * CHUNK_SIZE;

  __shared__ SmemPackT smem_packed_bytes[SMEM_PACKED_ELEMS_PER_BLOCK];

  packed_bytes::stage_file_to_smem</*TRANSPOSED=*/false, GmemPackT, SmemPackT,
                                   BYTES_PER_THREAD, THREADS_PER_BLOCK>(
      file, fileSize, block_start, smem_packed_bytes);

  __syncthreads();

  const int local_start = tid * BYTES_PER_THREAD;
  const size_t thread_global_base = block_start + local_start;
  char carry = 0;

  for (int group = 0; group < SMEM_GROUPS_PER_THREAD; ++group) {
    const size_t group_global_base =
        thread_global_base + static_cast<size_t>(group) * SMEM_PACK_BYTES;

    if (group_global_base >= fileSize) {
      break;
    }

    const int smem_idx = tid * SMEM_GROUPS_PER_THREAD + group;
    const SmemPackT packed_bytes_word = smem_packed_bytes[smem_idx];
    const unsigned char *packed_bytes = packed_bytes::bytes(packed_bytes_word);

#pragma unroll
    for (int b = 0; b < SMEM_PACK_BYTES; ++b) {
      const size_t global_idx = group_global_base + b;
      if (global_idx >= fileSize) {
        break;
      }

      if (packed_bytes[b] == static_cast<unsigned char>('\\')) {
        carry ^= 1;
      } else {
        carry = 0;
      }
    }
  }

  const int index = blockIdx.x * blockDim.x + tid;
  if (thread_global_base < fileSize) {
    escapeCarryIndex[index] = carry;
  }
}

__device__ void escape_carry_index_sharemem_packed_backward_read(
    const char *file, size_t fileSize, char *escapeCarryIndex) {
  using GmemPackT = EscapeCarryGmemPackT;
  using SmemPackT = EscapeCarrySmemPackT;

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = THREADS_PER_BLOCK * BYTES_PER_THREAD;
  constexpr int GMEM_PACK_BYTES = static_cast<int>(sizeof(GmemPackT));
  constexpr int SMEM_PACK_BYTES = static_cast<int>(sizeof(SmemPackT));
  constexpr int SMEM_GROUPS_PER_THREAD = BYTES_PER_THREAD / SMEM_PACK_BYTES;
  constexpr int SMEM_PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / SMEM_PACK_BYTES;

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  static_assert(GMEM_PACK_BYTES == 2 || GMEM_PACK_BYTES == 4 ||
                    GMEM_PACK_BYTES == 8 || GMEM_PACK_BYTES == 16,
                "ESCAPE_CARRY_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(SMEM_PACK_BYTES == 2 || SMEM_PACK_BYTES == 4 ||
                    SMEM_PACK_BYTES == 8 || SMEM_PACK_BYTES == 16,
                "ESCAPE_CARRY_SMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");

  const int tid = threadIdx.x;
  const size_t block_start = static_cast<size_t>(blockIdx.x) * CHUNK_SIZE;

  __shared__ SmemPackT smem_packed_bytes[SMEM_PACKED_ELEMS_PER_BLOCK];

  packed_bytes::stage_file_to_smem</*TRANSPOSED=*/false, GmemPackT, SmemPackT,
                                   BYTES_PER_THREAD, THREADS_PER_BLOCK>(
      file, fileSize, block_start, smem_packed_bytes);

  __syncthreads();

  const int local_start = tid * BYTES_PER_THREAD;
  const size_t thread_global_base = block_start + local_start;
  char carry = 0;
  bool done = false;

  for (int group = SMEM_GROUPS_PER_THREAD - 1; group >= 0 && !done; --group) {
    const size_t group_global_base =
        thread_global_base + static_cast<size_t>(group) * SMEM_PACK_BYTES;

    if (group_global_base >= fileSize) {
      continue;
    }

    const int smem_idx = tid * SMEM_GROUPS_PER_THREAD + group;
    const SmemPackT packed_bytes_word = smem_packed_bytes[smem_idx];
    const unsigned char *packed_bytes = packed_bytes::bytes(packed_bytes_word);

#pragma unroll
    for (int b = SMEM_PACK_BYTES - 1; b >= 0; --b) {
      const size_t global_idx = group_global_base + b;
      if (global_idx >= fileSize) {
        continue;
      }

      if (packed_bytes[b] == static_cast<unsigned char>('\\')) {
        carry ^= 1;
      } else {
        done = true;
        break;
      }
    }
  }

  const int index = blockIdx.x * blockDim.x + tid;
  if (thread_global_base < fileSize) {
    escapeCarryIndex[index] = carry;
  }
}

__device__ void escape_carry_index_sharemem_transposed_packed_forward_read(
    const char *file, size_t fileSize, char *escapeCarryIndex) {
  using GmemPackT = EscapeCarryGmemPackT;
  using SmemPackT = EscapeCarrySmemPackT;

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = THREADS_PER_BLOCK * BYTES_PER_THREAD;
  constexpr int GMEM_PACK_BYTES = static_cast<int>(sizeof(GmemPackT));
  constexpr int SMEM_PACK_BYTES = static_cast<int>(sizeof(SmemPackT));
  constexpr int SMEM_GROUPS_PER_THREAD = BYTES_PER_THREAD / SMEM_PACK_BYTES;
  constexpr int SMEM_PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / SMEM_PACK_BYTES;

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  static_assert(GMEM_PACK_BYTES == 2 || GMEM_PACK_BYTES == 4 ||
                    GMEM_PACK_BYTES == 8 || GMEM_PACK_BYTES == 16,
                "ESCAPE_CARRY_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(SMEM_PACK_BYTES == 2 || SMEM_PACK_BYTES == 4 ||
                    SMEM_PACK_BYTES == 8 || SMEM_PACK_BYTES == 16,
                "ESCAPE_CARRY_SMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");

  const int tid = threadIdx.x;
  const size_t block_start = static_cast<size_t>(blockIdx.x) * CHUNK_SIZE;

  __shared__ SmemPackT smem_packed_bytes[SMEM_PACKED_ELEMS_PER_BLOCK];

  packed_bytes::stage_file_to_smem</*TRANSPOSED=*/true, GmemPackT, SmemPackT,
                                   BYTES_PER_THREAD, THREADS_PER_BLOCK>(
      file, fileSize, block_start, smem_packed_bytes);

  __syncthreads();

  char carry = 0;

  const size_t thread_global_base =
      block_start + static_cast<size_t>(tid) * BYTES_PER_THREAD;

#pragma unroll
  for (int group = 0; group < SMEM_GROUPS_PER_THREAD; ++group) {
    const size_t group_global_base =
        thread_global_base + static_cast<size_t>(group) * SMEM_PACK_BYTES;

    if (group_global_base >= fileSize) {
      break;
    }

    const int smem_idx = group * THREADS_PER_BLOCK + tid;
    const SmemPackT packed_bytes_word = smem_packed_bytes[smem_idx];
    const unsigned char *packed_bytes = packed_bytes::bytes(packed_bytes_word);

#pragma unroll
    for (int b = 0; b < SMEM_PACK_BYTES; ++b) {
      const size_t global_idx = group_global_base + b;

      if (global_idx >= fileSize) {
        break;
      }

      if (packed_bytes[b] == static_cast<unsigned char>('\\')) {
        carry ^= 1;
      } else {
        carry = 0;
      }
    }
  }

  const int index = blockIdx.x * blockDim.x + tid;

  // Coalesced global write.
  if (thread_global_base < fileSize) {
    escapeCarryIndex[index] = carry;
  }
}

__device__ void escape_carry_index_sharemem_transposed_packed_backward_read(
    const char *file, size_t fileSize, char *escapeCarryIndex) {
  using GmemPackT = EscapeCarryGmemPackT;
  using SmemPackT = EscapeCarrySmemPackT;

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = THREADS_PER_BLOCK * BYTES_PER_THREAD;
  constexpr int GMEM_PACK_BYTES = static_cast<int>(sizeof(GmemPackT));
  constexpr int SMEM_PACK_BYTES = static_cast<int>(sizeof(SmemPackT));
  constexpr int SMEM_GROUPS_PER_THREAD = BYTES_PER_THREAD / SMEM_PACK_BYTES;
  constexpr int SMEM_PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / SMEM_PACK_BYTES;

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  static_assert(GMEM_PACK_BYTES == 2 || GMEM_PACK_BYTES == 4 ||
                    GMEM_PACK_BYTES == 8 || GMEM_PACK_BYTES == 16,
                "ESCAPE_CARRY_GMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(SMEM_PACK_BYTES == 2 || SMEM_PACK_BYTES == 4 ||
                    SMEM_PACK_BYTES == 8 || SMEM_PACK_BYTES == 16,
                "ESCAPE_CARRY_SMEM_PACK_TYPE must be 2, 4, 8, or 16 bytes.");

  const int tid = threadIdx.x;
  const size_t block_start = static_cast<size_t>(blockIdx.x) * CHUNK_SIZE;

  __shared__ SmemPackT smem_packed_bytes[SMEM_PACKED_ELEMS_PER_BLOCK];

  packed_bytes::stage_file_to_smem</*TRANSPOSED=*/true, GmemPackT, SmemPackT,
                                   BYTES_PER_THREAD, THREADS_PER_BLOCK>(
      file, fileSize, block_start, smem_packed_bytes);

  __syncthreads();

  const int local_start = tid * BYTES_PER_THREAD;
  const size_t thread_global_base = block_start + local_start;

  char carry = 0;
  bool done = false;

#pragma unroll
  for (int group = SMEM_GROUPS_PER_THREAD - 1; group >= 0; --group) {
    if (done) {
      break;
    }

    const size_t group_global_base =
        thread_global_base + static_cast<size_t>(group) * SMEM_PACK_BYTES;

    // If this whole packed group starts past EOF, skip it. This matters for
    // the last partial tile: the highest groups of some threads may be padding.
    if (group_global_base >= fileSize) {
      continue;
    }

    const int smem_idx = group * THREADS_PER_BLOCK + tid;
    const SmemPackT packed_bytes_word = smem_packed_bytes[smem_idx];
    const unsigned char *packed_bytes = packed_bytes::bytes(packed_bytes_word);

#pragma unroll
    for (int b = SMEM_PACK_BYTES - 1; b >= 0; --b) {
      const size_t global_idx = group_global_base + b;

      if (global_idx >= fileSize) {
        continue;
      }

      if (packed_bytes[b] == static_cast<unsigned char>('\\')) {
        carry ^= 1;
      } else {
        done = true;
        break;
      }
    }
  }

  const int index = blockIdx.x * blockDim.x + tid;

  // Coalesced global write.
  if (thread_global_base < fileSize) {
    escapeCarryIndex[index] = carry;
  }
}

__global__ void escape_carry_index(const char *file, size_t fileSize,
                                   char *escapeCarryIndex) {

  // escape_carry_index_packed_forward_read(file, fileSize, escapeCarryIndex);

  // While we theoretically do less work with backward read, it makes the kernel
  // more branchy
  escape_carry_index_packed_backward_read(file, fileSize, escapeCarryIndex);

  // escape_carry_index_packed_backward_read(file, fileSize,
  //                                                  escapeCarryIndex);

  // escape_carry_index_sharemem_packed_forward_read(file, fileSize,
  //                                                 escapeCarryIndex);

  // escape_carry_index_sharemem_packed_backward_read(file, fileSize,
  //                                                  escapeCarryIndex);

  // escape_carry_index_sharemem_transposed_packed_forward_read(file, fileSize,
  //                                                            escapeCarryIndex);

  // NOTE: While both transposed version suffers from uncoalesced smem write due
  // to transpose, this version optimzies for the common case: when we read from
  // smem to compute carry, we commonly only read the last few bytes, thust most
  // of the time reading one packed bytes will do the job. This outweigh the
  // overhead from coalesced smem write.
  // escape_carry_index_sharemem_transposed_packed_backward_read(file, fileSize,
  //                                                             escapeCarryIndex);
}
} // namespace gpjson::index::kernels::sharemem
