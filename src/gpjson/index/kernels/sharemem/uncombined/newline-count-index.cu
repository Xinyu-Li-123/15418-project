#include "gpjson/index/kernels/sharemem/packed-bytes.cuh"
#include "gpjson/log/log.hpp"

namespace gpjson::index::kernels::sharemem {

#ifndef NEWLINE_PACK_TYPE
#define NEWLINE_PACK_TYPE uint4
#endif

#ifdef FORCED_GMEM_PACK_TYPE
using NewlinePackT = FORCED_GMEM_PACK_TYPE;
#else
using NewlinePackT = NEWLINE_PACK_TYPE;
#endif

namespace {

__device__ __forceinline__ int
count_newlines_in_pack(const NewlinePackT packed) {
  constexpr int PACK_BYTES = static_cast<int>(sizeof(NewlinePackT));
  static_assert(PACK_BYTES == 2 || PACK_BYTES == 4 || PACK_BYTES == 8 ||
                    PACK_BYTES == 16,
                "NEWLINE_PACK_TYPE must be 2, 4, 8, or 16 bytes.");

  const unsigned char *chars = packed_bytes::bytes(packed);

  int count = 0;

#pragma unroll
  for (int i = 0; i < PACK_BYTES; ++i) {
    count += chars[i] == static_cast<unsigned char>('\n');
  }

  return count;
}

} // namespace

__device__ void newline_count_index_per_thread_packed(const char *file,
                                                      size_t fileSize,
                                                      int *newlineCountIndex) {
  // REQUIRES:
  //   length of newlineCountIndex == gridDim.x * blockDim.x + 1
  //
  // This computes one newline count per CUDA thread. The final +1 slot is
  // initialized by the caller before exclusive scan.
  //
  // This version uses fixed per-thread contiguous ownership:
  //
  //   thread index owns file[index * 64, index * 64 + 64)
  //
  // but uses aligned packed global reads where possible.

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int PACK_BYTES = static_cast<int>(sizeof(NewlinePackT));
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;

  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  static_assert(PACK_BYTES == 2 || PACK_BYTES == 4 || PACK_BYTES == 8 ||
                    PACK_BYTES == 16,
                "NEWLINE_PACK_TYPE must be 2, 4, 8, or 16 bytes.");
  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t thread_global_base =
      static_cast<size_t>(index) * BYTES_PER_THREAD;

  int count = 0;

#pragma unroll
  for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
    const size_t group_global_base =
        thread_global_base + static_cast<size_t>(group) * PACK_BYTES;

    if (group_global_base >= fileSize) {
      break;
    }

    const NewlinePackT packed =
        packed_bytes::load_gmem_pack_or_tail<NewlinePackT>(file, fileSize,
                                                           group_global_base);
    count += count_newlines_in_pack(packed);
  }

  newlineCountIndex[index] = count;

#ifdef GPJSON_CPP_DEBUG
  int expected_count = 0;
  for (size_t j = thread_global_base;
       j < thread_global_base + BYTES_PER_THREAD && j < fileSize; ++j) {
    if (file[j] == '\n') {
      expected_count += 1;
    }
  }

  Check(expected_count == count,
        "Incorrect per-thread newline count at thread %d. Expect %d, got %d.",
        index, expected_count, count);
#endif
}

/**
 * A per-thread newline count index.
 *
 * After this kernel:
 *
 *   newlineCountIndex[i] = number of newline characters owned by thread i
 *
 * The builder then runs an exclusive scan over:
 *
 *   newlineCountIndex[0 : num_cuda_threads + 1]
 *
 * After scan:
 *
 *   newlineCountIndex[i]                 = output offset for thread i
 *   newlineCountIndex[num_cuda_threads]  = total number of newline chars
 */
__global__ void newline_count_index(const char *file, size_t fileSize,
                                    int *newlineCountIndex) {
  newline_count_index_per_thread_packed(file, fileSize, newlineCountIndex);
}

} // namespace gpjson::index::kernels::sharemem
