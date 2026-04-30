#include "gpjson/index/kernels/sharemem/packed-bytes.cuh"
#include "gpjson/log/log.hpp"

namespace gpjson::index::kernels::sharemem {

#ifndef NEWLINE_PACK_TYPE
#define NEWLINE_PACK_TYPE uint2
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
  // This version preserves the original per-thread contiguous ownership:
  //
  //   thread index owns file[start, end)
  //
  // but uses aligned packed global reads where possible.

  constexpr int PACK_BYTES = static_cast<int>(sizeof(NewlinePackT));
  static_assert(PACK_BYTES == 2 || PACK_BYTES == 4 || PACK_BYTES == 8 ||
                    PACK_BYTES == 16,
                "NEWLINE_PACK_TYPE must be 2, 4, 8, or 16 bytes.");

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  const long charsPerThread =
      (static_cast<long>(fileSize) + stride - 1) / stride;

  const long start = static_cast<long>(index) * charsPerThread;
  const long end = start + charsPerThread;

  int count = 0;
  long i = start;

  for (; i < end && i < static_cast<long>(fileSize) &&
         (i % PACK_BYTES) != 0;
       ++i) {
    if (file[i] == '\n') {
      count += 1;
    }
  }

  const NewlinePackT *file_packed =
      reinterpret_cast<const NewlinePackT *>(file);

  for (; i + PACK_BYTES <= end &&
         i + PACK_BYTES <= static_cast<long>(fileSize);
       i += PACK_BYTES) {
    const NewlinePackT packed = file_packed[i / PACK_BYTES];
    count += count_newlines_in_pack(packed);
  }

  for (; i < end && i < static_cast<long>(fileSize); ++i) {
    if (file[i] == '\n') {
      count += 1;
    }
  }

  newlineCountIndex[index] = count;

#ifdef GPJSON_CPP_DEBUG
  int expected_count = 0;
  for (long j = start; j < end && j < fileSize; ++j) {
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
