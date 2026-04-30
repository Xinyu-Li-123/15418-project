#include "gpjson/index/kernels/sharemem/packed-bytes.cuh"
#include "gpjson/log/log.hpp"

namespace gpjson::index::kernels::sharemem {

#ifndef NEWLINE_PACK_TYPE
#define NEWLINE_PACK_TYPE uint2
#endif

using NewlinePackT = NEWLINE_PACK_TYPE;

namespace {

template <typename PackT>
__device__ __forceinline__ int count_newlines_in_pack(const PackT packed) {
  constexpr int PACK_BYTES = static_cast<int>(sizeof(PackT));
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

struct CountNewlineScalar {
  int *count;

  __device__ __forceinline__ void operator()(long, unsigned char value) const {
    if (value == static_cast<unsigned char>('\n')) {
      *count += 1;
    }
  }
};

template <typename PackT> struct CountNewlinePack {
  int *count;

  __device__ __forceinline__ void operator()(long, PackT packed) const {
    *count += count_newlines_in_pack(packed);
  }
};

} // namespace

__device__ void newline_count_index_per_thread_packed(const char *file,
                                                      size_t fileSize,
                                                      int *newlineCountIndex) {
  using PackT = NewlinePackT;

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

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  const long charsPerThread =
      (static_cast<long>(fileSize) + stride - 1) / stride;

  const long start = static_cast<long>(index) * charsPerThread;
  const long end = start + charsPerThread;

  int count = 0;
  packed_bytes::for_each_aligned_pack_in_range<PackT>(
      file, fileSize, start, end, CountNewlineScalar{&count},
      CountNewlinePack<PackT>{&count});

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
