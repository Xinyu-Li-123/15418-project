#include "gpjson/log/log.hpp"

namespace gpjson::index::kernels::sharemem {

namespace {

__device__ __forceinline__ int count_newlines_in_uint2(const uint2 packed) {
  const unsigned char *chars = reinterpret_cast<const unsigned char *>(&packed);

  int count = 0;

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    count += chars[i] == static_cast<unsigned char>('\n');
  }

  return count;
}

} // namespace

__device__ void newline_count_index_per_thread_packed(const char *file,
                                                      int fileSize,
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
  // but uses packed 8-byte global reads where possible.

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  const long charsPerThread =
      (static_cast<long>(fileSize) + stride - 1) / stride;

  const long start = static_cast<long>(index) * charsPerThread;
  const long end = start + charsPerThread;

  int count = 0;
  long i = start;

  // Scalar peel until the address is 8-byte aligned.
  //
  // This keeps reinterpret_cast<const uint2 *>(file) reads aligned by only
  // reading packed words at global byte indices divisible by 8.
  for (; i < end && i < fileSize && (i & 7L) != 0; ++i) {
    if (file[i] == '\n') {
      count += 1;
    }
  }

  const uint2 *file_packed = reinterpret_cast<const uint2 *>(file);

  // Packed 8-byte loop.
  for (; i + 7 < end && i + 7 < fileSize; i += 8) {
    const uint2 packed = file_packed[i >> 3];
    count += count_newlines_in_uint2(packed);
  }

  // Scalar tail.
  for (; i < end && i < fileSize; ++i) {
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
__global__ void newline_count_index(const char *file, int fileSize,
                                    int *newlineCountIndex) {
  newline_count_index_per_thread_packed(file, fileSize, newlineCountIndex);
}

} // namespace gpjson::index::kernels::sharemem
