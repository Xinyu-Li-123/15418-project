#include "gpjson/log/log.hpp"

namespace gpjson::index::kernels::sharemem {

namespace {

__device__ __forceinline__ void emit_newlines_in_uint2(const uint2 packed,
                                                       const long global_base,
                                                       int &offset,
                                                       long *newlineIndex) {
  const unsigned char *chars = reinterpret_cast<const unsigned char *>(&packed);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (chars[i] == static_cast<unsigned char>('\n')) {
      newlineIndex[offset + 1] = global_base + i;
      offset += 1;
    }
  }
}

} // namespace

__device__ void newline_index_per_thread_offset_packed(const char *file,
                                                       int fileSize,
                                                       int *newlineCountIndex,
                                                       long *newlineIndex) {
  // REQUIRES:
  //   newlineCountIndex has already been exclusive-scanned.
  //
  // Therefore:
  //
  //   newlineCountIndex[index] = output offset for this CUDA thread
  //
  // The leading sentinel newlineIndex[0] is written by the caller, so physical
  // newline positions start at newlineIndex[offset + 1].
  //
  // This version preserves the original per-thread contiguous ownership:
  //
  //   thread index owns file[start, end)
  //
  // so newlineIndex remains globally sorted by file position.

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  int offset = newlineCountIndex[index];

  const long charsPerThread =
      (static_cast<long>(fileSize) + stride - 1) / stride;

  const long start = static_cast<long>(index) * charsPerThread;
  const long end = start + charsPerThread;

  long i = start;

  // Scalar peel until the address is 8-byte aligned.
  for (; i < end && i < fileSize && (i & 7L) != 0; ++i) {
    if (file[i] == '\n') {
      newlineIndex[offset + 1] = i;
      offset += 1;
    }
  }

  const uint2 *file_packed = reinterpret_cast<const uint2 *>(file);

  // Packed 8-byte loop.
  for (; i + 7 < end && i + 7 < fileSize; i += 8) {
    const uint2 packed = file_packed[i >> 3];
    emit_newlines_in_uint2(packed, i, offset, newlineIndex);
  }

  // Scalar tail.
  for (; i < end && i < fileSize; ++i) {
    if (file[i] == '\n') {
      newlineIndex[offset + 1] = i;
      offset += 1;
    }
  }
}

__global__ void newline_index(const char *file, int fileSize,
                              int *newlineCountIndex, long *newlineIndex) {
  newline_index_per_thread_offset_packed(file, fileSize, newlineCountIndex,
                                         newlineIndex);
}

} // namespace gpjson::index::kernels::sharemem
