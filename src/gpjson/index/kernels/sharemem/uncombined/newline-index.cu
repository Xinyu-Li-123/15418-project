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

__device__ __forceinline__ void
emit_newlines_in_pack(const NewlinePackT packed, const long global_base,
                      int &offset, long *newlineIndex) {
  constexpr int PACK_BYTES = static_cast<int>(sizeof(NewlinePackT));
  static_assert(PACK_BYTES == 2 || PACK_BYTES == 4 || PACK_BYTES == 8 ||
                    PACK_BYTES == 16,
                "NEWLINE_PACK_TYPE must be 2, 4, 8, or 16 bytes.");

  const unsigned char *chars = packed_bytes::bytes(packed);

#pragma unroll
  for (int i = 0; i < PACK_BYTES; ++i) {
    if (chars[i] == static_cast<unsigned char>('\n')) {
      newlineIndex[offset + 1] = global_base + i;
      offset += 1;
    }
  }
}

} // namespace

__device__ void newline_index_per_thread_offset_packed(const char *file,
                                                       size_t fileSize,
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

  constexpr int PACK_BYTES = static_cast<int>(sizeof(NewlinePackT));
  static_assert(PACK_BYTES == 2 || PACK_BYTES == 4 || PACK_BYTES == 8 ||
                    PACK_BYTES == 16,
                "NEWLINE_PACK_TYPE must be 2, 4, 8, or 16 bytes.");

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  int offset = newlineCountIndex[index];

  const long charsPerThread =
      (static_cast<long>(fileSize) + stride - 1) / stride;

  const long start = static_cast<long>(index) * charsPerThread;
  const long end = start + charsPerThread;

  long i = start;

  for (; i < end && i < static_cast<long>(fileSize) &&
         (i % PACK_BYTES) != 0;
       ++i) {
    if (file[i] == '\n') {
      newlineIndex[offset + 1] = i;
      offset += 1;
    }
  }

  const NewlinePackT *file_packed =
      reinterpret_cast<const NewlinePackT *>(file);

  for (; i + PACK_BYTES <= end &&
         i + PACK_BYTES <= static_cast<long>(fileSize);
       i += PACK_BYTES) {
    const NewlinePackT packed = file_packed[i / PACK_BYTES];
    emit_newlines_in_pack(packed, i, offset, newlineIndex);
  }

  for (; i < end && i < static_cast<long>(fileSize); ++i) {
    if (file[i] == '\n') {
      newlineIndex[offset + 1] = i;
      offset += 1;
    }
  }
}

__global__ void newline_index(const char *file, size_t fileSize,
                              int *newlineCountIndex, long *newlineIndex) {
  newline_index_per_thread_offset_packed(file, fileSize, newlineCountIndex,
                                         newlineIndex);
}

} // namespace gpjson::index::kernels::sharemem
