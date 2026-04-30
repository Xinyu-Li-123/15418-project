#include "gpjson/index/kernels/sharemem/packed-bytes.cuh"
#include "gpjson/log/log.hpp"

namespace gpjson::index::kernels::sharemem {

#ifndef NEWLINE_PACK_TYPE
#define NEWLINE_PACK_TYPE uint2
#endif

using NewlinePackT = NEWLINE_PACK_TYPE;

namespace {

template <typename PackT>
__device__ __forceinline__ void
emit_newlines_in_pack(const PackT packed, const long global_base, int &offset,
                      long *newlineIndex) {
  constexpr int PACK_BYTES = static_cast<int>(sizeof(PackT));
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

struct EmitNewlineScalar {
  int *offset;
  long *newlineIndex;

  __device__ __forceinline__ void operator()(long idx,
                                             unsigned char value) const {
    if (value == static_cast<unsigned char>('\n')) {
      newlineIndex[*offset + 1] = idx;
      *offset += 1;
    }
  }
};

template <typename PackT> struct EmitNewlinePack {
  int *offset;
  long *newlineIndex;

  __device__ __forceinline__ void operator()(long idx, PackT packed) const {
    emit_newlines_in_pack(packed, idx, *offset, newlineIndex);
  }
};

} // namespace

__device__ void newline_index_per_thread_offset_packed(const char *file,
                                                       size_t fileSize,
                                                       int *newlineCountIndex,
                                                       long *newlineIndex) {
  using PackT = NewlinePackT;

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

  packed_bytes::for_each_aligned_pack_in_range<PackT>(
      file, fileSize, start, end, EmitNewlineScalar{&offset, newlineIndex},
      EmitNewlinePack<PackT>{&offset, newlineIndex});
}

__global__ void newline_index(const char *file, size_t fileSize,
                              int *newlineCountIndex, long *newlineIndex) {
  newline_index_per_thread_offset_packed(file, fileSize, newlineCountIndex,
                                         newlineIndex);
}

} // namespace gpjson::index::kernels::sharemem
