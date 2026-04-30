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

__device__ __forceinline__ void emit_newlines_in_pack(const NewlinePackT packed,
                                                      const long global_base,
                                                      int &offset,
                                                      long *newlineIndex) {
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
  // This version uses fixed per-thread contiguous ownership:
  //
  //   thread index owns file[index * 64, index * 64 + 64)
  //
  // so newlineIndex remains globally sorted by file position.

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

  int offset = newlineCountIndex[index];

  const size_t thread_global_base =
      static_cast<size_t>(index) * BYTES_PER_THREAD;

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
    emit_newlines_in_pack(packed, static_cast<long>(group_global_base), offset,
                          newlineIndex);
  }
}

__global__ void newline_index(const char *file, size_t fileSize,
                              int *newlineCountIndex, long *newlineIndex) {
  newline_index_per_thread_offset_packed(file, fileSize, newlineCountIndex,
                                         newlineIndex);
}

} // namespace gpjson::index::kernels::sharemem
