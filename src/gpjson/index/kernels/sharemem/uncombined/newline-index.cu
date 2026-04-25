#include "gpjson/log/log.hpp"

namespace gpjson::index::kernels::sharemem {

/**
 * On top of strided read provided in newline_index_sharemem_transposed, we also
 * want to read multiple bytes at a time, i.e. we want packed bytes. To achieve
 * this, we change our write pattern
 *
 *   file_smem[tid][packed_byte_idx][packed_byte_offset] =>
 * file_smem[packed_byte_idx][tid][packed_byte_offset]
 *
 * where
 * - tid=[0, 512)
 * - packed_byte_idx=[0, 8), the idx of uint2
 * - packed_byte_offset=[0, 8), the offset into uint2 as an array of 8 chars
 *
 * This way, each thread can read 64 bytes from shared memory with a stride of
 * 512 bytes, and read 8 bytes at a time, instead of sequentially read a
 * continguous chunk of 64 bytes.
 */
__device__ void
newline_index_sharemem_transposed_packed(const char *file, int fileSize,
                                         int *perTileNewlineOffsetIndex,
                                         long *newlineIndex) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = 32768; // 512 * 64

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int WARP_SIZE = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  constexpr int PACK_BYTES = 8;
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;
  constexpr int PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / PACK_BYTES;

  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % PACK_BYTES == 0);

  const int tid = threadIdx.x;
  const int tile_idx = blockIdx.x;
  const int block_start = tile_idx * CHUNK_SIZE;
  const int tile_offset = perTileNewlineOffsetIndex[tile_idx];

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  // Transposed packed layout:
  //   smem_flags[group][tid]
  // where each entry is one 8-byte packed flag word (0/1 bytes).
  __shared__ uint2
      smem_packed_flags[PACKED_GROUPS_PER_THREAD * THREADS_PER_BLOCK];

  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  // Coalesced global load, then transpose+pack into shared memory.
  for (int p = tid; p < PACKED_ELEMS_PER_BLOCK; p += blockDim.x) {
    const int global_byte_idx = block_start + p * PACK_BYTES;
    const int global_packed_idx = global_byte_idx / PACK_BYTES;

    uint2 packed_bytes = make_uint2(0u, 0u);
    if (global_byte_idx < fileSize) {
      packed_bytes = file_packed_gmem[global_packed_idx];
    }

    const char *packed_chars = reinterpret_cast<const char *>(&packed_bytes);

    unsigned char flag_bytes[PACK_BYTES];
#pragma unroll
    for (int i = 0; i < PACK_BYTES; ++i) {
      const int global_idx = global_byte_idx + i;
      flag_bytes[i] =
          (global_idx < fileSize && packed_chars[i] == '\n') ? 1 : 0;
    }

    uint2 packed_flags_word;
    unsigned char *packed_flag_chars =
        reinterpret_cast<unsigned char *>(&packed_flags_word);
#pragma unroll
    for (int i = 0; i < PACK_BYTES; ++i) {
      packed_flag_chars[i] = flag_bytes[i];
    }

    // Original logical ownership is contiguous 64-byte chunks per thread.
    // Map packed tile index p -> (owner thread, packed group within thread).
    const int owner_tid = p / PACKED_GROUPS_PER_THREAD;
    const int packed_group = p % PACKED_GROUPS_PER_THREAD;

    // Transposed packed shared-memory index: [packed_group][owner_tid]
    const int smem_idx = packed_group * THREADS_PER_BLOCK + owner_tid;
    smem_packed_flags[smem_idx] = packed_flags_word;
  }
  __syncthreads();

  int in_thread_count = 0;

#pragma unroll
  for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
    const int smem_idx = group * THREADS_PER_BLOCK + tid;
    const uint2 packed_flags_word = smem_packed_flags[smem_idx];
    const unsigned char *packed_flags =
        reinterpret_cast<const unsigned char *>(&packed_flags_word);

#pragma unroll
    for (int i = 0; i < PACK_BYTES; ++i) {
      in_thread_count += static_cast<int>(packed_flags[i]);
    }
  }

  __shared__ int per_warp_totals[WARPS_PER_BLOCK];
  __shared__ int per_warp_offsets[WARPS_PER_BLOCK];

  int in_warp_inclusive_offset = in_thread_count;
#pragma unroll
  for (int pass_offset = 1; pass_offset < WARP_SIZE; pass_offset <<= 1) {
    const int other_thread_count =
        __shfl_up_sync(0xffffffffu, in_warp_inclusive_offset, pass_offset);
    if (lane_id >= pass_offset) {
      in_warp_inclusive_offset += other_thread_count;
    }
  }

  const int in_warp_exclusive_offset =
      in_warp_inclusive_offset - in_thread_count;
  const int in_warp_num_newlines =
      __shfl_sync(0xffffffffu, in_warp_inclusive_offset, WARP_SIZE - 1);

  if (lane_id == WARP_SIZE - 1) {
    per_warp_totals[warp_id] = in_warp_num_newlines;
  }
  __syncthreads();

  if (warp_id == 0) {
    int per_warp_inclusive_offset =
        (lane_id < WARPS_PER_BLOCK) ? per_warp_totals[lane_id] : 0;

#pragma unroll
    for (int pass_offset = 1; pass_offset < WARP_SIZE; pass_offset <<= 1) {
      const int other_warp_count =
          __shfl_up_sync(0xffffffffu, per_warp_inclusive_offset, pass_offset);
      if (lane_id >= pass_offset) {
        per_warp_inclusive_offset += other_warp_count;
      }
    }

    if (lane_id < WARPS_PER_BLOCK) {
      per_warp_offsets[lane_id] =
          per_warp_inclusive_offset - per_warp_totals[lane_id];
    }
  }
  __syncthreads();

  int global_offset =
      tile_offset + per_warp_offsets[warp_id] + in_warp_exclusive_offset;

  const int thread_global_base = block_start + tid * BYTES_PER_THREAD;

#pragma unroll
  for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
    const int smem_idx = group * THREADS_PER_BLOCK + tid;
    const uint2 packed_flags_word = smem_packed_flags[smem_idx];
    const unsigned char *packed_flags =
        reinterpret_cast<const unsigned char *>(&packed_flags_word);

    const int group_global_base = thread_global_base + group * PACK_BYTES;

#pragma unroll
    for (int i = 0; i < PACK_BYTES; ++i) {
      const int global_idx = group_global_base + i;
      if (global_idx < fileSize && packed_flags[i]) {
        // Keep GPJSON's leading sentinel at index 0.
        newlineIndex[global_offset + 1] = static_cast<long>(global_idx);
        global_offset += 1;
      }
    }
  }
}

__global__ void newline_index(const char *file, int fileSize,
                              int *perTileNewlineOffsetIndex,
                              long *newlineIndex) {
  // Check out ./newline-index.cu.disabled for different optimization on newline
  // index kernels we have tried. Current one is the best one.

  newline_index_sharemem_transposed_packed(
      file, fileSize, perTileNewlineOffsetIndex, newlineIndex);
}

} // namespace gpjson::index::kernels::sharemem
