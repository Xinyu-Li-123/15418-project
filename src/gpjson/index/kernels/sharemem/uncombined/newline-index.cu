#include "gpjson/log/log.hpp"

namespace gpjson::index::kernels::sharemem {

__global__ void newline_index(const char *file, int fileSize,
                              int *perTileNewlineOffsetIndex,
                              long *newlineIndex) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = 32768; // 512 * 64

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  const int tid = threadIdx.x;
  const int tile_idx = blockIdx.x;
  const int block_start = tile_idx * CHUNK_SIZE;

  const int tile_offset = perTileNewlineOffsetIndex[tile_idx];

  // in-tile per-thread flag of is_newline
  __shared__ char in_tile_is_newline[CHUNK_SIZE];

  // Read file chunk into shared memory using coalesced global reads.
  constexpr int packed_bytes_gmem = 8;
  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);
  constexpr int packed_elems_per_block = CHUNK_SIZE / packed_bytes_gmem;

  for (int p = tid; p < packed_elems_per_block; p += blockDim.x) {
    const int global_byte_idx = block_start + p * packed_bytes_gmem;
    const int global_packed_byte_idx = global_byte_idx / packed_bytes_gmem;
    if (global_byte_idx >= fileSize) {
      break;
    }
    const uint2 packed_bytes = file_packed_gmem[global_packed_byte_idx];
    const char *packed_chars = reinterpret_cast<const char *>(&packed_bytes);
#pragma unroll
    for (int local_idx = 0; local_idx < packed_bytes_gmem; ++local_idx) {
      const int global_idx = global_byte_idx + local_idx;
      const char curChar = packed_chars[local_idx];
      const int in_tile_idx = p * packed_bytes_gmem + local_idx;
      if (global_idx < fileSize && curChar == '\n') {
        in_tile_is_newline[in_tile_idx] = 1;
      } else {
        in_tile_is_newline[in_tile_idx] = 0;
      }
    }
  }
  __syncthreads();

  constexpr int WARP_SIZE = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;
  const int local_start = tid * BYTES_PER_THREAD;

  int in_thread_count = 0;
#pragma unroll
  for (int i = 0; i < BYTES_PER_THREAD; ++i) {
    const int local_idx = local_start + i;
    const int global_idx = block_start + local_idx;
    if (global_idx < fileSize && in_tile_is_newline[local_idx]) {
      in_thread_count += 1;
    }
  }

  // prefix sum on in_tile_counts to get per-thread newline offset.

  // We first do an in-warp inclusive scan over per-thread newline count using
  // __shfl_up_sync(), then convert it to an exclusive scan by subtracting the
  // original per-thread count. Lane 31 of each warp writes the in-warp total to
  // shared memory, and one warp then does an exclusive scan over per-warp
  // totals to get in-block warp offsets.
  __shared__ int per_warp_totals[WARPS_PER_BLOCK];
  __shared__ int per_warp_offsets[WARPS_PER_BLOCK];

  int in_warp_inclusive_offset = in_thread_count;
#pragma unroll
  for (int pass_offset = 1; pass_offset < WARP_SIZE; pass_offset <<= 1) {
    int other_thread_count =
        __shfl_up_sync(0xffffffff, in_warp_inclusive_offset, pass_offset);
    if (lane_id >= pass_offset) {
      in_warp_inclusive_offset += other_thread_count;
    }
  }

  const int in_warp_exclusive_offset =
      in_warp_inclusive_offset - in_thread_count;
  const int in_warp_num_newlines =
      __shfl_sync(0xffffffff, in_warp_inclusive_offset, WARP_SIZE - 1);

  if (lane_id == 0) {
    per_warp_totals[warp_id] = in_warp_num_newlines;
  }
  __syncthreads();

  if (warp_id == 0) {
    int per_warp_inclusive_offset =
        lane_id < WARPS_PER_BLOCK ? per_warp_totals[lane_id] : 0;
#pragma unroll
    for (int pass_offset = 1; pass_offset < WARP_SIZE; pass_offset <<= 1) {
      int other_warp_count =
          __shfl_up_sync(0xffffffff, per_warp_inclusive_offset, pass_offset);
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

#pragma unroll
  for (int i = 0; i < BYTES_PER_THREAD; ++i) {
    const int local_idx = local_start + i;
    const int global_idx = block_start + local_idx;
    if (global_idx < fileSize && in_tile_is_newline[local_idx]) {
      newlineIndex[global_offset + 1] = global_idx;
      global_offset += 1;
    }
  }
}

} // namespace gpjson::index::kernels::sharemem
