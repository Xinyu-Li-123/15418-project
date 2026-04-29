#include "gpjson/log/log.hpp"

namespace gpjson::index::kernels::sharemem {

__device__ void newline_count_index_warp_reduce(const char *file, int fileSize,
                                                int *perTileNewlineCountIndex) {
  // REQUIRES: gridDim.x * blockDim.x * 64 >= file partition size
  // REQUIRES: length of newlineCountIndex == gridDim.x, i.e. we count the num
  // of newline for each tile
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = 32768; // 512 * 64
  constexpr int WARP_SIZE = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  int tid = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  int warp_id = tid / WARP_SIZE;
  int tile_id = blockIdx.x;
  int block_start = tile_id * CHUNK_SIZE;

  // each thread read BYTES_PER_THREAD bytes, count newline, sum over warp, and
  // write per-warp count to a shared memory array in_tile_counts.
  __shared__ int per_warp_counts[WARPS_PER_BLOCK];

  // Note that the BYTES_PER_THREAD bytes assigned to each thread is not
  // continguous bytes. This is to allow coalesced global read from file
  // partition.
  constexpr int packed_bytes_gmem = 8;
  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);
  constexpr int packed_elems_per_block = CHUNK_SIZE / packed_bytes_gmem;

  int in_thread_count = 0;
  // Each thread reads `packed_bytes_gmem` bytes at a time
  for (int p = tid; p < packed_elems_per_block; p += blockDim.x) {
    int global_byte_idx = block_start + p * packed_bytes_gmem;
    int global_packed_byte_idx = global_byte_idx / packed_bytes_gmem;
    if (global_byte_idx >= fileSize) {
      break;
    }
    uint2 packed_bytes = file_packed_gmem[global_packed_byte_idx];
    const char *packed_chars = reinterpret_cast<const char *>(&packed_bytes);
#pragma unroll
    for (int local_idx = 0; local_idx < packed_bytes_gmem; ++local_idx) {
      int global_idx = global_byte_idx + local_idx;
      char curChar = packed_chars[local_idx];
      if (global_idx < fileSize && curChar == '\n') {
        in_thread_count += 1;
      }
    }
  }

  // TODO: We first do a in-warp reduction to get newline count per warp, then
  // do block-wise reduction over warp-level count

  // When computing per-warp sum, we can use __shfl_sync() (and its variants) to
  // exchange variable within warp without going through shared memory
  for (int pass_round = (WARP_SIZE >> 1); pass_round > 0; pass_round >>= 1) {
    int other_thread_count =
        __shfl_down_sync(0xffffffff, in_thread_count, pass_round);
    if (lane_id < pass_round) {
      in_thread_count += other_thread_count;
    }
  }

  if (lane_id == 0) {
    per_warp_counts[warp_id] = in_thread_count;
  }
  __syncthreads();

  // use one warp to reduce on in_tile_counts to get per-tile newline count.

  if (warp_id == 0) {
    int per_warp_count =
        lane_id < WARPS_PER_BLOCK ? per_warp_counts[lane_id] : 0;
    for (int pass_bound = (WARPS_PER_BLOCK >> 1); pass_bound > 0;
         pass_bound >>= 1) {
      per_warp_count +=
          __shfl_down_sync(0xffffffff, per_warp_count, pass_bound);
    }
    if (lane_id == 0) {
      perTileNewlineCountIndex[tile_id] = per_warp_count;
    }
  }

#ifdef GPJSON_CPP_DEBUG
  if (tid == 0) {
    int expected_count = 0;
    for (int p = 0; p < CHUNK_SIZE; p++) {
      int global_byte_idx = block_start + p;
      if (global_byte_idx >= fileSize) {
        break;
      }
      if (file[global_byte_idx] == '\n') {
        expected_count += 1;
      }
    }
    Check(expected_count == perTileNewlineCountIndex[tile_id],
          "Incorrect per-tile newline count. Expect %d, got %d.",
          expected_count, perTileNewlineCountIndex[tile_id]);
  }
#endif
}

__device__ void
newline_count_index_packed_bytes(const char *file, int fileSize,
                                 int *perTileNewlineCountIndex) {
  // REQUIRES: gridDim.x * blockDim.x * 64 >= file partition size
  // REQUIRES: length of newlineCountIndex == gridDim.x, i.e. we count the num
  // of newline for each tile
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
  int count = 0;

  // each thread read BYTES_PER_THREAD bytes, count newline, and write
  // per-thread count to a shared memory array in_tile_counts.
  __shared__ int in_tile_counts[THREADS_PER_BLOCK];

  // Note that the BYTES_PER_THREAD bytes assigned to each thread is not
  // continguous bytes. This is to allow coalesced global read from file
  // partition.
  constexpr int packed_bytes_gmem = 8;
  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);
  constexpr int packed_elems_per_block = CHUNK_SIZE / packed_bytes_gmem;

  int in_thread_count = 0;
  // Each thread reads `packed_bytes_gmem` bytes at a time
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
      if (global_idx < fileSize && curChar == '\n') {
        in_thread_count += 1;
      }
    }
  }

  in_tile_counts[tid] = in_thread_count;
  __syncthreads();

  // reduce on in_tile_counts to get per-tile newline count.
  for (int pass_bound = (THREADS_PER_BLOCK >> 1); pass_bound > 0;
       pass_bound >>= 1) {
    if (tid < pass_bound) {
      in_tile_counts[tid] += in_tile_counts[tid + pass_bound];
    }
    __syncthreads();
  }

  if (tid == 0) {
    perTileNewlineCountIndex[tile_idx] = in_tile_counts[0];
  }

#ifdef GPJSON_CPP_DEBUG
  if (tid == 0) {
    int expected_count = 0;
    for (int p = 0; p < CHUNK_SIZE; p++) {
      int global_byte_idx = block_start + p;
      if (global_byte_idx >= fileSize) {
        break;
      }
      if (file[global_byte_idx] == '\n') {
        expected_count += 1;
      }
    }
    Check(expected_count == perTileNewlineCountIndex[tile_idx],
          "Incorrect per-tile newline count. Expect %d, got %d.",
          expected_count, perTileNewlineCountIndex[tile_idx]);
  }
#endif
}

/**
 * A tile-based newline count index
 *
 * File partition is divided into tiles. Each thread block handles a tile. A
 * thread block of T threads, each thread handle B bytes. Thus, a thread block
 * process a tile size of T * B bytes.
 *
 */
__global__ void newline_count_index(const char *file, int fileSize,
                                    int *perTileNewlineCountIndex) {
  bool use_warp_reduce = true;
  if (use_warp_reduce) {
    newline_count_index_warp_reduce(file, fileSize, perTileNewlineCountIndex);
  } else {
    newline_count_index_packed_bytes(file, fileSize, perTileNewlineCountIndex);
  }
}

} // namespace gpjson::index::kernels::sharemem
