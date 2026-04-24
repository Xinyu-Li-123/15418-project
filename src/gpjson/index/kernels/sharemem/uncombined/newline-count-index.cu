#include "gpjson/log/log.hpp"

namespace gpjson::index::kernels::sharemem {

/**
 * A tile-based newline count index
 *
 * File partition is divided into tiles. Each thread block handles a tile. A
 * thread block of T threads, each thread handle B bytes. Thus, a thread block
 * process a tile size of T * B bytes.
 *
 */
__global__ void newline_count_index(const char *file, int fileSize,
                                    int *newlineCountIndex) {
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

  int tid = threadIdx.x;
  int tile_idx = blockIdx.x;
  int block_start = tile_idx * CHUNK_SIZE;
  int count = 0;

  // each thread read BYTES_PER_THREAD bytes, count newline, and write
  // per-thread count to a shared memory array in_tile_counts.
  __shared__ int in_tile_counts[THREADS_PER_BLOCK];

  // Note that the BYTES_PER_THREAD bytes assigned to each thread is not
  // continguous bytes. This is to allow coalesced global read from file
  // partition.
  // TODO: Read packed bytes
  int in_thread_count = 0;
  for (int p = tid; p < CHUNK_SIZE; p += blockDim.x) {
    int global_byte_idx = block_start + p;
    if (global_byte_idx >= fileSize) {
      break;
    }
    if (file[global_byte_idx] == '\n') {
      in_thread_count += 1;
    }
  }
  in_tile_counts[tid] = in_thread_count;
  __syncthreads();

  // reduce on in_tile_counts to get per-tile newline count.

  // TODO: Optimize this with warp-level reduction (sum within warp w/
  // __shfl_down_sync w/o share mem access)
  for (int pass_bound = (THREADS_PER_BLOCK >> 1); pass_bound > 0;
       pass_bound >>= 1) {
    if (tid < pass_bound) {
      in_tile_counts[tid] += in_tile_counts[tid + pass_bound];
    }
    __syncthreads();
  }

  if (tid == 0) {
    newlineCountIndex[tile_idx] = in_tile_counts[0];
  }

  // if (tid == 0) {
  //   int expected_count = 0;
  //   for (int p = 0; p < CHUNK_SIZE; p++) {
  //     int global_byte_idx = block_start + p;
  //     if (global_byte_idx >= fileSize) {
  //       break;
  //     }
  //     if (file[global_byte_idx] == '\n') {
  //       expected_count += 1;
  //     }
  //   }
  //   Check(expected_count == newlineCountIndex[tile_idx],
  //         "Incorrect per-tile newline count. Expect %d, got %d.",
  //         expected_count, newlineCountIndex[tile_idx]);
  // }
}
} // namespace gpjson::index::kernels::sharemem
