#include "gpjson/log/log.hpp"

namespace gpjson::index::kernels::sharemem {

__global__ void newline_index(const char *file, int fileSize,
                              int *perTileNewlineCountIndex,
                              long *newlineIndex) {
  // int index = blockIdx.x * blockDim.x + threadIdx.x;
  // int stride = blockDim.x * gridDim.x;
  // int offset = perTileNewlineCountIndex[index];
  //
  // long charsPerThread = (fileSize + stride - 1) / stride;
  // long start = index * charsPerThread;
  // long end = start + charsPerThread;
  //
  // for (int i = start; i < end && i < fileSize; i += 1) {
  //   if (file[i] == '\n') {
  //     newlineIndex[offset++] = i;
  //   }
  // }

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
}

} // namespace gpjson::index::kernels::sharemem
