#include "gpjson/index/kernels/sharemem.cuh"
#include "gpjson/log/log.hpp"
#include "gpjson/profiler/profiler.hpp"

#include <cassert>
#include <cstdio>
namespace gpjson::index::kernels::sharemem {

/*
 * A thread block will process a file chunk of size and range
 *
 *    size : B = blockDim.x * 64 bytes
 *    range: file[bid * B: (bid + 1) * B]
 *
 * where each thread will process 64 bytes
 *
 */
__global__ void escape_carry_index(const char *file, int fileSize,
                                   char *escapeCarryIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int charsPerThread = (fileSize + stride - 1) / stride;
  int bitmapAlignedCharsPerThread = ((charsPerThread + 32 - 1) / 32) * 32;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("index = %d, stride=%d", index, stride);
    printf("blockDim.x = %d, gridDim.x = %d\n", blockDim.x, gridDim.x);
    printf("charsPerThread: %d\n", charsPerThread);
    printf("bitmapAlignedCharsPerThread: %d\n", bitmapAlignedCharsPerThread);
  }
  int start = index * bitmapAlignedCharsPerThread;
  if (start >= fileSize) {
    return;
  }
  // int end = start + bitmapAlignedCharsPerThread;

  __shared__ char file_subpartition[BYTES_PER_THREAD_BLOCK];
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    Check(bitmapAlignedCharsPerThread == BYTES_PER_THREAD,
          "We require each thread to process %d bytes", BYTES_PER_THREAD);
    Check(bitmapAlignedCharsPerThread * blockDim.x == BYTES_PER_THREAD_BLOCK,
          "We require each thread block to process %d bytes",
          BYTES_PER_THREAD_BLOCK);
  }
  // Read the file chunk one block at a time
  for (int i = 0; i < BYTES_PER_THREAD; i += 1) {
    int subpart_idx = threadIdx.x + i * blockDim.x;
    int block_start = blockIdx.x * BYTES_PER_THREAD_BLOCK;
    file_subpartition[subpart_idx] = file[block_start + subpart_idx];
  }
  __syncthreads();

  char carry = 0;

  for (int i = threadIdx.x * BYTES_PER_THREAD;
       i < (threadIdx.x + 1) * BYTES_PER_THREAD; i += 1) {
    char curChar = file_subpartition[i];
    // int fake_i = i;
    // int fake_i = start;
    // int fake_i = start + (i - start) % 2;
    // char curChar = file[fake_i];
    if (curChar == '\\') {
      carry = 1 ^ carry;
    } else {
      carry = 0;
    }
  }

  escapeCarryIndex[index] = carry;
}

} // namespace gpjson::index::kernels::sharemem
