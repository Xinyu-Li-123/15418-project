#include "gpjson/index/kernels/sharemem.cuh"
#include "gpjson/log/log.hpp"
#include "gpjson/profiler/profiler.hpp"

#include <cassert>
#include <cstdio>
namespace gpjson::index::kernels::sharemem {

__global__ void escape_carry_index(const char *file, int fileSize,
                                   char *escapeCarryIndex) {
  Check(blockDim.x == 512, "block size must be 512!");

  constexpr int BYTES_PER_THREAD = 64;
  constexpr int CHUNK_SIZE =
      BYTES_PER_THREAD * 512; // assuming blockDim.x == 512

  __shared__ char file_chunk[CHUNK_SIZE];

  int tid = threadIdx.x;
  int block_start = blockIdx.x * CHUNK_SIZE;

  // coalesced load into shared
  for (int j = tid; j < CHUNK_SIZE; j += blockDim.x) {
    int global_idx = block_start + j;
    file_chunk[j] = (global_idx < fileSize) ? file[global_idx] : 0;
  }
  __syncthreads();

  int local_start = tid * BYTES_PER_THREAD;
  int local_end = local_start + BYTES_PER_THREAD;

  char carry = 0;

  for (int j = local_start; j < local_end && block_start + j < fileSize; ++j) {
    char cur_char = file_chunk[j];
    if (cur_char == '\\') {
      carry ^= 1;
    } else {
      carry = 0;
    }
  }

  int index = blockIdx.x * blockDim.x + tid;
  if (block_start + local_start < fileSize) {
    escapeCarryIndex[index] = carry;
  }
}

} // namespace gpjson::index::kernels::sharemem
