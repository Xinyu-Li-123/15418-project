#include "gpjson/index/kernels/sharemem.cuh"
#include "gpjson/log/log.hpp"
#include "gpjson/profiler/profiler.hpp"

#include <cassert>
#include <cstdio>
namespace gpjson::index::kernels::sharemem {

// __global__ void escape_carry_index(const char *file, int fileSize,
//                                    char *escapeCarryIndex) {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;
//
//   int charsPerThread = (fileSize + stride - 1) / stride;
//   int bitmapAlignedCharsPerThread = ((charsPerThread + 64 - 1) / 64) * 64;
//   int start = index * bitmapAlignedCharsPerThread;
//   int end = start + bitmapAlignedCharsPerThread;
//
//   char carry = 0;
//
//   for (int i = start; i < end && i < fileSize; i += 1) {
//     if (file[i] == '\\') {
//       carry = 1 ^ carry;
//     } else {
//       carry = 0;
//     }
//   }
//
//   escapeCarryIndex[index] = carry;
// }

// __global__ void escape_carry_index(const char *file, int fileSize,
//                                    char *escapeCarryIndex) {
//   constexpr int BYTES_PER_THREAD = 64;
//   constexpr int CHUNK_SIZE =
//       BYTES_PER_THREAD * 512; // assuming blockDim.x == 512
//
//   __shared__ char file_chunk[CHUNK_SIZE];
//
//   int tid = threadIdx.x;
//   int block_start = blockIdx.x * CHUNK_SIZE;
//
//   // coalesced load into shared
//   for (int j = tid; j < CHUNK_SIZE; j += blockDim.x) {
//     int global_idx = block_start + j;
//     file_chunk[j] = (global_idx < fileSize) ? file[global_idx] : 0;
//   }
//   __syncthreads();
//
//   int local_start = tid * BYTES_PER_THREAD;
//   int local_end = local_start + BYTES_PER_THREAD;
//
//   char carry = 0;
//
//   for (int j = local_start; j < local_end && block_start + j < fileSize; ++j)
//   {
//     char cur_char = file_chunk[j];
//     if (cur_char == '\\') {
//       carry ^= 1;
//     } else {
//       carry = 0;
//     }
//   }
//
//   int index = blockIdx.x * blockDim.x + tid;
//   if (block_start + local_start < fileSize) {
//     escapeCarryIndex[index] = carry;
//   }
// }

__global__ void escape_carry_index(const char *file, int fileSize,
                                   char *escapeCarryIndex) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int CHUNK_SIZE = 32768; // 512 * 64

  __shared__ char file_chunk[CHUNK_SIZE];

  int tid = threadIdx.x;
  int block_start = blockIdx.x * CHUNK_SIZE;

  // Coalesced global load into shared memory
  for (int j = tid; j < CHUNK_SIZE; j += blockDim.x) {
    int global_idx = block_start + j;
    file_chunk[j] = (global_idx < fileSize) ? file[global_idx] : 0;
  }
  __syncthreads();

  int local_start = tid * BYTES_PER_THREAD;
  int local_end = local_start + BYTES_PER_THREAD;

  char carry = 0;

  // const uint32_t *file_chunk_packed =
  //     reinterpret_cast<const uint32_t *>(file_chunk);
  // // shared memory bank size
  // const int smem_bs = 4;

  const uint64_t *file_chunk_packed =
      reinterpret_cast<const uint64_t *>(file_chunk);

  const int packed_bytes = 8;

  int word_start = local_start / packed_bytes;
  int word_end = local_end / packed_bytes;

  for (int w = word_start;
       w < word_end && block_start + w * packed_bytes < fileSize; ++w) {
    uint64_t word = file_chunk_packed[w];

#pragma unroll
    for (int b = 0; b < packed_bytes; ++b) {
      int global_idx = block_start + w * packed_bytes + b;
      if (global_idx >= fileSize) {
        break;
      }

      char cur_char = static_cast<char>((word >> (8 * b)) & 0xFF);

      if (cur_char == '\\') {
        carry ^= 1;
      } else {
        carry = 0;
      }
    }
  }

  int index = blockIdx.x * blockDim.x + tid;
  if (block_start + local_start < fileSize) {
    escapeCarryIndex[index] = carry;
  }
}

} // namespace gpjson::index::kernels::sharemem
