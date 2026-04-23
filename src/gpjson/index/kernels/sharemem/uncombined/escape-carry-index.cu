#include "gpjson/index/kernels/sharemem.cuh"
#include "gpjson/log/log.hpp"
#include "gpjson/profiler/profiler.hpp"

#include <cassert>
#include <cstdio>
namespace gpjson::index::kernels::sharemem {

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

  // Since most of the time block don't carry, and if carry, most of the time
  // only a few backslashes at the chunk end, we read from chunk end backward
  // to chunk start, and imemdaitely decide carry flag value if encounter
  // non-consecutive backslash

  // const bool READ_BACKWARD = false;
  const bool READ_BACKWARD = true;

  char carry = 0;
  if (!READ_BACKWARD) {

    // const uint32_t *file_chunk_packed =
    //     reinterpret_cast<const uint32_t *>(file_chunk);
    // // shared memory bank size
    // const int smem_bs = 4;

    // To load from shared memory, we need to avoid bank conflict
    // 2080 Ti has a bank size of 4 bytes

    // Load multiple bytes at a time from shared memory, both to avoid bank
    // conflict and to
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

  } else {
    const uint64_t *file_chunk_packed =
        reinterpret_cast<const uint64_t *>(file_chunk);

    const int packed_bytes = 8;

    int word_start = local_start / packed_bytes;
    int word_end = local_end / packed_bytes;

    char carry = 0;
    bool done = false;

    for (int w = word_end - 1; w >= word_start && !done; --w) {
      uint64_t word = file_chunk_packed[w];

#pragma unroll
      for (int b = packed_bytes - 1; b >= 0; --b) {
        int local_byte = (w - word_start) * packed_bytes + b;
        int global_idx = block_start + local_start + local_byte;
        if (global_idx >= fileSize) {
          continue;
        }

        char cur_char = static_cast<char>((word >> (8 * b)) & 0xFF);

        if (cur_char == '\\') {
          carry ^= 1;
        } else {
          done = true;
          break;
        }
      }
    }
  }

  int index = blockIdx.x * blockDim.x + tid;
  // coalesced global write
  if (block_start + local_start < fileSize) {
    escapeCarryIndex[index] = carry;
  }
}

} // namespace gpjson::index::kernels::sharemem
