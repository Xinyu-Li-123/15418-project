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

  __shared__ alignas(16) char file_chunk[CHUNK_SIZE];

  int tid = threadIdx.x;
  int block_start = blockIdx.x * CHUNK_SIZE;

  // Coalesced global load into shared memory, 16 bytes at a time to make

  char *file_chunk_bytes = file_chunk;
  // NOTE: uint4 is a cuda data type of 4 unsigned int (16 bytes), while
  // uint64_t is a 64-bit / 8 bytes unsigned integer. The naming is a bit
  // confusing. We use uint64_t instead of uint2 (also 8 bytes) later because
  // uint2 can't do bitwise opeartions like right shift

  // TODO: Why uint4 is slower than uint2?

  // constexpr int packed_bytes_gmem = 16;
  // static_assert(CHUNK_SIZE % packed_bytes_gmem == 0);
  // uint4 *file_chunk_packed_gmem = reinterpret_cast<uint4
  // *>(file_chunk_bytes); const uint4 *file_packed_gmem =
  // reinterpret_cast<const uint4 *>(file);

  constexpr int packed_bytes_gmem = 8;
  static_assert(CHUNK_SIZE % packed_bytes_gmem == 0);
  uint2 *file_chunk_packed_gmem = reinterpret_cast<uint2 *>(file_chunk_bytes);
  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  constexpr int packed_elems_per_block = CHUNK_SIZE / packed_bytes_gmem;

  for (int p = tid; p < packed_elems_per_block; p += blockDim.x) {
    int global_byte_idx = block_start + p * packed_bytes_gmem;

    if (global_byte_idx + packed_bytes_gmem <= fileSize) {
      file_chunk_packed_gmem[p] =
          file_packed_gmem[global_byte_idx / packed_bytes_gmem];
    } else {
      // Tail handling for the last partial packed load in the file.
      char *dst = &file_chunk_bytes[p * packed_bytes_gmem];

#pragma unroll
      for (int b = 0; b < packed_bytes_gmem; ++b) {
        int idx = global_byte_idx + b;
        dst[b] = (idx < fileSize) ? file[idx] : 0;
      }
    }
  }

  __syncthreads();

  // file chunk load in packed byte from shared memory
  const uint64_t *file_chunk_packed_smem =
      reinterpret_cast<const uint64_t *>(file_chunk);

  const int packed_bytes_smem = 8;

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
    // const uint64_t *file_chunk_packed =
    //     reinterpret_cast<const uint64_t *>(file_chunk);
    //
    // const int packed_bytes = 8;

    int word_start = local_start / packed_bytes_smem;
    int word_end = local_end / packed_bytes_smem;

    for (int w = word_start;
         w < word_end && block_start + w * packed_bytes_smem < fileSize; ++w) {
      uint64_t word = file_chunk_packed_smem[w];

#pragma unroll
      for (int b = 0; b < packed_bytes_smem; ++b) {
        int global_idx = block_start + w * packed_bytes_smem + b;
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
    // const uint64_t *file_chunk_packed =
    //     reinterpret_cast<const uint64_t *>(file_chunk);
    //
    // const int packed_bytes = 8;

    int word_start = local_start / packed_bytes_smem;
    int word_end = local_end / packed_bytes_smem;

    bool done = false;

    for (int w = word_end - 1; w >= word_start && !done; --w) {
      uint64_t word = file_chunk_packed_smem[w];

#pragma unroll
      for (int b = packed_bytes_smem - 1; b >= 0; --b) {
        int local_byte = (w - word_start) * packed_bytes_smem + b;
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
