#include "gpjson/index/kernels/sharemem.cuh"
#include "gpjson/log/log.hpp"
#include "gpjson/profiler/profiler.hpp"

#include <cassert>
#include <cstddef>
#include <cstdio>
namespace gpjson::index::kernels::sharemem {

__device__ void escape_carry_index_packed_forward_read(const char *file,
                                                       size_t fileSize,
                                                       char *escapeCarryIndex) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;

  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int PACK_BYTES = 8;
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;

  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);
  static_assert(sizeof(uint2) == PACK_BYTES);

  const int tid = threadIdx.x;

  const size_t global_thread_id =
      static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

  const size_t thread_global_base = global_thread_id * BYTES_PER_THREAD;

  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  char carry = 0;

#pragma unroll
  for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
    const size_t group_global_base =
        thread_global_base + static_cast<size_t>(group) * PACK_BYTES;

    if (group_global_base >= fileSize) {
      break;
    }

    uint2 packed_bytes_word = make_uint2(0u, 0u);

    if (group_global_base + PACK_BYTES <= fileSize) {
      packed_bytes_word = file_packed_gmem[group_global_base / PACK_BYTES];
    } else {
      unsigned char *packed_chars =
          reinterpret_cast<unsigned char *>(&packed_bytes_word);

#pragma unroll
      for (int b = 0; b < PACK_BYTES; ++b) {
        const size_t global_idx = group_global_base + b;
        packed_chars[b] = (global_idx < fileSize)
                              ? static_cast<unsigned char>(file[global_idx])
                              : static_cast<unsigned char>(0);
      }
    }

    const uint64_t word = (static_cast<uint64_t>(packed_bytes_word.y) << 32) |
                          static_cast<uint64_t>(packed_bytes_word.x);

#pragma unroll
    for (int b = 0; b < PACK_BYTES; ++b) {
      const size_t global_idx = group_global_base + b;

      if (global_idx >= fileSize) {
        break;
      }

      const char cur_char = static_cast<char>((word >> (8 * b)) & 0xFF);

      if (cur_char == '\\') {
        carry ^= 1;
      } else {
        carry = 0;
      }
    }
  }

  const int index = blockIdx.x * blockDim.x + tid;

  if (thread_global_base < fileSize) {
    escapeCarryIndex[index] = carry;
  }
}

__device__ void
escape_carry_index_packed_backward_read(const char *file, size_t fileSize,
                                        char *escapeCarryIndex) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;

  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int PACK_BYTES = 8;
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;

  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);
  static_assert(sizeof(uint2) == PACK_BYTES);

  const int tid = threadIdx.x;

  const size_t global_thread_id =
      static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

  const size_t thread_global_base = global_thread_id * BYTES_PER_THREAD;

  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  char carry = 0;
  bool done = false;

#pragma unroll
  for (int group = PACKED_GROUPS_PER_THREAD - 1; group >= 0; --group) {
    if (done) {
      break;
    }

    const size_t group_global_base =
        thread_global_base + static_cast<size_t>(group) * PACK_BYTES;

    // Last partial tile: higher groups of some threads may be past EOF.
    if (group_global_base >= fileSize) {
      continue;
    }

    uint2 packed_bytes_word = make_uint2(0u, 0u);

    if (group_global_base + PACK_BYTES <= fileSize) {
      packed_bytes_word = file_packed_gmem[group_global_base / PACK_BYTES];
    } else {
      unsigned char *packed_chars =
          reinterpret_cast<unsigned char *>(&packed_bytes_word);

#pragma unroll
      for (int b = 0; b < PACK_BYTES; ++b) {
        const size_t global_idx = group_global_base + b;
        packed_chars[b] = (global_idx < fileSize)
                              ? static_cast<unsigned char>(file[global_idx])
                              : static_cast<unsigned char>(0);
      }
    }

    const uint64_t word = (static_cast<uint64_t>(packed_bytes_word.y) << 32) |
                          static_cast<uint64_t>(packed_bytes_word.x);

#pragma unroll
    for (int b = PACK_BYTES - 1; b >= 0; --b) {
      const size_t global_idx = group_global_base + b;

      if (global_idx >= fileSize) {
        continue;
      }

      const char cur_char = static_cast<char>((word >> (8 * b)) & 0xFF);

      if (cur_char == '\\') {
        carry ^= 1;
      } else {
        done = true;
        break;
      }
    }
  }

  const int index = blockIdx.x * blockDim.x + tid;

  if (thread_global_base < fileSize) {
    escapeCarryIndex[index] = carry;
  }
}

__device__ void escape_carry_index_sharemem_packed_forward_read(
    const char *file, size_t fileSize, char *escapeCarryIndex) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int CHUNK_SIZE = 32768; // 512 * 64

  __shared__ alignas(16) char file_chunk[CHUNK_SIZE];

  int tid = threadIdx.x;
  size_t block_start = static_cast<size_t>(blockIdx.x) * CHUNK_SIZE;

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
    size_t global_byte_idx =
        block_start + static_cast<size_t>(p) * packed_bytes_gmem;

    if (global_byte_idx + packed_bytes_gmem <= fileSize) {
      file_chunk_packed_gmem[p] =
          file_packed_gmem[global_byte_idx / packed_bytes_gmem];
    } else {
      // Tail handling for the last partial packed load in the file.
      char *dst = &file_chunk_bytes[p * packed_bytes_gmem];

#pragma unroll
      for (int b = 0; b < packed_bytes_gmem; ++b) {
        size_t idx = global_byte_idx + b;
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

  char carry = 0;
  int word_start = local_start / packed_bytes_smem;
  int word_end = local_end / packed_bytes_smem;

  for (int w = word_start;
       w < word_end &&
       block_start + static_cast<size_t>(w) * packed_bytes_smem < fileSize;
       ++w) {
    uint64_t word = file_chunk_packed_smem[w];

#pragma unroll
    for (int b = 0; b < packed_bytes_smem; ++b) {
      size_t global_idx =
          block_start + static_cast<size_t>(w) * packed_bytes_smem + b;
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
  // coalesced global write
  if (block_start + local_start < fileSize) {
    escapeCarryIndex[index] = carry;
  }
}

__device__ void escape_carry_index_sharemem_packed_backward_read(
    const char *file, size_t fileSize, char *escapeCarryIndex) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int CHUNK_SIZE = 32768; // 512 * 64

  __shared__ alignas(16) char file_chunk[CHUNK_SIZE];

  int tid = threadIdx.x;
  size_t block_start = static_cast<size_t>(blockIdx.x) * CHUNK_SIZE;

  // Coalesced global load into shared memory, 16 bytes at a time to make

  char *file_chunk_bytes = file_chunk;
  // NOTE: uint4 is a cuda data type of 4 unsigned int (16 bytes), while
  // uint64_t is a 64-bit / 8 bytes unsigned integer. The naming is a bit
  // confusing. We use uint64_t instead of uint2 (also 8 bytes) later because
  // uint2 can't do bitwise opeartions like right shift

  // TODO: Why uint4 is slower than uint2?

  constexpr int packed_bytes_gmem = 8;
  static_assert(CHUNK_SIZE % packed_bytes_gmem == 0);
  uint2 *file_chunk_packed_gmem = reinterpret_cast<uint2 *>(file_chunk_bytes);
  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  constexpr int packed_elems_per_block = CHUNK_SIZE / packed_bytes_gmem;

  for (int p = tid; p < packed_elems_per_block; p += blockDim.x) {
    size_t global_byte_idx =
        block_start + static_cast<size_t>(p) * packed_bytes_gmem;

    if (global_byte_idx + packed_bytes_gmem <= fileSize) {
      file_chunk_packed_gmem[p] =
          file_packed_gmem[global_byte_idx / packed_bytes_gmem];
    } else {
      // Tail handling for the last partial packed load in the file.
      char *dst = &file_chunk_bytes[p * packed_bytes_gmem];

#pragma unroll
      for (int b = 0; b < packed_bytes_gmem; ++b) {
        size_t idx = global_byte_idx + b;
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

  char carry = 0;

  int word_start = local_start / packed_bytes_smem;
  int word_end = local_end / packed_bytes_smem;

  bool done = false;

  for (int w = word_end - 1; w >= word_start && !done; --w) {
    uint64_t word = file_chunk_packed_smem[w];

#pragma unroll
    for (int b = packed_bytes_smem - 1; b >= 0; --b) {
      int local_byte = (w - word_start) * packed_bytes_smem + b;
      size_t global_idx = block_start + local_start + local_byte;
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

  int index = blockIdx.x * blockDim.x + tid;
  // coalesced global write
  if (block_start + local_start < fileSize) {
    escapeCarryIndex[index] = carry;
  }
}

__device__ void escape_carry_index_sharemem_transposed_packed_forward_read(
    const char *file, size_t fileSize, char *escapeCarryIndex) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = 32768; // 512 * 64

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int PACK_BYTES = 8;
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;
  constexpr int PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / PACK_BYTES;

  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % PACK_BYTES == 0);
  static_assert(sizeof(uint2) == PACK_BYTES);

  const int tid = threadIdx.x;
  const size_t block_start = static_cast<size_t>(blockIdx.x) * CHUNK_SIZE;

  // Transposed packed layout:
  //
  //   file_chunk_packed[group][tid]
  //
  // flattened as:
  //
  //   file_chunk_packed[group * THREADS_PER_BLOCK + tid]
  //
  // For a fixed group, warp lanes read contiguous uint2 values from shared
  // memory.
  __shared__ uint2
      file_chunk_packed[PACKED_GROUPS_PER_THREAD * THREADS_PER_BLOCK];

  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  // Coalesced global load, then transposed shared-memory write.
  //
  // Linear packed element p corresponds to global bytes:
  //
  //   [block_start + p * 8, block_start + p * 8 + 7]
  //
  // Original logical ownership:
  //
  //   owner_tid = p / 8
  //   group     = p % 8
  //
  // Transposed shared-memory destination:
  //
  //   file_chunk_packed[group][owner_tid]
  for (int p = tid; p < PACKED_ELEMS_PER_BLOCK; p += blockDim.x) {
    const size_t global_byte_idx =
        block_start + static_cast<size_t>(p) * PACK_BYTES;

    uint2 packed_bytes = make_uint2(0u, 0u);

    if (global_byte_idx + PACK_BYTES <= fileSize) {
      packed_bytes = file_packed_gmem[global_byte_idx / PACK_BYTES];
    } else if (global_byte_idx < fileSize) {
      unsigned char *packed_chars =
          reinterpret_cast<unsigned char *>(&packed_bytes);

#pragma unroll
      for (int b = 0; b < PACK_BYTES; ++b) {
        const size_t global_idx = global_byte_idx + b;
        packed_chars[b] = (global_idx < fileSize)
                              ? static_cast<unsigned char>(file[global_idx])
                              : static_cast<unsigned char>(0);
      }
    }

    const int owner_tid = p / PACKED_GROUPS_PER_THREAD;
    const int group = p % PACKED_GROUPS_PER_THREAD;

    const int smem_idx = group * THREADS_PER_BLOCK + owner_tid;
    file_chunk_packed[smem_idx] = packed_bytes;
  }

  __syncthreads();

  char carry = 0;

  const size_t thread_global_base =
      block_start + static_cast<size_t>(tid) * BYTES_PER_THREAD;

#pragma unroll
  for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
    const size_t group_global_base =
        thread_global_base + static_cast<size_t>(group) * PACK_BYTES;

    if (group_global_base >= fileSize) {
      break;
    }

    const int smem_idx = group * THREADS_PER_BLOCK + tid;
    const uint2 packed_bytes_word = file_chunk_packed[smem_idx];

    // Use uint64_t for cheap byte extraction via shifts.
    const uint64_t word = (static_cast<uint64_t>(packed_bytes_word.y) << 32) |
                          static_cast<uint64_t>(packed_bytes_word.x);

#pragma unroll
    for (int b = 0; b < PACK_BYTES; ++b) {
      const size_t global_idx = group_global_base + b;

      if (global_idx >= fileSize) {
        break;
      }

      const char cur_char = static_cast<char>((word >> (8 * b)) & 0xFF);

      if (cur_char == '\\') {
        carry ^= 1;
      } else {
        carry = 0;
      }
    }
  }

  const int index = blockIdx.x * blockDim.x + tid;

  // Coalesced global write.
  if (thread_global_base < fileSize) {
    escapeCarryIndex[index] = carry;
  }
}

__device__ void escape_carry_index_sharemem_transposed_packed_backward_read(
    const char *file, size_t fileSize, char *escapeCarryIndex) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = 32768; // 512 * 64

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int PACK_BYTES = 8;
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;
  constexpr int PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / PACK_BYTES;

  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % PACK_BYTES == 0);
  static_assert(sizeof(uint2) == PACK_BYTES);

  const int tid = threadIdx.x;
  const size_t block_start = static_cast<size_t>(blockIdx.x) * CHUNK_SIZE;

  // Transposed packed layout:
  //
  //   file_chunk_packed[group][tid]
  //
  // flattened as:
  //
  //   file_chunk_packed[group * THREADS_PER_BLOCK + tid]
  //
  // For a fixed group, warp lanes read contiguous uint2 values from shared
  // memory. The backward scan simply visits group in reverse order.
  __shared__ uint2
      file_chunk_packed[PACKED_GROUPS_PER_THREAD * THREADS_PER_BLOCK];

  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  // Coalesced global load, then transposed shared-memory write.
  //
  // Linear packed element p corresponds to global bytes:
  //
  //   [block_start + p * 8, block_start + p * 8 + 7]
  //
  // Original logical ownership:
  //
  //   owner_tid = p / 8
  //   group     = p % 8
  //
  // Transposed shared-memory destination:
  //
  //   file_chunk_packed[group][owner_tid]
  for (int p = tid; p < PACKED_ELEMS_PER_BLOCK; p += blockDim.x) {
    const size_t global_byte_idx =
        block_start + static_cast<size_t>(p) * PACK_BYTES;

    uint2 packed_bytes = make_uint2(0u, 0u);

    if (global_byte_idx + PACK_BYTES <= fileSize) {
      packed_bytes = file_packed_gmem[global_byte_idx / PACK_BYTES];
    } else if (global_byte_idx < fileSize) {
      unsigned char *packed_chars =
          reinterpret_cast<unsigned char *>(&packed_bytes);

#pragma unroll
      for (int b = 0; b < PACK_BYTES; ++b) {
        const size_t global_idx = global_byte_idx + b;
        packed_chars[b] = (global_idx < fileSize)
                              ? static_cast<unsigned char>(file[global_idx])
                              : static_cast<unsigned char>(0);
      }
    }

    const int owner_tid = p / PACKED_GROUPS_PER_THREAD;
    const int group = p % PACKED_GROUPS_PER_THREAD;

    const int smem_idx = group * THREADS_PER_BLOCK + owner_tid;
    file_chunk_packed[smem_idx] = packed_bytes;
  }

  __syncthreads();

  const int local_start = tid * BYTES_PER_THREAD;
  const size_t thread_global_base = block_start + local_start;

  char carry = 0;
  bool done = false;

#pragma unroll
  for (int group = PACKED_GROUPS_PER_THREAD - 1; group >= 0; --group) {
    if (done) {
      break;
    }

    const size_t group_global_base =
        thread_global_base + static_cast<size_t>(group) * PACK_BYTES;

    // If this whole packed group starts past EOF, skip it. This matters for
    // the last partial tile: the highest groups of some threads may be padding.
    if (group_global_base >= fileSize) {
      continue;
    }

    const int smem_idx = group * THREADS_PER_BLOCK + tid;
    const uint2 packed_bytes_word = file_chunk_packed[smem_idx];

    // Use uint64_t for cheap byte extraction via shifts.
    const uint64_t word = (static_cast<uint64_t>(packed_bytes_word.y) << 32) |
                          static_cast<uint64_t>(packed_bytes_word.x);

#pragma unroll
    for (int b = PACK_BYTES - 1; b >= 0; --b) {
      const size_t global_idx = group_global_base + b;

      if (global_idx >= fileSize) {
        continue;
      }

      const char cur_char = static_cast<char>((word >> (8 * b)) & 0xFF);

      if (cur_char == '\\') {
        carry ^= 1;
      } else {
        done = true;
        break;
      }
    }
  }

  const int index = blockIdx.x * blockDim.x + tid;

  // Coalesced global write.
  if (thread_global_base < fileSize) {
    escapeCarryIndex[index] = carry;
  }
}

__global__ void escape_carry_index(const char *file, size_t fileSize,
                                   char *escapeCarryIndex) {

  // escape_carry_index_packed_forward_read(file, fileSize, escapeCarryIndex);

  // While we theoretically do less work with backward read, it makes the kernel
  // more branchy
  escape_carry_index_packed_backward_read(file, fileSize, escapeCarryIndex);

  // escape_carry_index_packed_backward_read(file, fileSize,
  //                                                  escapeCarryIndex);

  // escape_carry_index_sharemem_packed_forward_read(file, fileSize,
  //                                                 escapeCarryIndex);

  // escape_carry_index_sharemem_packed_backward_read(file, fileSize,
  //                                                  escapeCarryIndex);

  // escape_carry_index_sharemem_transposed_packed_forward_read(file, fileSize,
  //                                                            escapeCarryIndex);

  // NOTE: While both transposed version suffers from uncoalesced smem write due
  // to transpose, this version optimzies for the common case: when we read from
  // smem to compute carry, we commonly only read the last few bytes, thust most
  // of the time reading one packed bytes will do the job. This outweigh the
  // overhead from coalesced smem write.
  // escape_carry_index_sharemem_transposed_packed_backward_read(file, fileSize,
  //                                                             escapeCarryIndex);
}
} // namespace gpjson::index::kernels::sharemem
