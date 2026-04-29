#include "gpjson/log/log.hpp"

#include <cstddef>
#include <cstdint>

namespace gpjson::index::kernels::fuse {

namespace {

__device__ __forceinline__ uint64_t uint2_to_u64(uint2 x) {
  return (static_cast<uint64_t>(x.y) << 32) | static_cast<uint64_t>(x.x);
}

__device__ __forceinline__ void set_bit_at_offset(uint64_t &bitmap, int offset,
                                                  bool bit_value) {
  if (bit_value) {
    bitmap |= uint64_t{1} << offset;
  }
}

/**
 * Recompute the escape carry for the immediately previous 64-byte logical
 * chunk.
 *
 * This is only needed for tid == 0 in escape_carry_quote_carry_index.
 * We cannot safely read escapeCarryIndex[global_thread_id - 1] inside the same
 * kernel because CUDA does not guarantee inter-block execution order.
 *
 * Since BYTES_PER_THREAD == 64, the previous chunk alone is sufficient to
 * determine whether the first byte of this chunk is escaped.
 */
__device__ char previous_chunk_escape_carry(const char *file, size_t fileSize,
                                            size_t thread_global_base) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int PACK_BYTES = 8;

  if (thread_global_base == 0) {
    return 0;
  }

  const size_t prev_chunk_start = thread_global_base - BYTES_PER_THREAD;
  const size_t prev_chunk_end = thread_global_base;

  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  char carry = 0;

#pragma unroll
  for (int group = 0; group < BYTES_PER_THREAD / PACK_BYTES; ++group) {
    const size_t group_global_base =
        prev_chunk_start + static_cast<size_t>(group) * PACK_BYTES;

    uint2 packed_bytes = make_uint2(0u, 0u);

    if (group_global_base + PACK_BYTES <= fileSize) {
      packed_bytes = file_packed_gmem[group_global_base / PACK_BYTES];
    } else if (group_global_base < fileSize) {
      unsigned char *packed_chars =
          reinterpret_cast<unsigned char *>(&packed_bytes);

#pragma unroll
      for (int b = 0; b < PACK_BYTES; ++b) {
        const size_t global_idx = group_global_base + b;
        packed_chars[b] = (global_idx < fileSize)
                              ? static_cast<unsigned char>(file[global_idx])
                              : static_cast<unsigned char>(0);
      }
    }

    const uint64_t word = uint2_to_u64(packed_bytes);

#pragma unroll
    for (int b = 0; b < PACK_BYTES; ++b) {
      const size_t global_idx = group_global_base + b;

      if (global_idx >= fileSize || global_idx >= prev_chunk_end) {
        break;
      }

      const char cur_char = static_cast<char>((word >> (8 * b)) & 0xFFu);

      if (cur_char == '\\') {
        carry ^= 1;
      } else {
        carry = 0;
      }
    }
  }

  return carry;
}

} // namespace

/**
 * Fused escape-carry and quote-carry kernel.
 *
 * Per logical thread:
 *   - owns exactly 64 bytes
 *   - computes escapeCarryIndex[global_thread_id]
 *   - computes local quote parity quoteCarryIndex[global_thread_id]
 *
 * quoteCarryIndex is still only local parity here. It must be
 * XOR-prefix-scanned before being passed to
 * string_index_using_escape_carry_quote_carry_index_packed.
 */
__global__ void escape_carry_quote_carry_index(const char *file, int fileSize,
                                               char *escapeCarryIndex,
                                               char *quoteCarryIndex) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = THREADS_PER_BLOCK * BYTES_PER_THREAD;

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

  const size_t global_thread_id =
      static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

  const size_t block_start = static_cast<size_t>(blockIdx.x) * CHUNK_SIZE;

  const size_t thread_global_base =
      block_start + static_cast<size_t>(tid) * BYTES_PER_THREAD;

  /*
   * Transposed packed shared-memory layout:
   *
   *   smem_packed_bytes[group][tid]
   *
   * flattened as:
   *
   *   group * THREADS_PER_BLOCK + tid
   *
   * For a fixed group, warp lanes read contiguous uint2 values from shared
   * memory.
   */
  __shared__ uint2
      smem_packed_bytes[PACKED_GROUPS_PER_THREAD * THREADS_PER_BLOCK];

  __shared__ char tb_escape_carry[THREADS_PER_BLOCK];

  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  /*
   * Coalesced global load, then transposed shared-memory write.
   *
   * Linear packed element p corresponds to logical owner:
   *
   *   owner_tid = p / PACKED_GROUPS_PER_THREAD
   *   group     = p % PACKED_GROUPS_PER_THREAD
   */
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

    smem_packed_bytes[smem_idx] = packed_bytes;
  }

  __syncthreads();

  /*
   * First pass over the staged tile:
   * compute per-thread escape carry.
   */
  char local_escape_carry = 0;

  if (thread_global_base < fileSize) {
#pragma unroll
    for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
      const size_t group_global_base =
          thread_global_base + static_cast<size_t>(group) * PACK_BYTES;

      if (group_global_base >= fileSize) {
        break;
      }

      const int smem_idx = group * THREADS_PER_BLOCK + tid;
      const uint64_t word = uint2_to_u64(smem_packed_bytes[smem_idx]);

#pragma unroll
      for (int b = 0; b < PACK_BYTES; ++b) {
        const size_t global_idx = group_global_base + b;

        if (global_idx >= fileSize) {
          break;
        }

        const char cur_char = static_cast<char>((word >> (8 * b)) & 0xFFu);

        if (cur_char == '\\') {
          local_escape_carry ^= 1;
        } else {
          local_escape_carry = 0;
        }
      }
    }
  }

  tb_escape_carry[tid] = local_escape_carry;

  __syncthreads();

  /*
   * Second pass over the same staged tile:
   * compute quote parity using previous chunk's escape carry.
   */
  char quote_parity = 0;

  if (thread_global_base < fileSize) {
    char is_cur_escaped;

    if (global_thread_id == 0) {
      is_cur_escaped = 0;
    } else if (tid > 0) {
      is_cur_escaped = tb_escape_carry[tid - 1];
    } else {
      is_cur_escaped =
          previous_chunk_escape_carry(file, fileSize, thread_global_base);
    }

#pragma unroll
    for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
      const size_t group_global_base =
          thread_global_base + static_cast<size_t>(group) * PACK_BYTES;

      if (group_global_base >= fileSize) {
        break;
      }

      const int smem_idx = group * THREADS_PER_BLOCK + tid;
      const uint64_t word = uint2_to_u64(smem_packed_bytes[smem_idx]);

#pragma unroll
      for (int b = 0; b < PACK_BYTES; ++b) {
        const size_t global_idx = group_global_base + b;

        if (global_idx >= fileSize) {
          break;
        }

        if (is_cur_escaped) {
          is_cur_escaped = 0;
          continue;
        }

        const char cur_char = static_cast<char>((word >> (8 * b)) & 0xFFu);

        if (cur_char == '"') {
          quote_parity ^= 1;
          continue;
        }

        is_cur_escaped = static_cast<char>(cur_char == '\\');
      }
    }
  }

  escapeCarryIndex[global_thread_id] =
      thread_global_base < fileSize ? local_escape_carry : 0;

  quoteCarryIndex[global_thread_id] =
      thread_global_base < fileSize ? quote_parity : 0;
}

/**
 * Packed string-index kernel.
 *
 * Preconditions:
 *   - quoteCarryIndex has already been XOR-prefix-scanned.
 *   - escapeCarryIndex is the local per-64-byte escape carry from
 *     escape_carry_quote_carry_index.
 *
 * This kernel does not use shared memory. Each thread reads its own 64-byte
 * chunk using eight packed 8-byte loads and writes one 64-bit string bitmap.
 */
__global__ void string_index_using_escape_carry_quote_carry_index_packed(
    const char *file, int fileSize, const char *escapeCarryIndex,
    const char *quoteCarryIndex, long *stringIndex, size_t stringIndexSize) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;

  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int PACK_BYTES = 8;
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;

  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);
  static_assert(sizeof(uint2) == PACK_BYTES);

  const size_t word_idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (word_idx >= stringIndexSize) {
    return;
  }

  const size_t thread_global_base = word_idx * BYTES_PER_THREAD;

  if (thread_global_base >= fileSize) {
    stringIndex[word_idx] = 0;
    return;
  }

  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  bool is_cur_escaped =
      word_idx == 0 ? false : static_cast<bool>(escapeCarryIndex[word_idx - 1]);

  bool is_cur_quoted =
      word_idx == 0 ? false : static_cast<bool>(quoteCarryIndex[word_idx - 1]);

  uint64_t string_word = 0;

#pragma unroll
  for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
    const size_t group_global_base =
        thread_global_base + static_cast<size_t>(group) * PACK_BYTES;

    if (group_global_base >= fileSize) {
      break;
    }

    uint2 packed_bytes = make_uint2(0u, 0u);

    if (group_global_base + PACK_BYTES <= fileSize) {
      packed_bytes = file_packed_gmem[group_global_base / PACK_BYTES];
    } else {
      unsigned char *packed_chars =
          reinterpret_cast<unsigned char *>(&packed_bytes);

#pragma unroll
      for (int b = 0; b < PACK_BYTES; ++b) {
        const size_t global_idx = group_global_base + b;
        packed_chars[b] = (global_idx < fileSize)
                              ? static_cast<unsigned char>(file[global_idx])
                              : static_cast<unsigned char>(0);
      }
    }

    const uint64_t file_word = uint2_to_u64(packed_bytes);

#pragma unroll
    for (int b = 0; b < PACK_BYTES; ++b) {
      const int byte_offset = group * PACK_BYTES + b;
      const size_t global_idx = thread_global_base + byte_offset;

      if (global_idx >= fileSize) {
        break;
      }

      const char cur_char = static_cast<char>((file_word >> (8 * b)) & 0xFFu);

      if (is_cur_escaped) {
        set_bit_at_offset(string_word, byte_offset, is_cur_quoted);
        is_cur_escaped = false;
        continue;
      }

      if (cur_char == '"') {
        is_cur_quoted = !is_cur_quoted;
        set_bit_at_offset(string_word, byte_offset, is_cur_quoted);
        continue;
      }

      is_cur_escaped = cur_char == '\\';
      set_bit_at_offset(string_word, byte_offset, is_cur_quoted);
    }
  }

  stringIndex[word_idx] = static_cast<long>(string_word);
}

} // namespace gpjson::index::kernels::fuse
