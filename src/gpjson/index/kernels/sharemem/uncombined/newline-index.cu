#include "gpjson/log/log.hpp"

namespace gpjson::index::kernels::sharemem {

__device__ void newline_index_sharemem_naive(const char *file, int fileSize,
                                             int *perTileNewlineOffsetIndex,
                                             long *newlineIndex) {
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

  const int tile_offset = perTileNewlineOffsetIndex[tile_idx];

  // in-tile per-thread flag of is_newline
  __shared__ char in_tile_is_newline[CHUNK_SIZE];

  // Read file chunk into shared memory using coalesced global reads.
  constexpr int packed_bytes_gmem = 8;
  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);
  constexpr int packed_elems_per_block = CHUNK_SIZE / packed_bytes_gmem;

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
      const int in_tile_idx = p * packed_bytes_gmem + local_idx;
      if (global_idx < fileSize && curChar == '\n') {
        in_tile_is_newline[in_tile_idx] = 1;
      } else {
        in_tile_is_newline[in_tile_idx] = 0;
      }
    }
  }
  __syncthreads();

  constexpr int WARP_SIZE = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;
  const int local_start = tid * BYTES_PER_THREAD;

  int in_thread_count = 0;
#pragma unroll
  for (int i = 0; i < BYTES_PER_THREAD; ++i) {
    const int local_idx = local_start + i;
    const int global_idx = block_start + local_idx;
    if (global_idx < fileSize && in_tile_is_newline[local_idx]) {
      in_thread_count += 1;
    }
  }

  // prefix sum on in_tile_counts to get per-thread newline offset.

  // We first do an in-warp inclusive scan over per-thread newline count using
  // __shfl_up_sync(), then convert it to an exclusive scan by subtracting the
  // original per-thread count. Lane 31 of each warp writes the in-warp total to
  // shared memory, and one warp then does an exclusive scan over per-warp
  // totals to get in-block warp offsets.
  __shared__ int per_warp_totals[WARPS_PER_BLOCK];
  __shared__ int per_warp_offsets[WARPS_PER_BLOCK];

  int in_warp_inclusive_offset = in_thread_count;
#pragma unroll
  for (int pass_offset = 1; pass_offset < WARP_SIZE; pass_offset <<= 1) {
    int other_thread_count =
        __shfl_up_sync(0xffffffffu, in_warp_inclusive_offset, pass_offset);
    if (lane_id >= pass_offset) {
      in_warp_inclusive_offset += other_thread_count;
    }
  }

  const int in_warp_exclusive_offset =
      in_warp_inclusive_offset - in_thread_count;
  const int in_warp_num_newlines =
      __shfl_sync(0xffffffffu, in_warp_inclusive_offset, WARP_SIZE - 1);

  if (lane_id == 0) {
    per_warp_totals[warp_id] = in_warp_num_newlines;
  }
  __syncthreads();

  if (warp_id == 0) {
    int per_warp_inclusive_offset =
        lane_id < WARPS_PER_BLOCK ? per_warp_totals[lane_id] : 0;
#pragma unroll
    for (int pass_offset = 1; pass_offset < WARP_SIZE; pass_offset <<= 1) {
      int other_warp_count =
          __shfl_up_sync(0xffffffffu, per_warp_inclusive_offset, pass_offset);
      if (lane_id >= pass_offset) {
        per_warp_inclusive_offset += other_warp_count;
      }
    }

    if (lane_id < WARPS_PER_BLOCK) {
      per_warp_offsets[lane_id] =
          per_warp_inclusive_offset - per_warp_totals[lane_id];
    }
  }
  __syncthreads();

  int global_offset =
      tile_offset + per_warp_offsets[warp_id] + in_warp_exclusive_offset;

#pragma unroll
  for (int i = 0; i < BYTES_PER_THREAD; ++i) {
    const int local_idx = local_start + i;
    const int global_idx = block_start + local_idx;
    if (global_idx < fileSize && in_tile_is_newline[local_idx]) {
      newlineIndex[global_offset + 1] = global_idx;
      global_offset += 1;
    }
  }
}

__device__ void newline_index_sharemem(const char *file, int fileSize,
                                       int *perTileNewlineOffsetIndex,
                                       long *newlineIndex) {
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
  const int tile_offset = perTileNewlineOffsetIndex[tile_idx];

  constexpr int WARP_SIZE = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  // Store one 0/1 byte per input byte in the tile.
  // Align to 8 bytes so we can safely read it back as uint2-packed values.
  __shared__ unsigned char in_tile_is_newline[CHUNK_SIZE];

  // Coalesced global-memory load: 8 bytes at a time.
  constexpr int PACKED_BYTES_GMEM = 8;
  static_assert(CHUNK_SIZE % PACKED_BYTES_GMEM == 0);
  static_assert(BYTES_PER_THREAD % PACKED_BYTES_GMEM == 0);

  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);
  constexpr int packed_elems_per_block = CHUNK_SIZE / PACKED_BYTES_GMEM;

  for (int p = tid; p < packed_elems_per_block; p += blockDim.x) {
    const int global_byte_idx = block_start + p * PACKED_BYTES_GMEM;
    const int global_packed_idx = global_byte_idx / PACKED_BYTES_GMEM;

    uint2 packed_bytes = make_uint2(0u, 0u);
    if (global_byte_idx < fileSize) {
      packed_bytes = file_packed_gmem[global_packed_idx];
    }

    const char *packed_chars = reinterpret_cast<const char *>(&packed_bytes);
#pragma unroll
    for (int local_idx = 0; local_idx < PACKED_BYTES_GMEM; ++local_idx) {
      const int global_idx = global_byte_idx + local_idx;
      const int in_tile_idx = p * PACKED_BYTES_GMEM + local_idx;
      in_tile_is_newline[in_tile_idx] =
          (global_idx < fileSize && packed_chars[local_idx] == '\n') ? 1 : 0;
    }
  }
  __syncthreads();

  // Read shared-memory flags back 8 bytes at a time per thread.
  constexpr int PACKED_BYTES_SMEM = 8;
  constexpr int packed_elems_per_thread = BYTES_PER_THREAD / PACKED_BYTES_SMEM;
  const uint2 *packed_in_tile_is_newline =
      reinterpret_cast<const uint2 *>(in_tile_is_newline);

  const int local_packed_start = tid * packed_elems_per_thread;

  int in_thread_count = 0;
#pragma unroll
  for (int p = 0; p < packed_elems_per_thread; ++p) {
    const uint2 packed_flags_word =
        packed_in_tile_is_newline[local_packed_start + p];
    const unsigned char *packed_flags =
        reinterpret_cast<const unsigned char *>(&packed_flags_word);
#pragma unroll
    for (int i = 0; i < PACKED_BYTES_SMEM; ++i) {
      in_thread_count += static_cast<int>(packed_flags[i]);
    }
  }

  // In-warp exclusive scan over per-thread newline counts.
  __shared__ int per_warp_totals[WARPS_PER_BLOCK];
  __shared__ int per_warp_offsets[WARPS_PER_BLOCK];

  int in_warp_inclusive_offset = in_thread_count;
#pragma unroll
  for (int pass_offset = 1; pass_offset < WARP_SIZE; pass_offset <<= 1) {
    const int other_thread_count =
        __shfl_up_sync(0xffffffffu, in_warp_inclusive_offset, pass_offset);
    if (lane_id >= pass_offset) {
      in_warp_inclusive_offset += other_thread_count;
    }
  }

  const int in_warp_exclusive_offset =
      in_warp_inclusive_offset - in_thread_count;
  const int in_warp_num_newlines =
      __shfl_sync(0xffffffffu, in_warp_inclusive_offset, WARP_SIZE - 1);

  if (lane_id == WARP_SIZE - 1) {
    per_warp_totals[warp_id] = in_warp_num_newlines;
  }
  __syncthreads();

  // Warp 0 scans per-warp totals to get per-warp offsets within the block.
  if (warp_id == 0) {
    int per_warp_inclusive_offset =
        (lane_id < WARPS_PER_BLOCK) ? per_warp_totals[lane_id] : 0;
#pragma unroll
    for (int pass_offset = 1; pass_offset < WARP_SIZE; pass_offset <<= 1) {
      const int other_warp_count =
          __shfl_up_sync(0xffffffffu, per_warp_inclusive_offset, pass_offset);
      if (lane_id >= pass_offset) {
        per_warp_inclusive_offset += other_warp_count;
      }
    }

    if (lane_id < WARPS_PER_BLOCK) {
      per_warp_offsets[lane_id] =
          per_warp_inclusive_offset - per_warp_totals[lane_id];
    }
  }
  __syncthreads();

  int global_offset =
      tile_offset + per_warp_offsets[warp_id] + in_warp_exclusive_offset;

  // Scatter newline positions. Keep GPJSON's leading sentinel at index 0,
  // so physical newline positions start at newlineIndex[1].
#pragma unroll
  for (int p = 0; p < packed_elems_per_thread; ++p) {
    const int local_packed_idx = local_packed_start + p;
    const int global_base_idx =
        block_start + local_packed_idx * PACKED_BYTES_SMEM;

    const uint2 packed_flags_word = packed_in_tile_is_newline[local_packed_idx];
    const unsigned char *packed_flags =
        reinterpret_cast<const unsigned char *>(&packed_flags_word);

#pragma unroll
    for (int i = 0; i < PACKED_BYTES_SMEM; ++i) {
      if (packed_flags[i]) {
        newlineIndex[global_offset + 1] =
            static_cast<long>(global_base_idx + i);
        global_offset += 1;
      }
    }
  }
}

/**
 * We want strided read from shared memory to reduce bank conflict. To do this,
 * we write to shared memory in a transposed way. Given file_smem, the array of
 * file bytes stored in shared dmemory, we transpose our write pattern from
 *
 *   file_smem[tid][byte_idx] => file_smem[byte_idx][tid]
 *
 * where
 * - tid=[0, 512)
 * - byte_idx=[0, 64)
 *
 * This way, each thread can read 64 bytes from shared memory with a stride of
 * 512 bytes, instead of sequentially read a continguous chunk of 64 bytes.
 *
 * This will reduce bank conflict in the two passes of read, at the cost of more
 * bank conflict in the one pass of write. This is acceptable since we read
 * twice as much we write, and that workload in read passes are heavier then
 * that in write pass, thus stall due to bank conflict are more severe.
 */
__device__ void
newline_index_sharemem_transposed(const char *file, int fileSize,
                                  int *perTileNewlineOffsetIndex,
                                  long *newlineIndex) {
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
  const int tile_offset = perTileNewlineOffsetIndex[tile_idx];

  constexpr int WARP_SIZE = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  // Transposed layout:
  //   smem_flag[byte_in_thread][thread]
  // flattened as:
  //   in_tile_is_newline[byte_in_thread * THREADS_PER_BLOCK + owner_tid]
  __shared__ unsigned char in_tile_is_newline[CHUNK_SIZE];

  // Coalesced global-memory load: 8 bytes at a time across the block.
  constexpr int PACKED_BYTES_GMEM = 8;
  static_assert(CHUNK_SIZE % PACKED_BYTES_GMEM == 0);
  static_assert(BYTES_PER_THREAD % PACKED_BYTES_GMEM == 0);

  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);
  constexpr int packed_elems_per_block = CHUNK_SIZE / PACKED_BYTES_GMEM;

  for (int p = tid; p < packed_elems_per_block; p += blockDim.x) {
    const int global_byte_idx = block_start + p * PACKED_BYTES_GMEM;
    const int global_packed_idx = global_byte_idx / PACKED_BYTES_GMEM;

    uint2 packed_bytes = make_uint2(0u, 0u);
    if (global_byte_idx < fileSize) {
      packed_bytes = file_packed_gmem[global_packed_idx];
    }

    const char *packed_chars = reinterpret_cast<const char *>(&packed_bytes);

#pragma unroll
    for (int local_idx = 0; local_idx < PACKED_BYTES_GMEM; ++local_idx) {
      const int in_tile_idx = p * PACKED_BYTES_GMEM + local_idx;
      const int global_idx = global_byte_idx + local_idx;

      // Map flat tile byte index -> logical owner thread and byte-in-thread.
      const int owner_tid = in_tile_idx / BYTES_PER_THREAD;
      const int byte_in_thread = in_tile_idx % BYTES_PER_THREAD;

      // Transposed shared-memory index.
      const int smem_idx = byte_in_thread * THREADS_PER_BLOCK + owner_tid;

      in_tile_is_newline[smem_idx] =
          (global_idx < fileSize && packed_chars[local_idx] == '\n') ? 1 : 0;
    }
  }
  __syncthreads();

  // Count this thread's 64 logical bytes from transposed shared memory.
  int in_thread_count = 0;
#pragma unroll
  for (int i = 0; i < BYTES_PER_THREAD; ++i) {
    const int smem_idx = i * THREADS_PER_BLOCK + tid;
    in_thread_count += static_cast<int>(in_tile_is_newline[smem_idx]);
  }

  // In-warp exclusive scan over per-thread newline counts.
  __shared__ int per_warp_totals[WARPS_PER_BLOCK];
  __shared__ int per_warp_offsets[WARPS_PER_BLOCK];

  int in_warp_inclusive_offset = in_thread_count;
#pragma unroll
  for (int pass_offset = 1; pass_offset < WARP_SIZE; pass_offset <<= 1) {
    const int other_thread_count =
        __shfl_up_sync(0xffffffffu, in_warp_inclusive_offset, pass_offset);
    if (lane_id >= pass_offset) {
      in_warp_inclusive_offset += other_thread_count;
    }
  }

  const int in_warp_exclusive_offset =
      in_warp_inclusive_offset - in_thread_count;
  const int in_warp_num_newlines =
      __shfl_sync(0xffffffffu, in_warp_inclusive_offset, WARP_SIZE - 1);

  if (lane_id == WARP_SIZE - 1) {
    per_warp_totals[warp_id] = in_warp_num_newlines;
  }
  __syncthreads();

  // Warp 0 scans per-warp totals to get per-warp offsets within the block.
  if (warp_id == 0) {
    int per_warp_inclusive_offset =
        (lane_id < WARPS_PER_BLOCK) ? per_warp_totals[lane_id] : 0;

#pragma unroll
    for (int pass_offset = 1; pass_offset < WARP_SIZE; pass_offset <<= 1) {
      const int other_warp_count =
          __shfl_up_sync(0xffffffffu, per_warp_inclusive_offset, pass_offset);
      if (lane_id >= pass_offset) {
        per_warp_inclusive_offset += other_warp_count;
      }
    }

    if (lane_id < WARPS_PER_BLOCK) {
      per_warp_offsets[lane_id] =
          per_warp_inclusive_offset - per_warp_totals[lane_id];
    }
  }
  __syncthreads();

  int global_offset =
      tile_offset + per_warp_offsets[warp_id] + in_warp_exclusive_offset;

  // Scatter newline positions in increasing order within this thread's
  // contiguous 64-byte segment.
  const int thread_global_base = block_start + tid * BYTES_PER_THREAD;

#pragma unroll
  for (int i = 0; i < BYTES_PER_THREAD; ++i) {
    const int global_idx = thread_global_base + i;
    const int smem_idx = i * THREADS_PER_BLOCK + tid;

    if (global_idx < fileSize && in_tile_is_newline[smem_idx]) {
      // Keep GPJSON's leading sentinel at index 0.
      newlineIndex[global_offset + 1] = static_cast<long>(global_idx);
      global_offset += 1;
    }
  }
}

/**
 * On top of strided read provided in newline_index_sharemem_transposed, we also
 * want to read multiple bytes at a time, i.e. we want packed bytes. To achieve
 * this, we change our write pattern
 *
 *   file_smem[tid][packed_byte_idx][packed_byte_offset] =>
 * file_smem[packed_byte_idx][tid][packed_byte_offset]
 *
 * where
 * - tid=[0, 512)
 * - packed_byte_idx=[0, 8), the idx of uint2
 * - packed_byte_offset=[0, 8), the offset into uint2 as an array of 8 chars
 *
 * This way, each thread can read 64 bytes from shared memory with a stride of
 * 512 bytes, and read 8 bytes at a time, instead of sequentially read a
 * continguous chunk of 64 bytes.
 */
__device__ void
newline_index_sharemem_transposed_packed(const char *file, int fileSize,
                                         int *perTileNewlineOffsetIndex,
                                         long *newlineIndex) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = 32768; // 512 * 64

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int WARP_SIZE = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  constexpr int PACK_BYTES = 8;
  constexpr int PACKED_GROUPS_PER_THREAD = BYTES_PER_THREAD / PACK_BYTES;
  constexpr int PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / PACK_BYTES;

  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % PACK_BYTES == 0);

  const int tid = threadIdx.x;
  const int tile_idx = blockIdx.x;
  const int block_start = tile_idx * CHUNK_SIZE;
  const int tile_offset = perTileNewlineOffsetIndex[tile_idx];

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  // Transposed packed layout:
  //   smem_flags[group][tid]
  // where each entry is one 8-byte packed flag word (0/1 bytes).
  __shared__ uint2
      smem_packed_flags[PACKED_GROUPS_PER_THREAD * THREADS_PER_BLOCK];

  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  // Coalesced global load, then transpose+pack into shared memory.
  for (int p = tid; p < PACKED_ELEMS_PER_BLOCK; p += blockDim.x) {
    const int global_byte_idx = block_start + p * PACK_BYTES;
    const int global_packed_idx = global_byte_idx / PACK_BYTES;

    uint2 packed_bytes = make_uint2(0u, 0u);
    if (global_byte_idx < fileSize) {
      packed_bytes = file_packed_gmem[global_packed_idx];
    }

    const char *packed_chars = reinterpret_cast<const char *>(&packed_bytes);

    unsigned char flag_bytes[PACK_BYTES];
#pragma unroll
    for (int i = 0; i < PACK_BYTES; ++i) {
      const int global_idx = global_byte_idx + i;
      flag_bytes[i] =
          (global_idx < fileSize && packed_chars[i] == '\n') ? 1 : 0;
    }

    uint2 packed_flags_word;
    unsigned char *packed_flag_chars =
        reinterpret_cast<unsigned char *>(&packed_flags_word);
#pragma unroll
    for (int i = 0; i < PACK_BYTES; ++i) {
      packed_flag_chars[i] = flag_bytes[i];
    }

    // Original logical ownership is contiguous 64-byte chunks per thread.
    // Map packed tile index p -> (owner thread, packed group within thread).
    const int owner_tid = p / PACKED_GROUPS_PER_THREAD;
    const int packed_group = p % PACKED_GROUPS_PER_THREAD;

    // Transposed packed shared-memory index: [packed_group][owner_tid]
    const int smem_idx = packed_group * THREADS_PER_BLOCK + owner_tid;
    smem_packed_flags[smem_idx] = packed_flags_word;
  }
  __syncthreads();

  int in_thread_count = 0;

#pragma unroll
  for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
    const int smem_idx = group * THREADS_PER_BLOCK + tid;
    const uint2 packed_flags_word = smem_packed_flags[smem_idx];
    const unsigned char *packed_flags =
        reinterpret_cast<const unsigned char *>(&packed_flags_word);

#pragma unroll
    for (int i = 0; i < PACK_BYTES; ++i) {
      in_thread_count += static_cast<int>(packed_flags[i]);
    }
  }

  __shared__ int per_warp_totals[WARPS_PER_BLOCK];
  __shared__ int per_warp_offsets[WARPS_PER_BLOCK];

  int in_warp_inclusive_offset = in_thread_count;
#pragma unroll
  for (int pass_offset = 1; pass_offset < WARP_SIZE; pass_offset <<= 1) {
    const int other_thread_count =
        __shfl_up_sync(0xffffffffu, in_warp_inclusive_offset, pass_offset);
    if (lane_id >= pass_offset) {
      in_warp_inclusive_offset += other_thread_count;
    }
  }

  const int in_warp_exclusive_offset =
      in_warp_inclusive_offset - in_thread_count;
  const int in_warp_num_newlines =
      __shfl_sync(0xffffffffu, in_warp_inclusive_offset, WARP_SIZE - 1);

  if (lane_id == WARP_SIZE - 1) {
    per_warp_totals[warp_id] = in_warp_num_newlines;
  }
  __syncthreads();

  if (warp_id == 0) {
    int per_warp_inclusive_offset =
        (lane_id < WARPS_PER_BLOCK) ? per_warp_totals[lane_id] : 0;

#pragma unroll
    for (int pass_offset = 1; pass_offset < WARP_SIZE; pass_offset <<= 1) {
      const int other_warp_count =
          __shfl_up_sync(0xffffffffu, per_warp_inclusive_offset, pass_offset);
      if (lane_id >= pass_offset) {
        per_warp_inclusive_offset += other_warp_count;
      }
    }

    if (lane_id < WARPS_PER_BLOCK) {
      per_warp_offsets[lane_id] =
          per_warp_inclusive_offset - per_warp_totals[lane_id];
    }
  }
  __syncthreads();

  int global_offset =
      tile_offset + per_warp_offsets[warp_id] + in_warp_exclusive_offset;

  const int thread_global_base = block_start + tid * BYTES_PER_THREAD;

#pragma unroll
  for (int group = 0; group < PACKED_GROUPS_PER_THREAD; ++group) {
    const int smem_idx = group * THREADS_PER_BLOCK + tid;
    const uint2 packed_flags_word = smem_packed_flags[smem_idx];
    const unsigned char *packed_flags =
        reinterpret_cast<const unsigned char *>(&packed_flags_word);

    const int group_global_base = thread_global_base + group * PACK_BYTES;

#pragma unroll
    for (int i = 0; i < PACK_BYTES; ++i) {
      const int global_idx = group_global_base + i;
      if (global_idx < fileSize && packed_flags[i]) {
        // Keep GPJSON's leading sentinel at index 0.
        newlineIndex[global_offset + 1] = static_cast<long>(global_idx);
        global_offset += 1;
      }
    }
  }
}

/**
 * This kernel utilize warp-level primitives of __ballot_sync and __popc to
 * efficiently compute newline index without storing file bytes into shared
 * memory
 *
 * We use __ballot_sync to check if cur byte is newline in a warp, and use
 * __popc to compute total number of newline in a warp, and use __popc with a
 * mask to compute rank (local index) of newline of a lane to order the newline
 * index write within each warp.
 */
__device__ void newline_index_sharemem_ballot(const char *file, int fileSize,
                                              int *perTileNewlineOffsetIndex,
                                              long *newlineIndex) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = 32768; // 512 * 64

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int WARP_SIZE = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
  constexpr int BYTES_PER_WARP = BYTES_PER_THREAD * WARP_SIZE; // 2048
  constexpr int ROUNDS_PER_WARP = BYTES_PER_WARP / WARP_SIZE;  // 64
  constexpr unsigned FULL_MASK = 0xffffffffu;

  static_assert(THREADS_PER_BLOCK % WARP_SIZE == 0);
  static_assert(BYTES_PER_WARP * WARPS_PER_BLOCK == CHUNK_SIZE);
  static_assert(ROUNDS_PER_WARP == BYTES_PER_THREAD);

  const int tid = threadIdx.x;
  const int tile_idx = blockIdx.x;
  const int block_start = tile_idx * CHUNK_SIZE;
  const int tile_offset = perTileNewlineOffsetIndex[tile_idx];

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  // printf("Shared memory usage: \n");
  // printf("- warp_round_bitmaps: %d bytes\n",
  //        WARPS_PER_BLOCK * ROUNDS_PER_WARP * sizeof(unsigned int));
  // printf("- per_warp_totals: %d bytes\n", WARPS_PER_BLOCK * sizeof(int));
  // printf("- per_warp_offsets: %d bytes\n", WARPS_PER_BLOCK * sizeof(int));

  // One 32-bit bitmap per warp per round. Bit l is set iff lane l saw '\n'
  // in that round.
  // warp_round_bitmaps[warp_id][round] stores the is_newline bitmap of each
  // warp in each round
  __shared__ unsigned int warp_round_bitmaps[WARPS_PER_BLOCK * ROUNDS_PER_WARP];
  __shared__ int per_warp_totals[WARPS_PER_BLOCK];
  __shared__ int per_warp_offsets[WARPS_PER_BLOCK];

  const int warp_global_base = block_start + warp_id * BYTES_PER_WARP;

  // First pass:
  // - each warp scans its 2048-byte region in 64 rounds
  // - stores one bitmap per round
  // - accumulates one total newline count per warp
  int warp_total = 0;

  for (int round = 0; round < ROUNDS_PER_WARP; ++round) {
    const int global_idx = warp_global_base + round * WARP_SIZE + lane_id;
    const bool is_newline = (global_idx < fileSize && file[global_idx] == '\n');

    const unsigned int bitmap = __ballot_sync(FULL_MASK, is_newline);

    if (lane_id == 0) {
      warp_round_bitmaps[warp_id * ROUNDS_PER_WARP + round] = bitmap;
      warp_total += __popc(bitmap);
    }
  }

  if (lane_id == 0) {
    per_warp_totals[warp_id] = warp_total;
  }
  __syncthreads();

  // Warp 0 scans per-warp totals to get per-warp offsets within the block.
  if (warp_id == 0) {
    int per_warp_inclusive_offset =
        (lane_id < WARPS_PER_BLOCK) ? per_warp_totals[lane_id] : 0;

#pragma unroll
    for (int pass_offset = 1; pass_offset < WARP_SIZE; pass_offset <<= 1) {
      const int other_warp_count =
          __shfl_up_sync(FULL_MASK, per_warp_inclusive_offset, pass_offset);
      if (lane_id >= pass_offset) {
        per_warp_inclusive_offset += other_warp_count;
      }
    }

    if (lane_id < WARPS_PER_BLOCK) {
      per_warp_offsets[lane_id] =
          per_warp_inclusive_offset - per_warp_totals[lane_id];
    }
  }
  __syncthreads();

  // Second pass:
  // Replay the stored bitmaps. Each warp writes into its own disjoint segment:
  //   tile_offset + per_warp_offsets[warp_id] + [0, per_warp_totals[warp_id))
  int warp_running_offset = 0;

  for (int round = 0; round < ROUNDS_PER_WARP; ++round) {
    const unsigned int bitmap =
        warp_round_bitmaps[warp_id * ROUNDS_PER_WARP + round];
    const int round_count = __popc(bitmap);

    const bool is_newline = ((bitmap >> lane_id) & 1u) != 0u;
    if (is_newline) {
      const unsigned int lower_lane_mask =
          (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
      const int rank_in_round = __popc(bitmap & lower_lane_mask);

      const int global_idx = warp_global_base + round * WARP_SIZE + lane_id;
      newlineIndex[tile_offset + per_warp_offsets[warp_id] +
                   warp_running_offset + rank_in_round + 1] =
          static_cast<long>(global_idx);
    }

    warp_running_offset += round_count;
  }
}

/**
 * Unpacked read from global memory, packed read from shared memory. Compared to
 * newline_index_sharemem_transposed_packed, speed up from 8.15 ms to 6.53 ms
 */
__device__ void
newline_index_sharemem_ballot_packed_smem(const char *file, int fileSize,
                                          int *perTileNewlineOffsetIndex,
                                          long *newlineIndex) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = 32768; // 512 * 64

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int WARP_SIZE = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  constexpr int PACK_BYTES = 8;
  constexpr int BYTES_PER_WARP = BYTES_PER_THREAD * WARP_SIZE;         // 2048
  constexpr int SUPER_ROUNDS_PER_WARP = BYTES_PER_THREAD / PACK_BYTES; // 8

  static_assert(THREADS_PER_BLOCK % WARP_SIZE == 0);
  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);
  static_assert(BYTES_PER_WARP * WARPS_PER_BLOCK == CHUNK_SIZE);

  const int tid = threadIdx.x;
  const int tile_idx = blockIdx.x;
  const int block_start = tile_idx * CHUNK_SIZE;
  const int tile_offset = perTileNewlineOffsetIndex[tile_idx];

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  const int warp_global_base = block_start + warp_id * BYTES_PER_WARP;

  // Packed shared-memory layout in ballot order:
  //   smem_packed_bytes[super_round][tid]
  // Each uint2 stores 8 actual file bytes for one lane across 8 consecutive
  // ordinary rounds:
  //   byte i => ordinary_round = super_round * PACK_BYTES + i
  __shared__ uint2 smem_packed_bytes[SUPER_ROUNDS_PER_WARP * THREADS_PER_BLOCK];

  // One bitmap per (warp, super_round, byte_in_pack).
  __shared__ unsigned int
      warp_bitmaps[WARPS_PER_BLOCK * SUPER_ROUNDS_PER_WARP * PACK_BYTES];

  __shared__ int per_warp_totals[WARPS_PER_BLOCK];
  __shared__ int per_warp_offsets[WARPS_PER_BLOCK];

  // --------------------------------------------------------------------------
  // Stage file bytes into shared memory in packed ballot order.
  // --------------------------------------------------------------------------
#pragma unroll
  for (int super_round = 0; super_round < SUPER_ROUNDS_PER_WARP;
       ++super_round) {
    uint2 packed_word = make_uint2(0u, 0u);
    unsigned char *packed_chars =
        reinterpret_cast<unsigned char *>(&packed_word);

#pragma unroll
    for (int byte_in_pack = 0; byte_in_pack < PACK_BYTES; ++byte_in_pack) {
      const int ordinary_round = super_round * PACK_BYTES + byte_in_pack;
      const int global_idx =
          warp_global_base + ordinary_round * WARP_SIZE + lane_id;

      packed_chars[byte_in_pack] =
          (global_idx < fileSize) ? static_cast<unsigned char>(file[global_idx])
                                  : static_cast<unsigned char>(0);
    }

    const int smem_idx = super_round * THREADS_PER_BLOCK + tid;
    smem_packed_bytes[smem_idx] = packed_word;
  }
  __syncthreads();

  // --------------------------------------------------------------------------
  // First pass:
  // - read packed bytes from shared memory
  // - generate 8 ballots per super-round
  // - store bitmaps
  // - accumulate one total per warp
  // --------------------------------------------------------------------------
  constexpr unsigned FULL_MASK = 0xffffffffu;

  int warp_total = 0;

#pragma unroll
  for (int super_round = 0; super_round < SUPER_ROUNDS_PER_WARP;
       ++super_round) {
    const int smem_idx = super_round * THREADS_PER_BLOCK + tid;
    const uint2 packed_word = smem_packed_bytes[smem_idx];
    const unsigned char *packed_chars =
        reinterpret_cast<const unsigned char *>(&packed_word);

#pragma unroll
    for (int byte_in_pack = 0; byte_in_pack < PACK_BYTES; ++byte_in_pack) {
      const int ordinary_round = super_round * PACK_BYTES + byte_in_pack;
      const int global_idx =
          warp_global_base + ordinary_round * WARP_SIZE + lane_id;

      const bool is_newline =
          (global_idx < fileSize && packed_chars[byte_in_pack] == '\n');

      const unsigned int bitmap = __ballot_sync(FULL_MASK, is_newline);

      if (lane_id == 0) {
        const int bitmap_idx =
            ((warp_id * SUPER_ROUNDS_PER_WARP + super_round) * PACK_BYTES) +
            byte_in_pack;
        warp_bitmaps[bitmap_idx] = bitmap;
        warp_total += __popc(bitmap);
      }
    }
  }

  if (lane_id == 0) {
    per_warp_totals[warp_id] = warp_total;
  }
  __syncthreads();

  // --------------------------------------------------------------------------
  // Warp 0 scans per-warp totals to get per-warp offsets within the block.
  // --------------------------------------------------------------------------
  if (warp_id == 0) {
    int per_warp_inclusive_offset =
        (lane_id < WARPS_PER_BLOCK) ? per_warp_totals[lane_id] : 0;

#pragma unroll
    for (int pass_offset = 1; pass_offset < WARP_SIZE; pass_offset <<= 1) {
      const int other_warp_count =
          __shfl_up_sync(FULL_MASK, per_warp_inclusive_offset, pass_offset);
      if (lane_id >= pass_offset) {
        per_warp_inclusive_offset += other_warp_count;
      }
    }

    if (lane_id < WARPS_PER_BLOCK) {
      per_warp_offsets[lane_id] =
          per_warp_inclusive_offset - per_warp_totals[lane_id];
    }
  }
  __syncthreads();

  // --------------------------------------------------------------------------
  // Second pass:
  // Replay the stored bitmaps in increasing byte order within the warp's
  // 2048-byte region:
  //   super_round 0, byte 0..7, then super_round 1, ...
  // --------------------------------------------------------------------------
  int warp_running_offset = 0;

#pragma unroll
  for (int super_round = 0; super_round < SUPER_ROUNDS_PER_WARP;
       ++super_round) {
#pragma unroll
    for (int byte_in_pack = 0; byte_in_pack < PACK_BYTES; ++byte_in_pack) {
      const int bitmap_idx =
          ((warp_id * SUPER_ROUNDS_PER_WARP + super_round) * PACK_BYTES) +
          byte_in_pack;
      const unsigned int bitmap = warp_bitmaps[bitmap_idx];
      const int round_count = __popc(bitmap);

      const bool is_newline = ((bitmap >> lane_id) & 1u) != 0u;
      if (is_newline) {
        const unsigned int lower_lane_mask =
            (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
        const int rank_in_round = __popc(bitmap & lower_lane_mask);

        const int ordinary_round = super_round * PACK_BYTES + byte_in_pack;
        const int global_idx =
            warp_global_base + ordinary_round * WARP_SIZE + lane_id;

        // Keep GPJSON's leading sentinel at index 0.
        newlineIndex[tile_offset + per_warp_offsets[warp_id] +
                     warp_running_offset + rank_in_round + 1] =
            static_cast<long>(global_idx);
      }

      warp_running_offset += round_count;
    }
  }
}

__device__ void
newline_index_sharemem_ballot_packed_gmem_smem(const char *file, int fileSize,
                                               int *perTileNewlineOffsetIndex,
                                               long *newlineIndex) {
  constexpr int BYTES_PER_THREAD = 64;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int CHUNK_SIZE = 32768; // 512 * 64

  Check(CHUNK_SIZE == BYTES_PER_THREAD * THREADS_PER_BLOCK,
        "Invalid choice of kernel config.");
  Check(blockDim.x == THREADS_PER_BLOCK, "We require %d threads per block.",
        THREADS_PER_BLOCK);

  constexpr int WARP_SIZE = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  constexpr int PACK_BYTES = 8;
  constexpr int BYTES_PER_WARP = BYTES_PER_THREAD * WARP_SIZE;         // 2048
  constexpr int SUPER_ROUNDS_PER_WARP = BYTES_PER_THREAD / PACK_BYTES; // 8

  static_assert(THREADS_PER_BLOCK % WARP_SIZE == 0);
  static_assert(BYTES_PER_THREAD % PACK_BYTES == 0);
  static_assert(BYTES_PER_WARP * WARPS_PER_BLOCK == CHUNK_SIZE);

  const int tid = threadIdx.x;
  const int tile_idx = blockIdx.x;
  const int block_start = tile_idx * CHUNK_SIZE;
  const int tile_offset = perTileNewlineOffsetIndex[tile_idx];

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  const int warp_global_base = block_start + warp_id * BYTES_PER_WARP;

  // Packed shared-memory layout in ballot order:
  //   smem_packed_bytes[super_round][tid]
  // Each uint2 stores 8 actual file bytes for one lane across 8 consecutive
  // ordinary rounds:
  //   byte i => ordinary_round = super_round * PACK_BYTES + i

  // One bitmap per (warp, super_round, byte_in_pack).
  __shared__ unsigned int
      warp_bitmaps[WARPS_PER_BLOCK * SUPER_ROUNDS_PER_WARP * PACK_BYTES];

  __shared__ int per_warp_totals[WARPS_PER_BLOCK];
  __shared__ int per_warp_offsets[WARPS_PER_BLOCK];

  constexpr unsigned FULL_MASK = 0xffffffffu;

  // --------------------------------------------------------------------------
  // Stage file bytes into shared memory using packed global reads in contiguous
  // file order. Then lane 0 transposes the 8 raw ballot masks into the 8 normal
  // round-major bitmaps expected by the replay phase.
  // --------------------------------------------------------------------------
  const uint2 *file_packed_gmem = reinterpret_cast<const uint2 *>(file);

  int warp_total = 0;

#pragma unroll
  for (int super_round = 0; super_round < SUPER_ROUNDS_PER_WARP;
       ++super_round) {
    // Contiguous packed global load:
    // lane l loads bytes
    //   warp_global_base + super_round * (WARP_SIZE * PACK_BYTES) +
    //   l*PACK_BYTES
    const int packed_global_base = warp_global_base +
                                   super_round * (WARP_SIZE * PACK_BYTES) +
                                   lane_id * PACK_BYTES;

    uint2 packed_word = make_uint2(0u, 0u);
    if (packed_global_base < fileSize) {
      const int packed_global_idx = packed_global_base / PACK_BYTES;
      packed_word = file_packed_gmem[packed_global_idx];
    }

    // Keep the packed bytes around in shared memory. This preserves the packed
    // shared-memory read path used by the prior ballot helper.
    const int smem_idx = super_round * THREADS_PER_BLOCK + tid;

    const unsigned char *packed_chars =
        reinterpret_cast<const unsigned char *>(&packed_word);

    // Raw ballots produced from contiguous packed global loads.
    unsigned int raw_bitmaps[PACK_BYTES];

#pragma unroll
    for (int byte_in_pack = 0; byte_in_pack < PACK_BYTES; ++byte_in_pack) {
      const int global_idx = packed_global_base + byte_in_pack;
      const bool is_newline =
          (global_idx < fileSize && packed_chars[byte_in_pack] == '\n');
      raw_bitmaps[byte_in_pack] = __ballot_sync(FULL_MASK, is_newline);
    }

    // Lane 0 transposes the 8 raw bitmaps into the 8 round-major bitmaps used
    // by the normal ballot replay order.
    if (lane_id == 0) {
      unsigned int transposed_bitmaps[PACK_BYTES];

#pragma unroll
      for (int target_round = 0; target_round < PACK_BYTES; ++target_round) {
        unsigned int dst = 0u;

#pragma unroll
        for (int lane_group = 0; lane_group < PACK_BYTES / 2; ++lane_group) {
          const int src_bit = target_round * (PACK_BYTES / 2) + lane_group;

          unsigned int chunk = 0u;
#pragma unroll
          for (int raw_idx = 0; raw_idx < PACK_BYTES; ++raw_idx) {
            chunk |= ((raw_bitmaps[raw_idx] >> src_bit) & 1u) << raw_idx;
          }

          dst |= chunk << (lane_group * PACK_BYTES);
        }

        transposed_bitmaps[target_round] = dst;
        warp_total += __popc(dst);
      }

#pragma unroll
      for (int byte_in_pack = 0; byte_in_pack < PACK_BYTES; ++byte_in_pack) {
        const int bitmap_idx =
            ((warp_id * SUPER_ROUNDS_PER_WARP + super_round) * PACK_BYTES) +
            byte_in_pack;
        warp_bitmaps[bitmap_idx] = transposed_bitmaps[byte_in_pack];
      }
    }
  }
  __syncthreads();

  if (lane_id == 0) {
    per_warp_totals[warp_id] = warp_total;
  }
  __syncthreads();

  // --------------------------------------------------------------------------
  // Warp 0 scans per-warp totals to get per-warp offsets within the block.
  // --------------------------------------------------------------------------
  if (warp_id == 0) {
    int per_warp_inclusive_offset =
        (lane_id < WARPS_PER_BLOCK) ? per_warp_totals[lane_id] : 0;

#pragma unroll
    for (int pass_offset = 1; pass_offset < WARP_SIZE; pass_offset <<= 1) {
      const int other_warp_count =
          __shfl_up_sync(FULL_MASK, per_warp_inclusive_offset, pass_offset);
      if (lane_id >= pass_offset) {
        per_warp_inclusive_offset += other_warp_count;
      }
    }

    if (lane_id < WARPS_PER_BLOCK) {
      per_warp_offsets[lane_id] =
          per_warp_inclusive_offset - per_warp_totals[lane_id];
    }
  }
  __syncthreads();

  // --------------------------------------------------------------------------
  // Second pass:
  // Replay the stored bitmaps in increasing byte order within the warp's
  // 2048-byte region:
  //   super_round 0, byte 0..7, then super_round 1, ...
  // --------------------------------------------------------------------------
  int warp_running_offset = 0;

#pragma unroll
  for (int super_round = 0; super_round < SUPER_ROUNDS_PER_WARP;
       ++super_round) {
#pragma unroll
    for (int byte_in_pack = 0; byte_in_pack < PACK_BYTES; ++byte_in_pack) {
      const int bitmap_idx =
          ((warp_id * SUPER_ROUNDS_PER_WARP + super_round) * PACK_BYTES) +
          byte_in_pack;
      const unsigned int bitmap = warp_bitmaps[bitmap_idx];
      const int round_count = __popc(bitmap);

      const bool is_newline = ((bitmap >> lane_id) & 1u) != 0u;
      if (is_newline) {
        const unsigned int lower_lane_mask =
            (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
        const int rank_in_round = __popc(bitmap & lower_lane_mask);

        const int ordinary_round = super_round * PACK_BYTES + byte_in_pack;
        const int global_idx =
            warp_global_base + ordinary_round * WARP_SIZE + lane_id;

        // Keep GPJSON's leading sentinel at index 0.
        newlineIndex[tile_offset + per_warp_offsets[warp_id] +
                     warp_running_offset + rank_in_round + 1] =
            static_cast<long>(global_idx);
      }

      warp_running_offset += round_count;
    }
  }
}

__global__ void newline_index(const char *file, int fileSize,
                              int *perTileNewlineOffsetIndex,
                              long *newlineIndex) {
  // newline_index_sharemem_naive(file, fileSize, perTileNewlineOffsetIndex,
  //                              newlineIndex);

  // newline_index_sharemem(file, fileSize, perTileNewlineOffsetIndex,
  //                        newlineIndex);

  // newline_index_sharemem_transposed(file, fileSize,
  // perTileNewlineOffsetIndex,
  //                                   newlineIndex);

  // newline_index_sharemem_transposed_packed(
  //     file, fileSize, perTileNewlineOffsetIndex, newlineIndex);

  // newline_index_sharemem_ballot(file, fileSize, perTileNewlineOffsetIndex,
  //                               newlineIndex);

  // newline_index_sharemem_ballot_packed_smem(
  //     file, fileSize, perTileNewlineOffsetIndex, newlineIndex);
  newline_index_sharemem_ballot_packed_gmem_smem(
      file, fileSize, perTileNewlineOffsetIndex, newlineIndex);
}

} // namespace gpjson::index::kernels::sharemem
