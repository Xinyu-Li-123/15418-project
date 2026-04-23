#include <cassert>

namespace gpjson::index::kernels::sharemem {
namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxNumLevels = 22;

__device__ __forceinline__ int lane_id() {
  return threadIdx.x & (kWarpSize - 1);
}

__device__ __forceinline__ int warp_id_in_block() {
  return threadIdx.x / kWarpSize;
}

__device__ __forceinline__ int global_warp_id() {
  const int warps_per_block = blockDim.x / kWarpSize;
  return blockIdx.x * warps_per_block + warp_id_in_block();
}

__device__ __forceinline__ int chunk_chars_per_warp(int file_size,
                                                    int num_chunks) {
  return (file_size + num_chunks - 1) / num_chunks;
}

__device__ __forceinline__ int chunk_aligned_chars_per_warp(int file_size,
                                                            int num_chunks) {
  const int chars = chunk_chars_per_warp(file_size, num_chunks);
  return ((chars + 64 - 1) / 64) * 64;
}

} // namespace

/**
 * Warp-wide version of leveled_bitmaps_index.
 *
 * We compute, for each structural byte outside strings, the target level:
 *   - open  ('{' or '['):  level_after_open  = level_before + 1
 *   - close ('}' or ']'):  level_before
 *   - sep   (':' or ','):  level_before
 *
 * Then for each level < num_levels we ballot the matching lanes and pack
 * the result into the per-level bitmap arrays.
 *
 * This replaces the old lane-0 byte loop with warp-wide prefix-sum logic.
 */
__global__ void leveled_bitmaps_index(const char *file, int file_size,
                                      const long *string_index,
                                      const char *leveled_bitmaps_aux_index,
                                      long *leveled_bitmaps_index,
                                      int level_size, int num_levels,
                                      int num_chunks) {
  assert(num_levels <= kMaxNumLevels);

  const unsigned full_mask = 0xffffffffu;
  const int warp = global_warp_id();
  if (warp >= num_chunks) {
    return;
  }

  const int lane = lane_id();

  const int aligned_chars_per_chunk =
      chunk_aligned_chars_per_warp(file_size, num_chunks);
  const int start = warp * aligned_chars_per_chunk;
  const int end = min(start + aligned_chars_per_chunk, file_size);

  auto *string_index_u64 =
      reinterpret_cast<const unsigned long long *>(string_index);
  auto *leveled_bitmaps_index_u64 =
      reinterpret_cast<unsigned long long *>(leveled_bitmaps_index);

  int tile_start_level =
      static_cast<signed char>(leveled_bitmaps_aux_index[warp]);

  for (int base = start; base < end; base += 64) {
    const unsigned long long string_word = string_index_u64[base / 64];
    unsigned long long level_masks[kMaxNumLevels];

    if (lane == 0) {
      for (int level = 0; level < num_levels; ++level) {
        level_masks[level] = 0ull;
      }
    }

#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int pos = base + half * 32 + lane;
      const bool in_bounds = pos < end;
      const char c = in_bounds ? file[pos] : 0;

      const int bit_in_word = half * 32 + lane;
      const bool in_string =
          in_bounds && (((string_word >> bit_in_word) & 1ull) != 0ull);

      const bool is_open = !in_string && in_bounds && (c == '{' || c == '[');
      const bool is_close = !in_string && in_bounds && (c == '}' || c == ']');
      const bool is_sep = !in_string && in_bounds && (c == ':' || c == ',');

      // Delta contributed by this byte to the running level after it.
      int delta = 0;
      if (is_open) {
        delta = 1;
      } else if (is_close) {
        delta = -1;
      }

      // Inclusive scan of delta, then exclusive prefix.
      int prefix = delta;
      for (int offset = 1; offset < 32; offset <<= 1) {
        const int v = __shfl_up_sync(full_mask, prefix, offset);
        if (lane >= offset) {
          prefix += v;
        }
      }
      const int excl_prefix = prefix - delta;

      // Level before this byte.
      const int level_before = tile_start_level + excl_prefix;
      const int level_after_open = level_before + 1;

      // Structural target level for this lane.
      int target_level = -9999;
      bool emits = false;

      if (is_open) {
        target_level = level_after_open;
        emits = true;
      } else if (is_close || is_sep) {
        target_level = level_before;
        emits = true;
      }

      // For each level, ballot the lanes that emit into that level.
      for (int level = 0; level < num_levels; ++level) {
        const bool match = emits && (target_level == level);
        const unsigned level_mask32 = __ballot_sync(full_mask, match);
        if (lane == 0) {
          level_masks[level] |=
              (static_cast<unsigned long long>(level_mask32) << (half * 32));
        }
      }

      // Carry tile start level forward by the total delta of this half-tile.
      const int tile_delta = __shfl_sync(full_mask, prefix, 31);
      tile_start_level += tile_delta;
    }

    if (lane == 0) {
      const int word_id = base / 64;
      for (int level = 0; level < num_levels; ++level) {
        leveled_bitmaps_index_u64[level_size * level + word_id] =
            level_masks[level];
      }
    }
  }
}

} // namespace gpjson::index::kernels::sharemem
