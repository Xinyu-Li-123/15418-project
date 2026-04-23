namespace gpjson::index::kernels::sharemem {
namespace {

constexpr int kWarpSize = 32;

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
 * Warp-wide version of leveled_bitmaps_carry_index.
 *
 * Output:
 *   leveled_bitmaps_aux_index[chunk] = net level delta of the chunk
 *
 * We exclude bytes that are inside strings by consulting string_index.
 * One warp owns one chunk; each 64-byte region is processed as two 32-byte
 * half-tiles. All byte classification is done lane-parallel.
 */
__global__ void leveled_bitmaps_carry_index(const char *file, int file_size,
                                            const long *string_index,
                                            char *leveled_bitmaps_aux_index,
                                            int num_chunks) {
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

  int level_delta = 0;

  for (int base = start; base < end; base += 64) {
    const unsigned long long string_word = string_index_u64[base / 64];

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

      const unsigned open_mask = __ballot_sync(full_mask, is_open);
      const unsigned close_mask = __ballot_sync(full_mask, is_close);

      // All lanes see the same masks, so all can update the same scalar.
      level_delta += __popc(open_mask) - __popc(close_mask);
    }
  }

  if (lane == 0) {
    leveled_bitmaps_aux_index[warp] = static_cast<char>(level_delta);
  }
}

} // namespace gpjson::index::kernels::sharemem
