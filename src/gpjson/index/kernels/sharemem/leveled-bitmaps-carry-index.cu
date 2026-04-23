namespace gpjson::index::kernels::sharemem {
namespace {

constexpr int kWarpSize = 32;
constexpr int kTileBytes = 64;
constexpr int kMaxWarpsPerBlock = 32;

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

__global__ void leveled_bitmaps_carry_index(const char *file, int file_size,
                                            const long *string_index,
                                            char *leveled_bitmaps_aux_index,
                                            int num_chunks) {
  __shared__ unsigned char tile[kMaxWarpsPerBlock][kTileBytes];

  const int warp = global_warp_id();
  if (warp >= num_chunks) {
    return;
  }

  const int lane = lane_id();
  const int warp_local = warp_id_in_block();

  const int aligned_chars_per_chunk =
      chunk_aligned_chars_per_warp(file_size, num_chunks);
  const int start = warp * aligned_chars_per_chunk;
  const int end = min(start + aligned_chars_per_chunk, file_size);

  auto *string_index_u64 =
      reinterpret_cast<const unsigned long long *>(string_index);

  signed char level = 0;

  for (int base = start; base < end; base += kTileBytes) {
    const int pos0 = base + lane;
    const int pos1 = base + kWarpSize + lane;

    tile[warp_local][lane] =
        (pos0 < end) ? static_cast<unsigned char>(file[pos0]) : 0;
    tile[warp_local][kWarpSize + lane] =
        (pos1 < end) ? static_cast<unsigned char>(file[pos1]) : 0;
    __syncwarp();

    if (lane == 0) {
      const unsigned long long string_word = string_index_u64[base / 64];
      const int valid = min(kTileBytes, end - base);

      for (int i = 0; i < valid; ++i) {
        if (((string_word >> i) & 1ull) != 0ull) {
          continue;
        }

        const unsigned char c = tile[warp_local][i];
        if (c == static_cast<unsigned char>('{') ||
            c == static_cast<unsigned char>('[')) {
          ++level;
        } else if (c == static_cast<unsigned char>('}') ||
                   c == static_cast<unsigned char>(']')) {
          --level;
        }
      }
    }
    __syncwarp();
  }

  if (lane == 0) {
    leveled_bitmaps_aux_index[warp] = static_cast<char>(level);
  }
}

} // namespace gpjson::index::kernels::sharemem
