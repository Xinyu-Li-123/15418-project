namespace gpjson::index::kernels::sharemem {
namespace {

__device__ __forceinline__ int chunk_chars_per_thread(int file_size,
                                                      int num_chunks) {
  return (file_size + num_chunks - 1) / num_chunks;
}

__device__ __forceinline__ int chunk_aligned_chars_per_thread(int file_size,
                                                              int num_chunks) {
  const int chars = chunk_chars_per_thread(file_size, num_chunks);
  return ((chars + 64 - 1) / 64) * 64;
}

} // namespace

__global__ void string_index(long *string_index, int string_index_size,
                             const char *quote_carry_index, int file_size,
                             int num_chunks) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_chunks) {
    return;
  }

  const int aligned_chars_per_chunk =
      chunk_aligned_chars_per_thread(file_size, num_chunks);
  const int start = index * aligned_chars_per_chunk;
  const int end = min(start + aligned_chars_per_chunk, file_size);

  const int word_start = start / 64;
  const int word_end = (end + 63) / 64;

  auto *string_index_u64 = reinterpret_cast<unsigned long long *>(string_index);

  unsigned long long bit_string =
      (index > 0 && quote_carry_index[index - 1] == 1) ? 0xffffffffffffffffull
                                                       : 0ull;

  for (int word = word_start; word < word_end && word < string_index_size;
       ++word) {
    unsigned long long quotes = string_index_u64[word];

    quotes ^= quotes << 1;
    quotes ^= quotes << 2;
    quotes ^= quotes << 4;
    quotes ^= quotes << 8;
    quotes ^= quotes << 16;
    quotes ^= quotes << 32;

    quotes ^= bit_string;
    string_index_u64[word] = quotes;
    bit_string = ((quotes >> 63) != 0ull) ? 0xffffffffffffffffull : 0ull;
  }
}

} // namespace gpjson::index::kernels::sharemem
