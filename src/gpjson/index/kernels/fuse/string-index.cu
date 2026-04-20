namespace gpjson::index::kernels::fuse {

/* Set bit at offset to be bit_value in bitmap */
__device__ static void set_bit_at_offset(long &bitmap, int offset,
                                         bool bit_value) {
  if (!bit_value) {
    return;
  }
  bitmap |= 1L << offset;
}

/**
 * Compute string index using escape carry and quote carry index
 * impl)
 *
 * @param[in] file
 * @param[in] fileSize
 * @param[in] escapeCarryIndex
 * @param[in] quoteCarryIndex, after xor prefix sum the output of
 *  quote_carry_index_using_escape_carry_index
 * @param[out] stringIndex
 * @param[in] stringIndexSize
 *
 */
__global__ void string_index_using_escape_carry_index_quote_carry_index(
    const char *file, int fileSize, char *escapeCarryIndex,
    char *quoteCarryIndex, long *stringIndex, int stringIndexSize) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int charsPerThread = (fileSize + stride - 1) / stride;
  // a chunk is 64 bit
  int numChunk = (charsPerThread + 64 - 1) / 64;
  int bitmapAlignedCharsPerThread = numChunk * 64;

  int start = index * bitmapAlignedCharsPerThread;
  int end = start + bitmapAlignedCharsPerThread;

  // escapeCarryIndex[index-1] tells if first char in the chunk assigned to
  // thread index is escaped
  bool is_cur_escaped = index == 0 ? false : escapeCarryIndex[index - 1];
  bool is_cur_quoted = index == 0 ? false : quoteCarryIndex[index - 1];

  int i = start;
  // The file is divided into a large sequence of 64-bit chunks.
  // We use a long to represent the bitmap for each 64-bit chunk.
  // Each thread is assigned a sub-sequence of 64-bit chunks
  // relChunkIdx is the index into this per-thread sequence
  // chunkId is the index into this large sequence of file chunks
  for (int relChunkIdx = 0; relChunkIdx < numChunk; relChunkIdx++) {
    long curChunkStringBitmap = 0;
    int chunkId = i / 64;
    for (int chunkOffset = 0; chunkOffset < 64; chunkOffset++) {
      if (!(i < end && i < fileSize)) {
        stringIndex[chunkId] = curChunkStringBitmap;
        return;
      }
      char curChar = file[i];
      i += 1;

      // if cur char is escaped, next char can't be escaped
      if (is_cur_escaped) {
        set_bit_at_offset(curChunkStringBitmap, chunkOffset, is_cur_quoted);
        is_cur_escaped = 0;
        continue;
      }

      // cur char is not escaped

      // if cur char is quote, we need to flip is_cur_str
      if (curChar == '"') {
        is_cur_quoted = !is_cur_quoted;
        set_bit_at_offset(curChunkStringBitmap, chunkOffset, is_cur_quoted);
        continue;
      }

      // if cur char is backslash, next char is escaped
      is_cur_escaped = (curChar == '\\');
      set_bit_at_offset(curChunkStringBitmap, chunkOffset, is_cur_quoted);
    }
    stringIndex[chunkId] = curChunkStringBitmap;
  }
}

} // namespace gpjson::index::kernels::fuse
