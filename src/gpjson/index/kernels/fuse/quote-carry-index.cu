namespace gpjson::index::kernels::fuse {

/**
 * Compute quote carry index using escape carry index
 *
 * quoteCarryIndex[i]: does file chunk i ends with a value in string
 *  (we treat starting quote and string body as "in string", and treat ending
 * quote as not in string) i.e. is first char of next chunk inside a string?
 *
 * quoteCarryIndex will be xor prefix summed, just like the original impl.
 *
 * @param[in] file
 * @param[in] fileSize
 * @param[in] escapeCarryIndex: size=numThreads
 * @param[out] quoteCarryIndex: size=numThreads
 */
__global__ void
quote_carry_index_using_escape_carry_index(const char *file, int fileSize,
                                           char *escapeCarryIndex,
                                           char *quoteCarryIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int charsPerThread = (fileSize + stride - 1) / stride;
  int bitmapAlignedCharsPerThread = ((charsPerThread + 64 - 1) / 64) * 64;
  int start = index * bitmapAlignedCharsPerThread;
  int end = start + bitmapAlignedCharsPerThread;

  int quoteCount = 0;

  // escapeCarryIndex[index-1] tells if first char in the chunk assigned to
  // thread index is escaped
  char is_cur_escaped = index == 0 ? 0 : escapeCarryIndex[index - 1];

  for (long i = start; i < end && i < fileSize; i += 1) {
    // if cur char is escaped, next char can't be escaped
    if (is_cur_escaped) {
      is_cur_escaped = 0;
      continue;
    }

    // cur char is not escaped
    char curChar = file[i];
    if (curChar == '"') {
      quoteCount++;
      // since cur is quote, next char can't be escaped
      // is_cur_escaped = 0; is implied by prev if statement
      continue;
    }

    is_cur_escaped = curChar == '\\';
  }

  // carry = parity of quoteCount
  quoteCarryIndex[index] = quoteCount & 1;
}

} // namespace gpjson::index::kernels::fuse
