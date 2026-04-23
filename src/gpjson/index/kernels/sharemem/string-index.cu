namespace gpjson::index::kernels::sharemem {

__global__ void string_index(long *stringIndex, int stringIndexSize,
                             char *quoteCarryIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int elemsPerThread = (stringIndexSize + stride - 1) / stride;
  int start = index * elemsPerThread;
  int end = start + elemsPerThread;

  long bitString =
      index > 0 && quoteCarryIndex[index - 1] == 1 ? 0xffffffffffffffffL : 0;

  for (int i = start; i < end && i < stringIndexSize; i += 1) {
    long quotes = stringIndex[i];

    quotes ^= quotes << 1;
    quotes ^= quotes << 2;
    quotes ^= quotes << 4;
    quotes ^= quotes << 8;
    quotes ^= quotes << 16;
    quotes ^= quotes << 32;

    quotes = quotes ^ bitString;

    stringIndex[i] = quotes;

    bitString = quotes >> 63;
  }
}

} // namespace gpjson::index::kernels::sharemem
