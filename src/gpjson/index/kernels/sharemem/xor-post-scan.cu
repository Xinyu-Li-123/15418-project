namespace gpjson::index::kernels::sharemem {

__global__ void xor_post_scan(char *charArr, int n, int stride, char *base) {
  long elemsPerThread = (n + stride - 1) / stride;
  char prev = 0;
  for (long i = 0; i < stride - 1; i++) {
    base[i] = prev;
    prev ^= charArr[elemsPerThread * (i + 1) - 1];
  }
  base[stride - 1] = prev;
}

} // namespace gpjson::index::kernels::sharemem
