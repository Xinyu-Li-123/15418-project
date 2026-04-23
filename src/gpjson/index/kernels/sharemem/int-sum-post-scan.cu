namespace gpjson::index::kernels::sharemem {

__global__ void int_sum_post_scan(int *intArr, int n, int stride,
                                  int startingValue, int *base) {
  long elemsPerThread = (n + stride - 1) / stride;
  int sum = startingValue;
  for (long i = 0; i < stride - 1; i++) {
    base[i] = sum;
    sum += intArr[elemsPerThread * (i + 1) - 1];
  }
  base[stride - 1] = sum;
}

} // namespace gpjson::index::kernels::sharemem
