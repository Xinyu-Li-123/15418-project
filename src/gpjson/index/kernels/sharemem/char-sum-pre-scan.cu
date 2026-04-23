namespace gpjson::index::kernels::sharemem {

__global__ void char_sum_pre_scan(char *charArr, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  long elemsPerThread = (n + stride - 1) / stride;

  long start = index * elemsPerThread;
  long end = start + elemsPerThread;

  char sum = 0;
  for (long i = start; i < end && i < n; i++) {
    sum += charArr[i];
    charArr[i] = sum;
  }
}

} // namespace gpjson::index::kernels::sharemem
