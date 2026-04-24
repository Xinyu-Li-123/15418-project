namespace gpjson::index::kernels::sharemem {

__global__ void int_sum_pre_scan(int *intArr, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  long elemsPerThread = (n + stride - 1) / stride;

  long start = index * elemsPerThread;
  long end = start + elemsPerThread;

  int sum = 0;
  for (long i = start; i < end && i < n; i++) {
    sum += intArr[i];
    intArr[i] = sum;
  }
}

} // namespace gpjson::index::kernels::sharemem
