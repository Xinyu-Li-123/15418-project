#include "peak_kernels.cuh"

#include <cuda_runtime.h>

namespace peakfinder {
namespace {

__device__ bool is_peak_device(const int *input, int n, int i) {
  if (n <= 0 || i < 0 || i >= n) {
    return false;
  }
  if (n == 1) {
    return true;
  }
  if (i == 0) {
    return input[0] > input[1];
  }
  if (i == n - 1) {
    return input[n - 1] > input[n - 2];
  }
  return input[i] > input[i - 1] && input[i] > input[i + 1];
}

__global__ void flag_peaks_kernel(const int *input, int n, int *flags) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    flags[i] = is_peak_device(input, n, i) ? 1 : 0;
  }
}

__global__ void scatter_peaks_kernel(const int *flags, const int *positions,
                                     int n, int *output_indices) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && flags[i] != 0) {
    const int out_idx = positions[i];
    output_indices[out_idx] = i;
  }
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

} // namespace

void launch_flag_peaks_kernel(const int *d_input, int n, int *d_flags,
                              cudaStream_t stream) {
  constexpr int block_size = 256;
  const int grid_size = ceil_div(n, block_size);
  flag_peaks_kernel<<<grid_size, block_size, 0, stream>>>(d_input, n, d_flags);
}

void launch_scatter_peaks_kernel(const int *d_flags, const int *d_positions,
                                 int n, int *d_output_indices,
                                 cudaStream_t stream) {
  constexpr int block_size = 256;
  const int grid_size = ceil_div(n, block_size);
  scatter_peaks_kernel<<<grid_size, block_size, 0, stream>>>(
      d_flags, d_positions, n, d_output_indices);
}

} // namespace peakfinder
