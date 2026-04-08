#include "peakfinder/peak_finder.hpp"

#include "peak_kernels.cuh"
#include "peakfinder/cuda_status.hpp"
#include "peakfinder/log.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <vector>

namespace peakfinder {

std::vector<int> find_peaks_gpu(const std::vector<int> &input) {
  LogInfo("find_peaks_gpu()");
  const int n = static_cast<int>(input.size());
  if (n == 0) {
    return {};
  }

  int *d_input = nullptr;
  int *d_flags = nullptr;
  int *d_positions = nullptr;
  int *d_output = nullptr;

  PF_CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(int)));
  PF_CUDA_CHECK(cudaMalloc(&d_flags, n * sizeof(int)));
  PF_CUDA_CHECK(cudaMalloc(&d_positions, n * sizeof(int)));

  try {
    PF_CUDA_CHECK(cudaMemcpy(d_input, input.data(), n * sizeof(int),
                             cudaMemcpyHostToDevice));

    launch_flag_peaks_kernel(d_input, n, d_flags);
    PF_CUDA_CHECK(cudaGetLastError());
    PF_CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_ptr<int> flags_begin(d_flags);
    thrust::device_ptr<int> flags_end(d_flags + n);
    thrust::device_ptr<int> positions_begin(d_positions);

    thrust::exclusive_scan(thrust::device, flags_begin, flags_end,
                           positions_begin);
    PF_CUDA_CHECK(cudaDeviceSynchronize());

    int last_pos = 0;
    int last_flag = 0;
    PF_CUDA_CHECK(cudaMemcpy(&last_pos, d_positions + (n - 1), sizeof(int),
                             cudaMemcpyDeviceToHost));
    PF_CUDA_CHECK(cudaMemcpy(&last_flag, d_flags + (n - 1), sizeof(int),
                             cudaMemcpyDeviceToHost));

    const int peak_count = last_pos + last_flag;
    if (peak_count == 0) {
      PF_CUDA_CHECK(cudaFree(d_input));
      PF_CUDA_CHECK(cudaFree(d_flags));
      PF_CUDA_CHECK(cudaFree(d_positions));
      return {};
    }

    PF_CUDA_CHECK(cudaMalloc(&d_output, peak_count * sizeof(int)));

    launch_scatter_peaks_kernel(d_flags, d_positions, n, d_output);
    PF_CUDA_CHECK(cudaGetLastError());
    PF_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> output(peak_count);
    PF_CUDA_CHECK(cudaMemcpy(output.data(), d_output, peak_count * sizeof(int),
                             cudaMemcpyDeviceToHost));

    PF_CUDA_CHECK(cudaFree(d_input));
    PF_CUDA_CHECK(cudaFree(d_flags));
    PF_CUDA_CHECK(cudaFree(d_positions));
    PF_CUDA_CHECK(cudaFree(d_output));

    return output;
  } catch (...) {
    cudaFree(d_input);
    cudaFree(d_flags);
    cudaFree(d_positions);
    cudaFree(d_output);
    throw;
  }
}

} // namespace peakfinder
