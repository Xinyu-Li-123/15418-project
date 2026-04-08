#pragma once

#include <cuda_runtime.h>

namespace peakfinder {

// Kernel launch wrappers exposed to host-side C++ code.
void launch_flag_peaks_kernel(const int *d_input, int n, int *d_flags,
                              cudaStream_t stream = nullptr);

void launch_scatter_peaks_kernel(const int *d_flags, const int *d_positions,
                                 int n, int *d_output_indices,
                                 cudaStream_t stream = nullptr);

} // namespace peakfinder
