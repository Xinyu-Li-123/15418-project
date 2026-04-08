#pragma once

#include <vector>

namespace peakfinder {

// End-to-end GPU pipeline:
// 1) generate flags on device
// 2) exclusive_scan(flags)
// 3) scatter peak indices
// 4) return compact host vector of indices
std::vector<int> find_peaks_gpu(const std::vector<int> &input);

} // namespace peakfinder
