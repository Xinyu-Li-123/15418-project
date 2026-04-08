#pragma once

#include <cstddef>
#include <vector>

namespace peakfinder {

// Peak definition used by both CPU and GPU:
// i == 0      -> input[0] > input[1]
// i == n - 1  -> input[n-1] > input[n-2]
// otherwise   -> input[i] > input[i-1] && input[i] > input[i+1]
bool is_peak_at(const std::vector<int> &input, std::size_t i);

// CPU reference implementation for testing and validation.
std::vector<int> find_peaks_cpu(const std::vector<int> &input);

} // namespace peakfinder
