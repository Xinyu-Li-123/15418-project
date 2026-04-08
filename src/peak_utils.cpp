#include "peakfinder/peak_utils.hpp"
#include "peakfinder/log.hpp"

namespace peakfinder {

bool is_peak_at(const std::vector<int> &input, std::size_t i) {
  const std::size_t n = input.size();
  if (n == 0 || i >= n) {
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

std::vector<int> find_peaks_cpu(const std::vector<int> &input) {
  LogInfo("find_peaks_cpu()");
  std::vector<int> peaks;
  peaks.reserve(input.size() / 2 + 1);

  for (std::size_t i = 0; i < input.size(); ++i) {
    if (is_peak_at(input, i)) {
      peaks.push_back(static_cast<int>(i));
    }
  }

  return peaks;
}

} // namespace peakfinder
