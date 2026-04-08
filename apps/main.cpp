#include "peakfinder/peak_finder.hpp"
#include "peakfinder/peak_utils.hpp"

#include <iostream>
#include <vector>

int main() {
  std::vector<int> input{1, 3, 2, 5, 4, 7, 1, 6, 6, 2};

  auto gpu_peaks = peakfinder::find_peaks_gpu(input);
  auto cpu_peaks = peakfinder::find_peaks_cpu(input);

  std::cout << "Input: ";
  for (int x : input) {
    std::cout << x << ' ';
  }
  std::cout << "\nGPU peaks at indices: ";
  for (int i : gpu_peaks) {
    std::cout << i << ' ';
  }
  std::cout << "\nCPU peaks at indices: ";
  for (int i : cpu_peaks) {
    std::cout << i << ' ';
  }
  std::cout << '\n';

  return 0;
}
