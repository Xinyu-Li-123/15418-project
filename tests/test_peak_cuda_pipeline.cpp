#include "peakfinder/peak_finder.hpp"
#include "peakfinder/peak_utils.hpp"

#include <gtest/gtest.h>
#include <vector>

TEST(PeakCuda, EmptyInput) {
  std::vector<int> input;
  EXPECT_TRUE(peakfinder::find_peaks_gpu(input).empty());
}

TEST(PeakCuda, SingleElement) {
  std::vector<int> input{9};
  std::vector<int> expected{0};
  EXPECT_EQ(peakfinder::find_peaks_gpu(input), expected);
}

TEST(PeakCuda, DeterministicSmallCase) {
  std::vector<int> input{1, 3, 2, 5, 4, 7, 1};
  std::vector<int> expected{1, 3, 5};
  EXPECT_EQ(peakfinder::find_peaks_gpu(input), expected);
}

TEST(PeakCuda, EndpointPeaks) {
  std::vector<int> input{10, 1, 9};
  std::vector<int> expected{0, 2};
  EXPECT_EQ(peakfinder::find_peaks_gpu(input), expected);
}

TEST(PeakCuda, NoPeaksForFlatArrayLongerThanOne) {
  std::vector<int> input{4, 4, 4, 4, 4};
  std::vector<int> expected{};
  EXPECT_EQ(peakfinder::find_peaks_gpu(input), expected);
}

TEST(PeakCuda, AlternatingPatternManyPeaks) {
  std::vector<int> input{1, 5, 1, 5, 1, 5, 1};
  std::vector<int> expected{1, 3, 5};
  EXPECT_EQ(peakfinder::find_peaks_gpu(input), expected);
}

TEST(PeakCuda, GpuMatchesCpuReferenceSmall) {
  std::vector<int> input{2, 1, 4, 1, 6, 2, 3, 1, 5, 0};
  EXPECT_EQ(peakfinder::find_peaks_gpu(input),
            peakfinder::find_peaks_cpu(input));
}

TEST(PeakCuda, GpuMatchesCpuReferenceLargerDeterministic) {
  std::vector<int> input{3, 1, 7, 2, 8, 1, 0, 9, 3, 2, 5, 4, 6, 1,
                         7, 2, 8, 3, 0, 4, 9, 1, 2, 6, 1, 5, 0, 7,
                         2, 3, 8, 1, 4, 0, 6, 2, 9, 1, 3, 0};

  EXPECT_EQ(peakfinder::find_peaks_gpu(input),
            peakfinder::find_peaks_cpu(input));
}
