#include "peakfinder/peak_utils.hpp"

#include <gtest/gtest.h>
#include <vector>

TEST(PeakUtils, EmptyInputHasNoPeaks) {
  std::vector<int> input;
  EXPECT_TRUE(peakfinder::find_peaks_cpu(input).empty());
}

TEST(PeakUtils, SingleElementIsPeak) {
  std::vector<int> input{42};
  auto peaks = peakfinder::find_peaks_cpu(input);
  ASSERT_EQ(peaks.size(), 1u);
  EXPECT_EQ(peaks[0], 0);
}

TEST(PeakUtils, StrictInteriorPeaks) {
  std::vector<int> input{1, 3, 2, 5, 4, 7, 1};
  std::vector<int> expected{1, 3, 5};
  EXPECT_EQ(peakfinder::find_peaks_cpu(input), expected);
}

TEST(PeakUtils, EndpointsCanBePeaks) {
  std::vector<int> input{5, 1, 4};
  std::vector<int> expected{0, 2};
  EXPECT_EQ(peakfinder::find_peaks_cpu(input), expected);
}

TEST(PeakUtils, PlateauIsNotPeak) {
  std::vector<int> input{1, 3, 3, 1};
  std::vector<int> expected{};
  EXPECT_EQ(peakfinder::find_peaks_cpu(input), expected);
}

TEST(PeakUtils, IsPeakAtMatchesDefinition) {
  std::vector<int> input{1, 4, 2, 6, 5};
  EXPECT_FALSE(peakfinder::is_peak_at(input, 0));
  EXPECT_TRUE(peakfinder::is_peak_at(input, 1));
  EXPECT_FALSE(peakfinder::is_peak_at(input, 2));
  EXPECT_TRUE(peakfinder::is_peak_at(input, 3));
  EXPECT_FALSE(peakfinder::is_peak_at(input, 4));
}
