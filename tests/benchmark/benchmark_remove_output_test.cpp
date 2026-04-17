#include "benchmark/benchmark_output_test_utils.hpp"
#include "utils/utils.hpp"

namespace {

using gpjson::test::benchmark_output::BenchmarkOutputTestBase;
using gpjson::test::benchmark_output::BenchmarkResultMode;
using gpjson::test::benchmark_output::BenchmarkRunOptions;
using gpjson::test::benchmark_output::kTwitterBenchmarkCases;
using gpjson::test::benchmark_output::project_source_dir;

std::filesystem::path twitter_remove_dataset_path() {
  return project_source_dir() / "dataset" / "twitter_small_records_remove.json";
}

class BenchmarkRemoveOutputTest : public BenchmarkOutputTestBase {};

TEST_F(BenchmarkRemoveOutputTest, WritesTT1RemoveOutputFile) {
  GPJSON_SKIP_IF_CUDA_UNAVAILABLE();

  const auto run_result =
      run_case(twitter_remove_dataset_path(), kTwitterBenchmarkCases[0],
               BenchmarkRunOptions{"_remove",
                                   BenchmarkResultMode::CountMatchesOnly});
  EXPECT_TRUE(run_result.output_path.empty());
  EXPECT_EQ(run_result.query_match_counts.size(), 1U);
}

TEST_F(BenchmarkRemoveOutputTest, WritesTT2RemoveOutputFile) {
  GPJSON_SKIP_IF_CUDA_UNAVAILABLE();

  const auto run_result =
      run_case(twitter_remove_dataset_path(), kTwitterBenchmarkCases[1],
               BenchmarkRunOptions{"_remove",
                                   BenchmarkResultMode::CountMatchesOnly});
  EXPECT_TRUE(run_result.output_path.empty());
  EXPECT_EQ(run_result.query_match_counts.size(), 2U);
}

TEST_F(BenchmarkRemoveOutputTest, WritesTT3RemoveOutputFile) {
  GPJSON_SKIP_IF_CUDA_UNAVAILABLE();

  const auto run_result =
      run_case(twitter_remove_dataset_path(), kTwitterBenchmarkCases[2],
               BenchmarkRunOptions{"_remove",
                                   BenchmarkResultMode::CountMatchesOnly});
  EXPECT_TRUE(run_result.output_path.empty());
  EXPECT_EQ(run_result.query_match_counts.size(), 1U);
}

TEST_F(BenchmarkRemoveOutputTest, WritesTT4RemoveOutputFile) {
  GPJSON_SKIP_IF_CUDA_UNAVAILABLE();

  const auto run_result =
      run_case(twitter_remove_dataset_path(), kTwitterBenchmarkCases[3],
               BenchmarkRunOptions{"_remove",
                                   BenchmarkResultMode::CountMatchesOnly});
  EXPECT_TRUE(run_result.output_path.empty());
  EXPECT_EQ(run_result.query_match_counts.size(), 1U);
}

} // namespace
