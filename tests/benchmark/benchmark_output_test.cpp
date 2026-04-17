#include "benchmark/benchmark_output_test_utils.hpp"
#include "utils/utils.hpp"

namespace {
using gpjson::test::benchmark_output::BenchmarkOutputTestBase;
using gpjson::test::benchmark_output::BenchmarkRunOptions;
using gpjson::test::benchmark_output::kTwitterBenchmarkCases;
using gpjson::test::benchmark_output::project_source_dir;

std::filesystem::path twitter_dataset_path() {
  return project_source_dir() / "dataset" / "twitter_sample_small_records.json";
}

class BenchmarkOutputTest : public BenchmarkOutputTestBase {};

TEST_F(BenchmarkOutputTest, WritesTT1OutputFile) {
  GPJSON_SKIP_IF_CUDA_UNAVAILABLE();

  const auto run_result = run_case(
      twitter_dataset_path(), kTwitterBenchmarkCases[0], BenchmarkRunOptions{});
  EXPECT_EQ(run_result.output_path.filename(), "tt1_output.txt");
}

TEST_F(BenchmarkOutputTest, WritesTT2OutputFile) {
  GPJSON_SKIP_IF_CUDA_UNAVAILABLE();

  const auto run_result = run_case(
      twitter_dataset_path(), kTwitterBenchmarkCases[1], BenchmarkRunOptions{});
  EXPECT_EQ(run_result.output_path.filename(), "tt2_output.txt");
}

TEST_F(BenchmarkOutputTest, WritesTT3OutputFile) {
  GPJSON_SKIP_IF_CUDA_UNAVAILABLE();

  const auto run_result = run_case(
      twitter_dataset_path(), kTwitterBenchmarkCases[2], BenchmarkRunOptions{});
  EXPECT_EQ(run_result.output_path.filename(), "tt3_output.txt");
}

TEST_F(BenchmarkOutputTest, WritesTT4OutputFile) {
  GPJSON_SKIP_IF_CUDA_UNAVAILABLE();

  const auto run_result = run_case(
      twitter_dataset_path(), kTwitterBenchmarkCases[3], BenchmarkRunOptions{});
  EXPECT_EQ(run_result.output_path.filename(), "tt4_output.txt");
}

} // namespace
