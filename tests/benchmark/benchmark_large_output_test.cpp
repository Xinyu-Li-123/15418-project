#include "benchmark/benchmark_output_test_utils.hpp"
#include "utils/utils.hpp"

namespace {

using gpjson::test::benchmark_output::BenchmarkOutputTestBase;
using gpjson::test::benchmark_output::BenchmarkRunOptions;
using gpjson::test::benchmark_output::kTwitterBenchmarkCases;
using gpjson::test::benchmark_output::project_source_dir;

std::filesystem::path twitter_large_dataset_path() {
  return project_source_dir() / "dataset" / "twitter_sample_large_record.json";
}

std::string trim_carriage_return(std::string line) {
  while (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  return line;
}

class BenchmarkLargeOutputTest : public BenchmarkOutputTestBase {
protected:
  void TearDown() override {
    if (!normalized_dataset_path_.empty()) {
      std::error_code error;
      std::filesystem::remove(normalized_dataset_path_, error);
    }
  }

  std::filesystem::path normalized_large_dataset_path() {
    if (!normalized_dataset_path_.empty()) {
      return normalized_dataset_path_;
    }

    const auto source_path = twitter_large_dataset_path();
    if (!std::filesystem::exists(source_path)) {
      ADD_FAILURE() << "Missing dataset: " << source_path;
      return {};
    }

    normalized_dataset_path_ = std::filesystem::temp_directory_path() /
                               "gpjson_twitter_large_record.ldjson";
    std::ifstream input(source_path, std::ios::binary);
    std::ofstream output(normalized_dataset_path_,
                         std::ios::binary | std::ios::trunc);
    if (!input.is_open() || !output.is_open()) {
      ADD_FAILURE() << "Failed to normalize large dataset from " << source_path
                    << " to " << normalized_dataset_path_;
      return {};
    }

    std::string line;
    while (std::getline(input, line)) {
      line = trim_carriage_return(std::move(line));
      if (line.empty() || line == "[" || line == "]" || line == ",") {
        continue;
      }
      if (!line.empty() && line.back() == ',') {
        line.pop_back();
      }
      output << line << '\n';
    }

    return normalized_dataset_path_;
  }

private:
  std::filesystem::path normalized_dataset_path_;
};

TEST_F(BenchmarkLargeOutputTest, WritesTT1LargeOutputFile) {
  GPJSON_SKIP_IF_CUDA_UNAVAILABLE();

  const auto run_result = run_case(
      normalized_large_dataset_path(), kTwitterBenchmarkCases[0],
      BenchmarkRunOptions{"_large"});
  EXPECT_EQ(run_result.output_path.filename(), "tt1_large_output.txt");
}

TEST_F(BenchmarkLargeOutputTest, WritesTT2LargeOutputFile) {
  GPJSON_SKIP_IF_CUDA_UNAVAILABLE();

  const auto run_result = run_case(
      normalized_large_dataset_path(), kTwitterBenchmarkCases[1],
      BenchmarkRunOptions{"_large"});
  EXPECT_EQ(run_result.output_path.filename(), "tt2_large_output.txt");
}

TEST_F(BenchmarkLargeOutputTest, WritesTT3LargeOutputFile) {
  GPJSON_SKIP_IF_CUDA_UNAVAILABLE();

  const auto run_result = run_case(
      normalized_large_dataset_path(), kTwitterBenchmarkCases[2],
      BenchmarkRunOptions{"_large"});
  EXPECT_EQ(run_result.output_path.filename(), "tt3_large_output.txt");
}

TEST_F(BenchmarkLargeOutputTest, WritesTT4LargeOutputFile) {
  GPJSON_SKIP_IF_CUDA_UNAVAILABLE();

  const auto run_result = run_case(
      normalized_large_dataset_path(), kTwitterBenchmarkCases[3],
      BenchmarkRunOptions{"_large"});
  EXPECT_EQ(run_result.output_path.filename(), "tt4_large_output.txt");
}

} // namespace
