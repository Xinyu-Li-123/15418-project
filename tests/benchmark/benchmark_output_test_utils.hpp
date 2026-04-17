#pragma once

#include "gpjson/engine.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef GPJSON_PROJECT_SOURCE_DIR
#error "GPJSON_PROJECT_SOURCE_DIR must be defined for benchmark output tests."
#endif

#ifndef GPJSON_BENCHMARK_OUTPUT_DIR
#error "GPJSON_BENCHMARK_OUTPUT_DIR must be defined for benchmark output tests."
#endif

namespace gpjson::test::benchmark_output {

struct BenchmarkQueryCase {
  std::string name;
  std::vector<std::string> queries;
};

enum class BenchmarkResultMode {
  StoreResultInFile,
  CountMatchesOnly,
};

struct BenchmarkRunOptions {
  std::string output_suffix;
  BenchmarkResultMode result_mode{BenchmarkResultMode::StoreResultInFile};
};

struct BenchmarkRunResult {
  std::filesystem::path output_path;
  size_t total_match_count{0};
  std::vector<size_t> query_match_counts;
};

inline const std::array<BenchmarkQueryCase, 4> kTwitterBenchmarkCases{{
    {
        "TT1",
        {"$.user.lang"},
    },
    {
        "TT2",
        {"$.user.lang", "$.lang"},
    },
    {
        "TT3",
        {"$.user.lang[?(@ == \"nl\")]"},
    },
    {
        "TT4",
        {"$.user.lang[?(@ == \"en\")]"},
    },
}};

inline std::filesystem::path project_source_dir() {
  return std::filesystem::path(GPJSON_PROJECT_SOURCE_DIR);
}

inline std::filesystem::path output_dir() {
  return std::filesystem::path(GPJSON_BENCHMARK_OUTPUT_DIR);
}

inline std::string lowercase(std::string value) {
  for (char &character : value) {
    character = static_cast<char>(
        std::tolower(static_cast<unsigned char>(character)));
  }
  return value;
}

inline std::string serialize_result(
    const std::filesystem::path &dataset_path,
    const BenchmarkQueryCase &query_case,
    const gpjson::query::MaterializedBatchResult &result) {
  std::ostringstream output;
  output << "case=" << query_case.name << '\n';
  output << "dataset=" << dataset_path.string() << '\n';
  output << "query_count=" << result.queries().size() << "\n\n";

  for (size_t query_index = 0; query_index < result.queries().size();
       ++query_index) {
    const auto &query_result = result.queries()[query_index];
    output << "[query " << query_index << "] " << query_result.query_text()
           << '\n';
    output << "line_count=" << query_result.lines().size() << '\n';

    for (size_t line_index = 0; line_index < query_result.lines().size();
         ++line_index) {
      output << line_index << ": ";
      const auto &line = query_result.lines()[line_index];
      if (line.values().empty()) {
        output << "[]";
      } else {
        for (size_t value_index = 0; value_index < line.values().size();
             ++value_index) {
          if (value_index > 0) {
            output << " | ";
          }
          output << line.values()[value_index].json_text();
        }
      }
      output << '\n';
    }

    output << '\n';
  }

  return output.str();
}

inline gpjson::EngineOptions benchmark_engine_options() {
  return gpjson::EngineOptions{
      gpjson::index::IndexBuilderOptions{
          gpjson::index::IndexBuilderType::COMBINED,
          65536,
          1,
          256,
          32,
          32,
      },
  };
}

inline BenchmarkRunResult
count_matches(const gpjson::query::MaterializedBatchResult &result) {
  BenchmarkRunResult run_result;
  run_result.query_match_counts.reserve(result.queries().size());

  for (const auto &query_result : result.queries()) {
    size_t query_match_count = 0;
    for (const auto &line_result : query_result.lines()) {
      query_match_count += line_result.values().size();
    }
    run_result.total_match_count += query_match_count;
    run_result.query_match_counts.push_back(query_match_count);
  }

  return run_result;
}

inline void print_match_counts(
    const std::filesystem::path &dataset_path,
    const BenchmarkQueryCase &query_case,
    const gpjson::query::MaterializedBatchResult &result,
    const BenchmarkRunResult &run_result) {
  std::cout << "case=" << query_case.name << '\n';
  std::cout << "dataset=" << dataset_path.string() << '\n';
  std::cout << "total_match_count=" << run_result.total_match_count << '\n';

  for (size_t query_index = 0; query_index < result.queries().size();
       ++query_index) {
    std::cout << "[query " << query_index << "] "
              << result.queries()[query_index].query_text()
              << " match_count=" << run_result.query_match_counts[query_index]
              << '\n';
  }
  std::cout << std::flush;
}

class BenchmarkOutputTestBase : public ::testing::Test {
protected:
  BenchmarkRunResult run_case(
      const std::filesystem::path &dataset_path,
      const BenchmarkQueryCase &query_case,
      const BenchmarkRunOptions &options = BenchmarkRunOptions{}) {
    if (!std::filesystem::exists(dataset_path)) {
      ADD_FAILURE() << "Missing dataset: " << dataset_path;
      return BenchmarkRunResult{};
    }

    const auto result = engine_.query(dataset_path.string(), query_case.queries,
                                      benchmark_engine_options());
    if (result.queries().size() != query_case.queries.size()) {
      ADD_FAILURE() << "Expected " << query_case.queries.size()
                    << " query results, got " << result.queries().size();
      return BenchmarkRunResult{};
    }

    BenchmarkRunResult run_result = count_matches(result);
    
    if (options.result_mode == BenchmarkResultMode::CountMatchesOnly) {
      std::cout << "=== Match Counts Only ===\n";
      print_match_counts(dataset_path, query_case, result, run_result);
      return run_result;
    }
    std::cout << "=== Full Result Output ===\n";
    std::filesystem::create_directories(output_dir());
    run_result.output_path =
        output_dir() / (lowercase(query_case.name) + options.output_suffix +
                        "_output.txt");

    std::ofstream output(run_result.output_path,
                         std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
      ADD_FAILURE() << "Failed to open output file: " << run_result.output_path;
      return BenchmarkRunResult{};
    }
    output << serialize_result(dataset_path, query_case, result);
    output.close();

    EXPECT_TRUE(std::filesystem::exists(run_result.output_path));
    EXPECT_GT(std::filesystem::file_size(run_result.output_path), 0U);
    return run_result;
  }

private:
  gpjson::Engine engine_;
};

} // namespace gpjson::test::benchmark_output
