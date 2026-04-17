#include "gpjson/cuda/cuda.hpp"
#include "gpjson/engine.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifndef GPJSON_PROJECT_SOURCE_DIR
#error "GPJSON_PROJECT_SOURCE_DIR must be defined for the app target."
#endif

namespace {

std::filesystem::path dataset_path() {
  return std::filesystem::path(GPJSON_PROJECT_SOURCE_DIR) / "dataset" /
         "twitter_small_records_remove.json";
}

gpjson::EngineOptions benchmark_engine_options() {
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

size_t match_count(const gpjson::query::MaterializedQueryResult &query_result) {
  size_t total = 0;
  for (const auto &line_result : query_result.lines()) {
    total += line_result.values().size();
  }
  return total;
}

} // namespace

int main() {
  try {
    const std::filesystem::path input_path = dataset_path();
    if (!std::filesystem::exists(input_path)) {
      std::cerr << "Dataset not found: " << input_path << '\n';
      return 1;
    }

    if (!gpjson::cuda::device_available()) {
      std::cerr << "CUDA device unavailable\n";
      return 1;
    }

    const std::vector<std::string> queries{
        "$.user.lang",
    };

    gpjson::Engine engine;
    const auto start = std::chrono::steady_clock::now();
    const auto result =
        engine.query(input_path.string(), queries, benchmark_engine_options());
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Dataset: " << input_path << '\n';
    std::cout << "Batch query count: " << queries.size() << '\n';
    std::cout << std::fixed << std::setprecision(3)
              << "Query time: " << elapsed_ms << " ms\n";

    size_t total_matches = 0;
    for (const auto &query_result : result.queries()) {
      const size_t query_matches = match_count(query_result);
      total_matches += query_matches;
      std::cout << query_result.query_text()
                << " match_count=" << query_matches << '\n';
    }
    std::cout << "Total matches: " << total_matches << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Query failed: " << error.what() << '\n';
    return 1;
  }
}
