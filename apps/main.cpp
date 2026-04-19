#include "gpjson/engine.hpp"
#include "gpjson/index/index_builder.hpp"
#include "gpjson/log/log.hpp"

#include <iostream>
#include <string>
#include <vector>

int main() {

  LogInfo("The app is currently just a placeholder.");

  // Same parameters as the default ones in GpJSON Java codebase
  gpjson::EngineOptions options{
      .index_builder_options = gpjson::index::IndexBuilderOptions{
          .index_builder_type = gpjson::index::IndexBuilderType::COMBINED,
          .grid_size = 1024 * 16,
          .block_size = 1024,
          .reduction_grid_size = 32,
          .reduction_block_size = 32,
      }};

  gpjson::Engine engine{};

  // const std::string dataset_path =
  //     "/afs/andrew.cmu.edu/usr13/xinyuli4/private/15418/project/codebase/"
  //     "dataset/twitter_large_record.json";
  const std::string dataset_path =
      "/afs/andrew.cmu.edu/usr13/xinyuli4/private/15418/project/codebase/"
      "dataset/twitter_small_records_remove.json";
  const std::vector<std::string> queries{
      "$.user.lang",
  };

  auto batch_result = engine.query(dataset_path, queries, options);

  std::cout << "Initialized gpjson-cpp engine\n";
  std::cout << "Dataset: " << dataset_path << '\n';
  std::cout << "Submitted " << queries.size() << " batched queries\n";
  // // std::cout << "Batch result stats: " << batch_result << '\n';
  // for (const auto &result : batch_result.queries()) {
  //   std::cout << "query: " << result.query_text() << "\n";
  //   for (const auto &line : result.lines()) {
  //     std::cout << "\t";
  //     for (const auto &value : line.values()) {
  //       std::cout << value.json_text() << ", ";
  //     }
  //   }
  //   std::cout << "\n";
  // }

  return 0;
}
