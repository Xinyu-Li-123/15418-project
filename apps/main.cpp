#include "gpjson/engine.hpp"

#include <iostream>
#include <string>
#include <vector>

int main() {
  gpjson::EngineOptions options{};
  gpjson::Engine engine{};

  const std::string dataset_path = "./datasets/twitter_large_record.json";
  const std::vector<std::string> queries{
      "$.user.lang",
      "$.user.lang[?(@ == \"en\")]",
  };

  auto batch_result = engine.query(dataset_path, queries, options);

  std::cout << "Initialized gpjson-cpp engine\n";
  std::cout << "Dataset: " << dataset_path << '\n';
  std::cout << "Submitted " << queries.size() << " batched queries\n";
  (void)batch_result;

  return 0;
}
