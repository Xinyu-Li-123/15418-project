#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gpjson {

// Stores one query after JSONPath compilation.
struct JSONPathQuery {
  // Original query string.
  std::string source{};
  // Compiled IR consumed by the GPU query kernel.
  std::vector<std::uint8_t> ir{};
  // Maximum nesting depth required by this query.
  std::uint32_t max_depth = 0;
  // Number of values produced per line by this query.
  std::uint32_t num_results = 0;
  // Whether this query is supported by the GPU path.
  bool is_supported = true;
};

// Compiles JSONPath strings into the IR used by the query executor.
class QueryCompiler {
 public:
  // Creates a compiler that uses the project's fixed compilation policy.
  QueryCompiler() = default;

  // Compiles a batch of JSONPath query strings.
  std::vector<JSONPathQuery> compile(
      const std::vector<std::string> &queries) ;

  // Returns the maximum nesting depth required by a batch of queries.
  std::uint32_t get_max_depth(
      const std::vector<JSONPathQuery> &queries) ;
};

}  // namespace gpjson
