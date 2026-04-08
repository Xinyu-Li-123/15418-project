#pragma once

#include "gpjson/query_compiler.hpp"
#include "gpjson/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace gpjson {

// Stores the byte range of one query result inside the original file.
struct QueryValueRange {
  // Inclusive start byte offset of the matched value.
  std::int32_t value_start = -1;
  // Exclusive end byte offset of the matched value.
  std::int32_t value_end = -1;
};

// Stores all values returned for one input line.
struct QueryLineResult {
  // Zero-based line number in the input file.
  std::uint64_t line_index = 0;
  // All value ranges returned for this line.
  std::vector<QueryValueRange> values{};
};

// Stores the full result of executing one compiled query.
struct QueryResult {
  // Original query string.
  std::string query{};
  // Result rows, one entry per input line.
  std::vector<QueryLineResult> lines{};
};

// Stores the results of executing multiple compiled queries.
struct QueryBatchResult {
  // One result object per compiled query.
  std::vector<QueryResult> queries{};
};

// Executes compiled JSONPath queries on a loaded file and its indexes.
class QueryExecutor {
 public:
  // Creates an executor that uses the project's fixed execution policy.
  QueryExecutor() = default;

  // Executes a batch of compiled queries and returns all results.
  QueryBatchResult execute(const LoadedFile &loaded_file,
                           const BuiltIndex &built_index,
                           const std::vector<JSONPathQuery> &queries);

  // Releases any resources owned by a batch query result.
  void release(QueryBatchResult &result);
};

}  // namespace gpjson
