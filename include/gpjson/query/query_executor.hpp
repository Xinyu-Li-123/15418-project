#pragma once

#include "gpjson/file/file.hpp"
#include "gpjson/index/index.hpp"
#include "gpjson/query/query.hpp"

#include <iosfwd>
#include <ostream>
#include <type_traits>

namespace gpjson {
struct EngineOptions;
}

namespace gpjson::query {

enum class QueryExecutorType { ORIG = 0, OPTIMIZED };

inline std::ostream &operator<<(std::ostream &os, QueryExecutorType type) {
  switch (type) {
  case QueryExecutorType::ORIG:
    return os << "ORIG";
  case QueryExecutorType::OPTIMIZED:
    return os << "OPTIMIZED";
  }
  return os << "UNKNOWN_QUERY_EXECUTOR_TYPE("
            << static_cast<std::underlying_type_t<QueryExecutorType>>(type)
            << ")";
}

struct QueryExecutorOptions {
  QueryExecutorType query_executor_type{QueryExecutorType::ORIG};
  int grid_size{512};
  int block_size{1024};
};

inline std::ostream &operator<<(std::ostream &os,
                                const QueryExecutorOptions &options) {
  return os << "QueryExecutorOptions{query_executor_type="
            << options.query_executor_type
            << ", grid_size=" << options.grid_size
            << ", block_size=" << options.block_size << "}";
}

class QueryExecutor {
public:
  explicit QueryExecutor(const gpjson::EngineOptions &options);

  BatchQueryResult
  execute_batch(const BatchCompiledQuery &compiled_queries,
                const file::FilePartition &partition,
                const index::BuiltIndices &built_indices) const;

private:
  QueryExecutorType query_executor_type_{QueryExecutorType::ORIG};
  int grid_size_{512};
  int block_size_{1024};
};
} // namespace gpjson::query
