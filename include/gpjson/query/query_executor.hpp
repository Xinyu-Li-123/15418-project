#pragma once

#include "gpjson/file/file.hpp"
#include "gpjson/index/index.hpp"
#include "gpjson/query/query.hpp"

#include <iosfwd>
#include <ostream>

namespace gpjson {
struct EngineOptions;
}

namespace gpjson::query {

struct QueryExecutorOptions {
  int grid_size{512};
  int block_size{1024};
};

inline std::ostream &operator<<(std::ostream &os,
                                const QueryExecutorOptions &options) {
  return os << "QueryExecutorOptions{grid_size=" << options.grid_size
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
  int grid_size_{512};
  int block_size_{1024};
};
} // namespace gpjson::query
