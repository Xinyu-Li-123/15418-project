#pragma once

#include "gpjson/file/file.hpp"
#include "gpjson/index/index.hpp"
#include "gpjson/query/query.hpp"

namespace gpjson {
struct EngineOptions;
}

namespace gpjson::query {

class QueryExecutor {
public:
  explicit QueryExecutor(const gpjson::EngineOptions &options);

  BatchQueryResult
  execute_batch(const BatchCompiledQuery &compiled_queries,
                const file::PartitionView &partition_view,
                const index::BuiltIndices &built_indices) const;

private:
  int grid_size_{0};
  int block_size_{256};
};
} // namespace gpjson::query
