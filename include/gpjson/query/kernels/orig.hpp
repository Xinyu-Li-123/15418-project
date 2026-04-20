#pragma once

#include "gpjson/file/file.hpp"
#include "gpjson/index/index.hpp"
#include "gpjson/query/query.hpp"

namespace gpjson::query::kernels::orig {

inline constexpr int kMaxQueryLevels = 16;

struct QueryExecutorOptions {
  int grid_size{0};
  int block_size{256};
};

BatchQueryResult execute_batch(const BatchCompiledQuery &compiled_queries,
                               const file::FilePartition &partition,
                               const index::BuiltIndices &built_indices,
                               const QueryExecutorOptions &options);

} // namespace gpjson::query::kernels::orig
