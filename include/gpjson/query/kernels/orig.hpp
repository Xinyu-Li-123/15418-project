#pragma once

#include "gpjson/file/file.hpp"
#include "gpjson/index/index.hpp"
#include "gpjson/query/query.hpp"

#include <cstddef>

namespace gpjson::query::kernels::orig {

inline constexpr int kMaxQueryLevels = 16;
inline constexpr std::size_t kMaxQueryIrBytes = 4096;

struct QueryExecutorOptions {
  int grid_size{512};
  int block_size{1024};
};

BatchQueryResult execute_batch(const BatchCompiledQuery &compiled_queries,
                               const file::FilePartition &partition,
                               const index::BuiltIndices &built_indices,
                               const QueryExecutorOptions &options);

} // namespace gpjson::query::kernels::orig
