#pragma once

#include "gpjson/query/kernels/orig.hpp"

namespace gpjson::query::kernels::optimized {

using QueryExecutorOptions = orig::QueryExecutorOptions;

inline constexpr int kMaxQueryLevels = orig::kMaxQueryLevels;
inline constexpr std::size_t kMaxQueryIrBytes = orig::kMaxQueryIrBytes;

BatchQueryResult execute_batch(const BatchCompiledQuery &compiled_queries,
                               const file::FilePartition &partition,
                               const index::BuiltIndices &built_indices,
                               const QueryExecutorOptions &options);

} // namespace gpjson::query::kernels::optimized
