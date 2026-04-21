#include "gpjson/query/query_executor.hpp"

#include "gpjson/engine.hpp"
#include "gpjson/query/kernels/orig.hpp"

namespace gpjson::query {

QueryExecutor::QueryExecutor(const gpjson::EngineOptions &options)
    : grid_size_(options.query_executor_options.grid_size),
      block_size_(options.query_executor_options.block_size) {}

BatchQueryResult
QueryExecutor::execute_batch(const BatchCompiledQuery &compiled_queries,
                             const file::FilePartition &partition,
                             const index::BuiltIndices &built_indices) const {
  return kernels::orig::execute_batch(
      compiled_queries, partition, built_indices,
      kernels::orig::QueryExecutorOptions{grid_size_, block_size_});
}

} // namespace gpjson::query
