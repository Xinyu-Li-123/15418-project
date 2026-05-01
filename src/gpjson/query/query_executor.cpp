#include "gpjson/query/query_executor.hpp"

#include "gpjson/engine.hpp"
#include "gpjson/error/common.hpp"
#include "gpjson/query/kernels/orig.hpp"
#include "gpjson/query/kernels/optimized.hpp"

namespace gpjson::query {

QueryExecutor::QueryExecutor(const gpjson::EngineOptions &options)
    : query_executor_type_(
          options.query_executor_options.query_executor_type),
      grid_size_(options.query_executor_options.grid_size),
      block_size_(options.query_executor_options.block_size) {}

BatchQueryResult
QueryExecutor::execute_batch(const BatchCompiledQuery &compiled_queries,
                             const file::FilePartition &partition,
                             const index::BuiltIndices &built_indices) const {
  const kernels::orig::QueryExecutorOptions kernel_options{grid_size_,
                                                          block_size_};
  switch (query_executor_type_) {
  case QueryExecutorType::ORIG:
    return kernels::orig::execute_batch(compiled_queries, partition,
                                        built_indices, kernel_options);

  case QueryExecutorType::OPTIMIZED:
    return kernels::optimized::execute_batch(compiled_queries, partition,
                                             built_indices, kernel_options);
  }

  throw error::common::ImplementationError("Undefined query executor type.");
}

} // namespace gpjson::query
