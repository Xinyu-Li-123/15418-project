#include "gpjson/query/query_executor.hpp"

namespace gpjson::query {

QueryExecutor::QueryExecutor(const gpjson::EngineOptions &options) {
  (void)options;
}

BatchQueryResult
QueryExecutor::execute_batch(const BatchCompiledQuery &compiled_queries,
                             const file::PartitionView &partition_view,
                             const index::BuiltIndices &built_indices) const {
  (void)partition_view;
  (void)built_indices;
  return BatchQueryResult(compiled_queries.size());
}

} // namespace gpjson::query
