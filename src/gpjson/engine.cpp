#include "gpjson/engine.hpp"

#include "gpjson/error/common.hpp"
#include "gpjson/file/file_reader.hpp"
#include "gpjson/index/index_builder.hpp"
#include "gpjson/query/query_compiler.hpp"
#include "gpjson/query/query_executor.hpp"

#include <memory>

namespace gpjson {

query::MaterializedBatchResult
Engine::query(const std::string &file_path,
              const std::vector<std::string> &queries_src,
              const EngineOptions &options) const {
  query::QueryCompiler query_compiler(options);
  query::QueryExecutor query_executor(options);

  file::FileReader file_reader(file_path);
  file_reader.create_partitions(options.file_partition_size);

  std::unique_ptr<index::IndexBuilder> index_builder;
  switch (options.index_builder_type) {
  case index::IndexBuilderType::UNCOMBINED:
    index_builder =
        std::make_unique<index::UncombinedIndexBuilder>(file_reader);
    break;

  case index::IndexBuilderType::COMBINED:
    index_builder = std::make_unique<index::CombinedIndexBuilder>(file_reader);
    break;

  case index::IndexBuilderType::NO_ESCAPE_QUOTE:
    throw error::common::NotImplementedError(
        "NO_ESCAPE_QUOTE type of index builder is not implemented!");

  default:
    throw error::common::ImplementationError("Undefined index builder type.");
  }

  query::BatchCompiledQuery compiled_queries;
  for (const auto &query_src : queries_src) {
    const query::CompiledQuery compiled_query =
        query_compiler.compile(query_src, options);
    compiled_queries.add(compiled_query);
  }

  query::BatchQueryResult full_results(compiled_queries.size());

  for (const auto &partition_view : file_reader.get_partition_views()) {
    index::BuiltIndices built_indices = index_builder->build(
        partition_view, compiled_queries.get_max_depth(), options);

    query::BatchQueryResult part_result = query_executor.execute_batch(
        compiled_queries, partition_view, built_indices);

    full_results.merge(part_result);
  }

  return full_results.materialize();
}

} // namespace gpjson
