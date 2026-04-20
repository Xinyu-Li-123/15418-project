#include "gpjson/engine.hpp"

#include "gpjson/error/common.hpp"
#include "gpjson/file/file_reader.hpp"
#include "gpjson/index/index_builder.hpp"
#include "gpjson/query/query_compiler.hpp"
#include "gpjson/query/query_executor.hpp"

#include <memory>
#include <vector>

namespace gpjson {
namespace {

std::unique_ptr<index::IndexBuilder>
create_index_builder(const file::FileReader &file_reader,
                     const index::IndexBuilderOptions &options) {
  switch (options.index_builder_type) {
  case index::IndexBuilderType::UNCOMBINED:
    return std::make_unique<index::UncombinedIndexBuilder>(file_reader);

  case index::IndexBuilderType::COMBINED:
    return std::make_unique<index::CombinedIndexBuilder>(file_reader);

  case index::IndexBuilderType::NO_ESCAPE_QUOTE:
    throw error::common::NotImplementedError(
        "NO_ESCAPE_QUOTE type of index builder is not implemented!");

  default:
    throw error::common::ImplementationError("Undefined index builder type.");
  }
}

query::BatchCompiledQuery
compile_queries(const std::vector<std::string> &queries_src,
                const EngineOptions &options,
                const query::QueryCompiler &query_compiler) {
  query::BatchCompiledQuery compiled_queries;
  for (const auto &query_src : queries_src) {
    compiled_queries.add(query_compiler.compile(query_src, options));
  }
  return compiled_queries;
}

std::vector<query::MaterializedQueryResult>
initialize_materialized_query_results(
    const query::BatchCompiledQuery &compiled_queries) {
  std::vector<query::MaterializedQueryResult> merged_queries;
  merged_queries.reserve(compiled_queries.size());

  for (const auto &compiled_query : compiled_queries.queries()) {
    merged_queries.emplace_back(compiled_query.query_text());
  }

  return merged_queries;
}

void append_partition_results(
    std::vector<query::MaterializedQueryResult> &merged_queries,
    const query::MaterializedBatchResult &partition_result) {
  if (merged_queries.size() != partition_result.queries().size()) {
    throw error::common::ImplementationError(
        "Partition query results do not match the compiled query batch.");
  }

  for (size_t query_index = 0; query_index < merged_queries.size();
       ++query_index) {
    const auto &partition_lines =
        partition_result.queries()[query_index].lines();
    for (const auto &partition_line : partition_lines) {
      merged_queries[query_index].add_line_result(partition_line);
    }
  }
}

} // namespace

query::MaterializedBatchResult
Engine::query(const std::string &file_path,
              const std::vector<std::string> &queries_src,
              const EngineOptions &options) const {
  if (queries_src.empty()) {
    return {};
  }

  query::QueryCompiler query_compiler(options);
  query::QueryExecutor query_executor(options);
  const query::BatchCompiledQuery compiled_queries =
      compile_queries(queries_src, options, query_compiler);

  file::FileReader file_reader(file_path);
  file_reader.create_partitions(
      options.index_builder_options.file_partition_size);
  std::unique_ptr<index::IndexBuilder> index_builder =
      create_index_builder(file_reader, options.index_builder_options);

  std::vector<query::MaterializedQueryResult> merged_queries =
      initialize_materialized_query_results(compiled_queries);

  for (auto &partition_view : file_reader.get_partition_views()) {
    partition_view.load_to_device();
    index::BuiltIndices built_indices =
        index_builder->build(partition_view, compiled_queries.get_max_depth(),
                             options.index_builder_options);
    const query::MaterializedBatchResult partition_result =
        query_executor
            .execute_batch(compiled_queries, partition_view, built_indices)
            .materialize();
    append_partition_results(merged_queries, partition_result);
  }

  query::MaterializedBatchResult merged_result;
  for (auto &query_result : merged_queries) {
    merged_result.add_query_result(std::move(query_result));
  }

  return merged_result;
}

} // namespace gpjson
