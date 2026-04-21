#include "gpjson/engine.hpp"

#include "gpjson/error/common.hpp"
#include "gpjson/file/file_reader.hpp"
#include "gpjson/index/index_builder.hpp"
#include "gpjson/profiler/profiler.hpp"
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
  profiler::Profiler profiler;
  const profiler::Profiler::SegmentId engine_query =
      profiler.begin("Engine::query");

  query::QueryCompiler query_compiler(options);
  query::QueryExecutor query_executor(options);
  const profiler::Profiler::SegmentId query_compilation =
      profiler.begin("query_compilation");
  const query::BatchCompiledQuery compiled_queries =
      compile_queries(queries_src, options, query_compiler);
  profiler.end(query_compilation);

  file::FileReader file_reader(file_path);
  {

    const auto create_partition_scope =
        profiler.scope("file_reader.create_partitions");
    (void)create_partition_scope;
    file_reader.create_partitions(
        options.index_builder_options.file_partition_size);
  }
  std::unique_ptr<index::IndexBuilder> index_builder =
      create_index_builder(file_reader, options.index_builder_options);

  std::vector<query::MaterializedQueryResult> merged_queries =
      initialize_materialized_query_results(compiled_queries);

  for (auto &partition : file_reader.get_partitions()) {
    const profiler::Profiler::SegmentId partition_total =
        profiler.beginf("partition %zu total", partition.partition_id());

    {
      const auto load_scope = profiler.scopef("partition %zu load_to_device",
                                              partition.partition_id());
      (void)load_scope;
      partition.load_to_device();
    }

    index::BuiltIndices built_indices;
    {
      const auto index_scope = profiler.scopef("partition %zu build_index",
                                               partition.partition_id());
      (void)index_scope;
      built_indices =
          index_builder->build(partition, compiled_queries.get_max_depth(),
                               options.index_builder_options);
    }

    query::BatchQueryResult batch_result(compiled_queries.size());
    {
      const auto execute_scope = profiler.scopef("partition %zu execute_query",
                                                 partition.partition_id());
      (void)execute_scope;
      batch_result = query_executor.execute_batch(compiled_queries, partition,
                                                  built_indices);
    }

    query::MaterializedBatchResult partition_result;
    {
      const auto materialize_scope = profiler.scopef(
          "partition %zu materialize_results", partition.partition_id());
      (void)materialize_scope;
      partition_result = batch_result.materialize();
    }

    append_partition_results(merged_queries, partition_result);
    profiler.end(partition_total);
  }

  query::MaterializedBatchResult merged_result;
  for (auto &query_result : merged_queries) {
    merged_result.add_query_result(std::move(query_result));
  }

  profiler.end(engine_query);
  profiler.print();
  return merged_result;
}

} // namespace gpjson
