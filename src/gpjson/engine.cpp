#include "gpjson/engine.hpp"

#include "gpjson/error/common.hpp"
#include "gpjson/file/file_reader.hpp"
#include "gpjson/index/index_builder.hpp"
#include "gpjson/profiler/profiler.hpp"
#include "gpjson/query/query_compiler.hpp"
#include "gpjson/query/query_executor.hpp"

#include <iostream>
#include <memory>
#include <vector>

namespace gpjson {
namespace {

std::unique_ptr<file::FileReader>
create_file_reader(const std::string &file_path,
                   const file::FileReaderOptions &options) {
  switch (options.file_reader_type) {
  case file::FileReaderType::COPIED:
    return std::make_unique<file::CopiedFileReader>(file_path);

  case file::FileReaderType::MMAP:
    return std::make_unique<file::MmapFileReader>(file_path);

  default:
    throw error::common::ImplementationError("Undefined file reader type.");
  }
}

std::unique_ptr<index::IndexBuilder>
create_index_builder(const file::FileReader &file_reader,
                     const index::IndexBuilderOptions &options) {
  switch (options.index_builder_type) {
  case index::IndexBuilderType::UNCOMBINED:
    return std::make_unique<index::UncombinedIndexBuilder>(file_reader);

  case index::IndexBuilderType::COMBINED:
    return std::make_unique<index::CombinedIndexBuilder>(file_reader);

  case index::IndexBuilderType::FUSED:
    return std::make_unique<index::FusedIndexBuilder>(file_reader);

  case index::IndexBuilderType::SHAREMEM:
    return std::make_unique<index::SharememIndexBuilder>(file_reader);

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
  std::cout << "Engine::query options: " << options << '\n';
  if (queries_src.empty()) {
    return {};
  }
  profiler::Profiler profiler("Engine::query profiler");
  const profiler::Profiler::SegmentId engine_query_timer =
      profiler.begin("Engine::query");

  query::QueryCompiler query_compiler(options);
  query::QueryExecutor query_executor(options);
  const profiler::Profiler::SegmentId query_comp_timer =
      profiler.begin("query_compilation");
  const query::BatchCompiledQuery compiled_queries =
      compile_queries(queries_src, options, query_compiler);
  profiler.end(query_comp_timer);

  std::unique_ptr<file::FileReader> file_reader =
      create_file_reader(file_path, options.file_reader_options);
  const profiler::Profiler::SegmentId create_partitions_timer =
      profiler.begin("file_reader.create_partitions");
  file_reader->create_partitions(
      options.index_builder_options.file_partition_size);
  profiler.end(create_partitions_timer);

  std::unique_ptr<index::IndexBuilder> index_builder =
      create_index_builder(*file_reader, options.index_builder_options);

  std::vector<query::MaterializedQueryResult> merged_queries =
      initialize_materialized_query_results(compiled_queries);

  for (auto &partition : file_reader->get_partitions()) {
    const profiler::Profiler::SegmentId partition_total_timer =
        profiler.beginf("partition %zu total", partition.partition_id());
    const profiler::Profiler::SegmentId load_to_device_timer = profiler.beginf(
        "  partition %zu load_to_device", partition.partition_id());
    partition.load_to_device();
    profiler.end(load_to_device_timer);

    const profiler::Profiler::SegmentId build_index_timer = profiler.beginf(
        "  partition %zu build_index", partition.partition_id());
    index::BuiltIndices built_indices;
    built_indices =
        index_builder->build(partition, compiled_queries.get_max_depth(),
                             options.index_builder_options);
    profiler.end(build_index_timer);

    const profiler::Profiler::SegmentId execute_query_timer = profiler.beginf(
        "  partition %zu execute_query", partition.partition_id());
    query::BatchQueryResult batch_result(compiled_queries.size());
    batch_result = query_executor.execute_batch(compiled_queries, partition,
                                                built_indices);
    profiler.end(execute_query_timer);

    const profiler::Profiler::SegmentId materialize_results_timer =
        profiler.beginf("  partition %zu materialize_results",
                        partition.partition_id());
    query::MaterializedBatchResult partition_result;
    partition_result = batch_result.materialize();
    profiler.end(materialize_results_timer);

    append_partition_results(merged_queries, partition_result);
    profiler.end(partition_total_timer);
  }

  query::MaterializedBatchResult merged_result;
  for (auto &query_result : merged_queries) {
    merged_result.add_query_result(std::move(query_result));
  }

  profiler.end(engine_query_timer);
  return merged_result;
}

} // namespace gpjson
