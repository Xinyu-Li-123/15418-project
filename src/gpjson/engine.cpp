#include <string>
#include <vector>

namespace gpjson {
query::MaterializedBatchQueryResult
Engine::query(std::string file_path, std::vector<std::string> queries_src,
              EngineOptions options) {
  query::QueryCompiler query_compiler = query::QueryCompiler(options);
  query::QueryExecutor query_executor = query::QueryExecutor(options);

  file::FileReader file_reader = file::FileReader(file_path);
  file_reader.create_partitions(options.file_partition_size);

  // This is an abstract classes, so that we can experiment with
  // different index building methods
  index::IndexBuilder index_builder;
  switch (options.index_builder_type) {
  case UNCOMBINED:
    index_builder = index::UncombinedIndexBuilder(file_reader);
    break;
  case COMBINED:
    index_builder = index::CombinedIndexBuilder(file_reader);
    break;
  case NO_ESCAPE_QUOTE:
    throw error::common::NotImplementedError(
        "NO_ESCAPE_QUOTE type of index builder is not implemented!");
    break;
  default:
    throw error::common::ImplementationError("Undefined index builder type.")
  }

  // Compile all queries once
  query::BatchCompiledQuery compiled_queries{};
  for (const auto query_src : queries_src) {
    const CompiledQuery compiled_query =
        query_compiler.compile(query_src, options);
    compiled_queries.add(compiled_query);
  }

  // Execute all queries on one file partition at a time
  query::BatchQueryResult full_results(queries.length);
  // PartitionView provide shared read access to partition. It contains metadata
  // about partition, and a
  for (const auto partition_view : file::FileReader.get_partition_views()) {
    // e.g. built_indices.get_newline_index()
    // Tmp indices like escape carry index are created and dropped within
    // index_builder.build()
    index::BuiltIndices built_indices = index_builder.build(
        partition_view, compiled_queries.get_max_depth(), options);
    // Need file_reader as well because need to read file to materialize results
    query::BatchQueryResult part_result = query_executor.executeBatch(
        compiled_queries, partition_view, built_indices);
    // Preserve line order
    full_results.merge(part_result);
  }

  // Up until now, each QueryResult is only an array of offsets into the json
  // file. We need to materialize it by reading the file to obtain actual query
  // result
  return full_results.materialize();
}
} // namespace gpjson
