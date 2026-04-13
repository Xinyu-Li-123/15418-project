#include "gpjson/engine.hpp"
#include "gpjson/file/error.hpp"
#include "gpjson/file/file_reader.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using gpjson::file::FileReader;
using gpjson::file::PartitionView;
using gpjson::query::MaterializedBatchResult;
using gpjson::query::MaterializedLineResult;
using gpjson::query::MaterializedQueryResult;

std::string bytes_to_string(std::span<const std::byte> bytes) {
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::filesystem::path write_temp_file(const std::string &filename,
                                      const std::string &contents) {
  const auto path = std::filesystem::temp_directory_path() / filename;
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("Failed to open temporary test file: " +
                             path.string());
  }
  output << contents;
  output.close();
  return path;
}

void verify_partition(const PartitionView &partition,
                      size_t expected_id,
                      size_t expected_start,
                      size_t expected_end,
                      const std::string &expected_contents) {
  require(partition.get_partition_id() == expected_id,
          "Unexpected partition id.");
  require(partition.get_global_start_offset() == expected_start,
          "Unexpected partition start offset.");
  require(partition.get_global_end_offset() == expected_end,
          "Unexpected partition end offset.");
  require(partition.size_bytes() == expected_contents.size(),
          "Unexpected partition byte size.");
  require(bytes_to_string(partition.get_data()) == expected_contents,
          "Unexpected partition contents.");
}

void test_single_partition_reads_whole_file() {
  const std::string contents = "{\"id\":1}\n{\"id\":2}\n";
  const auto path = write_temp_file("gpjson_file_reader_single_partition.json",
                                    contents);

  FileReader reader(path.string());
  reader.create_partitions(0);

  const auto &metadata = reader.get_metadata();
  require(metadata.file_path == path.string(), "Unexpected file path metadata.");
  require(metadata.file_size_bytes == contents.size(),
          "Unexpected file size metadata.");
  require(metadata.num_partitions == 1, "Expected one partition.");

  const auto &partitions = reader.get_partition_views();
  require(partitions.size() == 1, "Expected one partition view.");
  verify_partition(partitions.front(), 0, 0, contents.size(), contents);

  std::filesystem::remove(path);
}

void test_batched_partitions_follow_newlines() {
  const std::string contents = "alpha\nbeta\ngamma";
  const auto path =
      write_temp_file("gpjson_file_reader_batched_partitions.json", contents);

  FileReader reader(path.string());
  reader.create_partitions(7);

  const auto &metadata = reader.get_metadata();
  require(metadata.file_size_bytes == contents.size(),
          "Unexpected batched file size metadata.");
  require(metadata.num_partitions == 3, "Expected three partitions.");

  const auto &partitions = reader.get_partition_views();
  require(partitions.size() == 3, "Expected three partition views.");
  verify_partition(partitions[0], 0, 0, 5, "alpha");
  verify_partition(partitions[1], 1, 6, 10, "beta");
  verify_partition(partitions[2], 2, 11, contents.size(), "gamma");

  std::filesystem::remove(path);
}

void test_partitioning_fails_without_newline_before_boundary() {
  const auto path =
      write_temp_file("gpjson_file_reader_partition_error.json", "abcdefghij");

  FileReader reader(path.string());
  bool threw = false;
  try {
    reader.create_partitions(4);
  } catch (const gpjson::error::file::PartitionError &) {
    threw = true;
  }

  require(threw, "Expected partitioning to fail without a newline boundary.");
  std::filesystem::remove(path);
}

std::vector<std::vector<std::string>>
materialized_lines_to_strings(const MaterializedQueryResult &query_result) {
  std::vector<std::vector<std::string>> lines;
  for (const MaterializedLineResult &line : query_result.lines()) {
    std::vector<std::string> values;
    for (const auto &value : line.values()) {
      values.push_back(value.json_text());
    }
    lines.push_back(std::move(values));
  }
  return lines;
}

void require_query_lines(const MaterializedBatchResult &batch_result,
                         size_t query_index,
                         const std::string &expected_query_text,
                         const std::vector<std::vector<std::string>>
                             &expected_lines) {
  require(query_index < batch_result.queries().size(),
          "Missing expected query result.");
  const auto &query_result = batch_result.queries()[query_index];
  require(query_result.query_text() == expected_query_text,
          "Unexpected materialized query text.");
  require(materialized_lines_to_strings(query_result) == expected_lines,
          "Unexpected materialized query results.");
}

void test_query_engine_executes_supported_jsonpath() {
  const std::string contents =
      "{\"user\":{\"lang\":\"en\",\"name\":\"Alice\"},\"arr\":[10,20,30]}\n"
      "{\"user\":{\"lang\":\"fr\",\"name\":\"Bob\"},\"arr\":[40,50]}\n";
  const auto path =
      write_temp_file("gpjson_query_executor_supported_queries.json", contents);

  gpjson::Engine engine;
  gpjson::EngineOptions options{};

  const MaterializedBatchResult batch_result = engine.query(
      path.string(),
      {"$.user.lang", "$.user.lang[?(@ == \"en\")]", "$.arr[0:2]"},
      options);

  require(batch_result.queries().size() == 3,
          "Expected three materialized query results.");
  require_query_lines(batch_result, 0, "$.user.lang",
                      {{"\"en\""}, {"\"fr\""}});
  require_query_lines(batch_result, 1, "$.user.lang[?(@ == \"en\")]",
                      {{"\"en\""}, {}});
  require_query_lines(batch_result, 2, "$.arr[0:2]",
                      {{"10", "20"}, {"40", "50"}});

  std::filesystem::remove(path);
}

void test_query_engine_merges_partitioned_results() {
  const std::string contents =
      "{\"user\":{\"lang\":\"en\"},\"arr\":[1,2,3]}\n"
      "{\"user\":{\"lang\":\"fr\"},\"arr\":[4,5,6]}\n"
      "{\"user\":{\"lang\":\"it\"},\"arr\":[7,8,9]}\n";
  const auto path =
      write_temp_file("gpjson_query_executor_partitioned_queries.json",
                      contents);

  gpjson::Engine engine;
  gpjson::EngineOptions options{};
  options.file_partition_size = 40;

  const MaterializedBatchResult batch_result =
      engine.query(path.string(), {"$.user.lang", "$.arr[-1]"}, options);

  require_query_lines(batch_result, 0, "$.user.lang",
                      {{"\"en\""}, {"\"fr\""}, {"\"it\""}});
  require_query_lines(batch_result, 1, "$.arr[-1]",
                      {{"3"}, {"6"}, {"9"}});

  std::filesystem::remove(path);
}

} // namespace

int main() {
  test_single_partition_reads_whole_file();
  test_batched_partitions_follow_newlines();
  test_partitioning_fails_without_newline_before_boundary();
  test_query_engine_executes_supported_jsonpath();
  test_query_engine_merges_partitioned_results();
  return 0;
}
