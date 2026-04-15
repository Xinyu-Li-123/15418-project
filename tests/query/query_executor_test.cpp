#include "gpjson/cuda/cuda.hpp"
#include "gpjson/engine.hpp"
#include "gpjson/file/file.hpp"
#include "gpjson/index/index.hpp"
#include "gpjson/query/query_compiler.hpp"
#include "gpjson/query/query_executor.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using gpjson::cuda::DeviceArray;
using gpjson::file::PartitionView;
using gpjson::index::BuiltIndices;
using gpjson::index::LeveledBitmapIndex;
using gpjson::index::NewlineIndex;
using gpjson::index::StringIndex;
using gpjson::query::BatchCompiledQuery;
using gpjson::query::MaterializedBatchResult;
using gpjson::query::QueryCompiler;
using gpjson::query::QueryExecutor;

void skip_if_cuda_unavailable() {
  if (!gpjson::cuda::device_available()) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
}

void set_bit(std::vector<long> &bitmap, size_t position) {
  const size_t word_index = position / 64U;
  const size_t bit_index = position % 64U;
  bitmap[word_index] |= static_cast<long>(1UL << bit_index);
}

BuiltIndices build_single_line_indices(const std::string &json_text) {
  // Until the real builder is wired in, construct the executor inputs directly.
  const size_t level_size = (json_text.size() + 63U) / 64U;

  std::vector<long> newline_offsets{0};
  std::vector<long> string_index(level_size, 0);
  std::vector<std::vector<size_t>> positions_by_level;
  std::vector<char> container_stack;

  bool in_string = false;
  bool escaped = false;
  for (size_t position = 0; position < json_text.size(); ++position) {
    const char current = json_text[position];

    if (in_string) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (current == '\\') {
        escaped = true;
        continue;
      }
      if (current == '"') {
        // The kernel scans backward from ':' and expects this index to land on
        // the last character before the closing quote, not the quote itself.
        if (position > 0) {
          set_bit(string_index, position - 1U);
        }
        in_string = false;
      }
      continue;
    }

    if (current == '"') {
      in_string = true;
      continue;
    }

    if (current == '{' || current == '[') {
      container_stack.push_back(current);
      if (positions_by_level.size() < container_stack.size()) {
        positions_by_level.resize(container_stack.size());
      }
      continue;
    }

    if (container_stack.empty()) {
      continue;
    }

    if (current == ':' || current == ',') {
      positions_by_level[container_stack.size() - 1U].push_back(position);
      continue;
    }

    if (current == '}' || current == ']') {
      positions_by_level[container_stack.size() - 1U].push_back(position);
      container_stack.pop_back();
    }
  }

  std::vector<long> leveled_bitmap_index(level_size * positions_by_level.size(),
                                         0);
  for (size_t level = 0; level < positions_by_level.size(); ++level) {
    for (size_t position : positions_by_level[level]) {
      set_bit(leveled_bitmap_index, level * 64U * level_size + position);
    }
  }

  DeviceArray newline_device(newline_offsets.size() * sizeof(long));
  newline_device.copy_from_host(newline_offsets.data(),
                                newline_offsets.size() * sizeof(long));

  DeviceArray string_device(string_index.size() * sizeof(long));
  string_device.copy_from_host(string_index.data(),
                               string_index.size() * sizeof(long));

  DeviceArray bitmap_device(leveled_bitmap_index.size() * sizeof(long));
  bitmap_device.copy_from_host(leveled_bitmap_index.data(),
                               leveled_bitmap_index.size() * sizeof(long));

  return BuiltIndices(
      NewlineIndex(std::move(newline_device),
                   static_cast<int>(newline_offsets.size())),
      StringIndex(std::move(string_device)),
      LeveledBitmapIndex(std::move(bitmap_device),
                         static_cast<int>(positions_by_level.size())));
}

BatchCompiledQuery compile_queries(const gpjson::EngineOptions &options) {
  QueryCompiler compiler(options);

  BatchCompiledQuery compiled_queries;
  compiled_queries.add(compiler.compile("$.expensive", options));
  compiled_queries.add(compiler.compile("$.store.book[1].author", options));
  return compiled_queries;
}

void print_materialized_result(const MaterializedBatchResult &materialized) {
  std::cout << "Materialized batch result\n";
  for (const auto &query : materialized.queries()) {
    std::cout << "  Query: " << query.query_text() << '\n';
    for (size_t line_index = 0; line_index < query.lines().size();
         ++line_index) {
      const auto &line = query.lines()[line_index];
      std::cout << "    Line " << line_index << ":";
      if (line.values().empty()) {
        std::cout << " <empty>";
      }
      for (const auto &value : line.values()) {
        std::cout << ' ' << value.json_text();
      }
      std::cout << '\n';
    }
  }
}

} // namespace

TEST(QueryExecutorTest, ExecutesCompiledQueriesWithHandcraftedIndices) {
  skip_if_cuda_unavailable();

  // Copied from /home/moratoryvan/gpjson/test/test.json.
  const std::string json_text =
      R"({"store":{"book":[{"category":"reference","author":"Nigel Rees","title":"Sayings of the Century","price":"8.95"},{"category":"fiction","author":"Herman Melville","title":"Moby Dick","isbn":"0-553-21311-3","price":"8.99"},{"category":"fiction","author":"J.R.R. Tolkien","title":"The Lord of the Rings","isbn":"0-395-19395-8","price":"22.99"}],"bicycle":{"color":"red","price":"19.95"}},"expensive":"10"})";

  const PartitionView partition_view(0, 0, json_text.size(), json_text.data());
  BuiltIndices built_indices = build_single_line_indices(json_text);

  gpjson::EngineOptions options{};
  const BatchCompiledQuery compiled_queries = compile_queries(options);
  QueryExecutor executor(options);

  const auto batch_result =
      executor.execute_batch(compiled_queries, partition_view, built_indices);
  const MaterializedBatchResult materialized = batch_result.materialize();
  print_materialized_result(materialized);

  ASSERT_EQ(batch_result.num_queries(), 2U);
  ASSERT_EQ(batch_result.queries()[0].lines().size(), 1U);
  ASSERT_EQ(batch_result.queries()[1].lines().size(), 1U);

  ASSERT_EQ(materialized.queries().size(), 2U);
  ASSERT_EQ(materialized.queries()[0].query_text(), "$.expensive");
  ASSERT_EQ(materialized.queries()[0].lines().size(), 1U);
  ASSERT_EQ(materialized.queries()[0].lines()[0].values().size(), 1U);
  EXPECT_EQ(materialized.queries()[0].lines()[0].values()[0].json_text(),
            "\"10\"");

  ASSERT_EQ(materialized.queries()[1].query_text(), "$.store.book[1].author");
  ASSERT_EQ(materialized.queries()[1].lines().size(), 1U);
  ASSERT_EQ(materialized.queries()[1].lines()[0].values().size(), 1U);
  EXPECT_EQ(materialized.queries()[1].lines()[0].values()[0].json_text(),
            "\"Herman Melville\"");
}
