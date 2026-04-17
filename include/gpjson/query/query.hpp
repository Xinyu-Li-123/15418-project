#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace gpjson::query {

enum class QueryOpcode : unsigned char {
  END = 0,
  STORE_RESULT = 1,
  MOVE_UP = 2,
  MOVE_DOWN = 3,
  MOVE_TO_KEY = 4,
  MOVE_TO_INDEX = 5,
  MOVE_TO_INDEX_REVERSE = 6,
  MARK_POSITION = 7,
  RESET_POSITION = 8,
  EXPRESSION_STRING_EQUALS = 9,
};

class CompiledQuery {
public:
  CompiledQuery() = default;
  CompiledQuery(std::string query_text,
                std::vector<std::byte> ir_bytes,
                size_t max_depth,
                size_t num_result_slots);

  const std::string &query_text() const;
  const std::vector<std::byte> &ir_bytes() const;
  size_t max_depth() const;
  size_t num_result_slots() const;

private:
  std::string query_text_;
  std::vector<std::byte> ir_bytes_;
  size_t max_depth_{1};
  size_t num_result_slots_{1};
};

class BatchCompiledQuery {
public:
  void add(CompiledQuery query);
  size_t size() const;
  size_t get_max_depth() const;
  const std::vector<CompiledQuery> &queries() const;

private:
  std::vector<CompiledQuery> queries_;
  size_t max_depth_{1};
};

struct QueryOffset {
  std::optional<size_t> start;
  std::optional<size_t> end;
  std::optional<std::string> json_text;
};

class LineQueryResult {
public:
  const std::vector<QueryOffset> &offsets() const;
  void add_offset(QueryOffset offset);

private:
  std::vector<QueryOffset> offsets_;
};

class QueryResult {
public:
  void add_line_result(LineQueryResult line_result);
  const std::vector<LineQueryResult> &lines() const;

private:
  std::vector<LineQueryResult> lines_;
};

class MaterializedBatchResult;

class BatchQueryResult {
public:
  explicit BatchQueryResult(size_t num_queries);

  void add_line_result(size_t query_index, LineQueryResult line_result);
  void merge(const BatchQueryResult &other);
  void set_query_text(size_t query_index, std::string query_text);
  MaterializedBatchResult materialize() const;
  size_t num_queries() const;
  const std::vector<QueryResult> &queries() const;

private:
  std::vector<QueryResult> queries_;
  std::vector<std::string> query_texts_;
};

class MaterializedValue {
public:
  explicit MaterializedValue(std::string json_text);
  const std::string &json_text() const;

private:
  std::string json_text_;
};

class MaterializedLineResult {
public:
  const std::vector<MaterializedValue> &values() const;
  void add_value(MaterializedValue value);

private:
  std::vector<MaterializedValue> values_;
};

class MaterializedQueryResult {
public:
  explicit MaterializedQueryResult(std::string query_text);

  const std::string &query_text() const;
  const std::vector<MaterializedLineResult> &lines() const;
  void add_line_result(MaterializedLineResult line);

private:
  std::string query_text_;
  std::vector<MaterializedLineResult> lines_;
};

class MaterializedBatchResult {
public:
  const std::vector<MaterializedQueryResult> &queries() const;
  void add_query_result(MaterializedQueryResult result);

private:
  std::vector<MaterializedQueryResult> queries_;
};

class QueryOptions {};
} // namespace gpjson::query
