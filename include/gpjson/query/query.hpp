#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace gpjson::query {

class CompiledQuery {
public:
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

  void merge(const BatchQueryResult &other);
  MaterializedBatchResult materialize() const;
  size_t num_queries() const;
  const std::vector<QueryResult> &queries() const;

private:
  std::vector<QueryResult> queries_;
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
