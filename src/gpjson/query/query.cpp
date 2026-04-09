#include "gpjson/query/query.hpp"

#include <algorithm>
#include <utility>

namespace gpjson::query {

const std::string &CompiledQuery::query_text() const { return query_text_; }

const std::vector<std::byte> &CompiledQuery::ir_bytes() const {
  return ir_bytes_;
}

size_t CompiledQuery::max_depth() const { return max_depth_; }

size_t CompiledQuery::num_result_slots() const { return num_result_slots_; }

void BatchCompiledQuery::add(CompiledQuery query) {
  max_depth_ = std::max(max_depth_, query.max_depth());
  queries_.push_back(std::move(query));
}

size_t BatchCompiledQuery::size() const { return queries_.size(); }

size_t BatchCompiledQuery::get_max_depth() const { return max_depth_; }

const std::vector<CompiledQuery> &BatchCompiledQuery::queries() const {
  return queries_;
}

const std::vector<QueryOffset> &LineQueryResult::offsets() const {
  return offsets_;
}

void LineQueryResult::add_offset(QueryOffset offset) {
  offsets_.push_back(std::move(offset));
}

void QueryResult::add_line_result(LineQueryResult line_result) {
  lines_.push_back(std::move(line_result));
}

const std::vector<LineQueryResult> &QueryResult::lines() const {
  return lines_;
}

BatchQueryResult::BatchQueryResult(size_t num_queries)
    : queries_(num_queries) {}

void BatchQueryResult::merge(const BatchQueryResult &other) { (void)other; }

MaterializedBatchResult BatchQueryResult::materialize() const { return {}; }

size_t BatchQueryResult::num_queries() const { return queries_.size(); }

const std::vector<QueryResult> &BatchQueryResult::queries() const {
  return queries_;
}

MaterializedValue::MaterializedValue(std::string json_text)
    : json_text_(std::move(json_text)) {}

const std::string &MaterializedValue::json_text() const { return json_text_; }

const std::vector<MaterializedValue> &MaterializedLineResult::values() const {
  return values_;
}

void MaterializedLineResult::add_value(MaterializedValue value) {
  values_.push_back(std::move(value));
}

MaterializedQueryResult::MaterializedQueryResult(std::string query_text)
    : query_text_(std::move(query_text)) {}

const std::string &MaterializedQueryResult::query_text() const {
  return query_text_;
}

const std::vector<MaterializedLineResult> &
MaterializedQueryResult::lines() const {
  return lines_;
}

void MaterializedQueryResult::add_line_result(MaterializedLineResult line) {
  lines_.push_back(std::move(line));
}

const std::vector<MaterializedQueryResult> &
MaterializedBatchResult::queries() const {
  return queries_;
}

void MaterializedBatchResult::add_query_result(MaterializedQueryResult result) {
  queries_.push_back(std::move(result));
}

} // namespace gpjson::query
