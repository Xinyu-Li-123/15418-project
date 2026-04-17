#include "gpjson/query/query.hpp"

#include <algorithm>
#include <utility>

namespace gpjson::query {

CompiledQuery::CompiledQuery(std::string query_text,
                             std::vector<std::byte> ir_bytes,
                             size_t max_depth,
                             size_t num_result_slots)
    : query_text_(std::move(query_text)), ir_bytes_(std::move(ir_bytes)),
      max_depth_(max_depth), num_result_slots_(num_result_slots) {}

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
    : queries_(num_queries), query_texts_(num_queries) {}

void BatchQueryResult::add_line_result(size_t query_index,
                                       LineQueryResult line_result) {
  if (query_index >= queries_.size()) {
    return;
  }
  queries_[query_index].add_line_result(std::move(line_result));
}

void BatchQueryResult::merge(const BatchQueryResult &other) {
  if (queries_.size() != other.queries_.size()) {
    return;
  }

  for (size_t query_index = 0; query_index < queries_.size(); ++query_index) {
    if (query_texts_[query_index].empty()) {
      query_texts_[query_index] = other.query_texts_[query_index];
    }
    for (const auto &line_result : other.queries_[query_index].lines()) {
      queries_[query_index].add_line_result(line_result);
    }
  }
}

void BatchQueryResult::set_query_text(size_t query_index, std::string query_text) {
  if (query_index >= query_texts_.size()) {
    return;
  }
  query_texts_[query_index] = std::move(query_text);
}

MaterializedBatchResult BatchQueryResult::materialize() const {
  MaterializedBatchResult batch_result;

  for (size_t query_index = 0; query_index < queries_.size(); ++query_index) {
    MaterializedQueryResult query_result(query_texts_[query_index]);
    for (const auto &line_result : queries_[query_index].lines()) {
      MaterializedLineResult materialized_line;
      for (const auto &offset : line_result.offsets()) {
        if (!offset.json_text.has_value()) {
          continue;
        }
        materialized_line.add_value(MaterializedValue(*offset.json_text));
      }
      if (materialized_line.values().empty()) {
        continue;
      }
      query_result.add_line_result(std::move(materialized_line));
    }
    batch_result.add_query_result(std::move(query_result));
  }

  return batch_result;
}

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
