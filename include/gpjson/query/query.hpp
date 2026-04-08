#pragma once

namespace gpjson::query {
class CompiledQuery {};

class BatchCompiledQuery {};

class QueryResult {};

class BatchQueryResult {};

class MaterializedValue {
private
  std::string json_text;
};

class MaterializedLineResult {
private:
  /* If empty, no match is found in this json object */
  std::vector<MaterializedValue> json_texts;
};

class MaterializedQueryResult {
private:
  std::string query_text;
  std::vector<MaterializedLineResult> lines;
};

class MaterializedBatchQueryResult {
private:
  std::vector<MaterializedQueryResult> queries;
};

class QueryOptions {};
} // namespace gpjson::query
