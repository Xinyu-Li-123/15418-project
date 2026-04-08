#pragma once

#include "index/index_builder.hpp"
#include "query/query.hpp"
namespace gpjson {
struct EngineOptions {
  index::IndexBuilderType index_builder_type;
};

class Engine {
public:
  std::vector<query::QueryResult> query(std::string file_path,
                                        std::vector<std::string> queries_src,
                                        EngineOptions options);
}
} // namespace gpjson
