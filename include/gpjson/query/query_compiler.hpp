#pragma once

#include "gpjson/query/query.hpp"

#include <string>

namespace gpjson {
struct EngineOptions;
}

namespace gpjson::query {

class QueryCompiler {
public:
  explicit QueryCompiler(const gpjson::EngineOptions &options);

  CompiledQuery compile(const std::string &query_src,
                        const gpjson::EngineOptions &options) const;
};

} // namespace gpjson::query
