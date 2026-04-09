#include "gpjson/query/query_compiler.hpp"

namespace gpjson::query {

QueryCompiler::QueryCompiler(const gpjson::EngineOptions &options) {
  (void)options;
}

CompiledQuery
QueryCompiler::compile(const std::string &query_src,
                       const gpjson::EngineOptions &options) const {
  (void)query_src;
  (void)options;
  return {};
}

} // namespace gpjson::query
