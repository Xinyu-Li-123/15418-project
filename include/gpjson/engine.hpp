#pragma once

#include "gpjson/index/index_builder.hpp"
#include "gpjson/query/query.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace gpjson {

struct EngineOptions {
  index::IndexBuilderType index_builder_type{
      index::IndexBuilderType::UNCOMBINED};
  size_t file_partition_size{0};
};

class Engine {
public:
  query::MaterializedBatchResult
  query(const std::string &file_path,
        const std::vector<std::string> &queries_src,
        const EngineOptions &options) const;
};

} // namespace gpjson
