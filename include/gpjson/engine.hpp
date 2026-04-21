#pragma once

#include "gpjson/file/file_reader.hpp"
#include "gpjson/index/index_builder.hpp"
#include "gpjson/query/query.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace gpjson {

struct EngineOptions {
  file::FileReaderOptions file_reader_options;
  index::IndexBuilderOptions index_builder_options;
};

class Engine {
public:
  query::MaterializedBatchResult
  query(const std::string &file_path,
        const std::vector<std::string> &queries_src,
        const EngineOptions &options) const;
};

} // namespace gpjson
