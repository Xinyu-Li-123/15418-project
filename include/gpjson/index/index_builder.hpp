#pragma once

#include "gpjson/file/file.hpp"
#include "gpjson/file/file_reader.hpp"
#include "gpjson/index/index.hpp"

#include <cstddef>

namespace gpjson {
struct EngineOptions;
}

namespace gpjson::index {

enum class IndexBuilderType { UNCOMBINED = 0, COMBINED, NO_ESCAPE_QUOTE };

class IndexBuilder {
public:
  virtual ~IndexBuilder() = default;

  virtual BuiltIndices build(const file::PartitionView &partition_view,
                             size_t max_depth,
                             const gpjson::EngineOptions &options) const = 0;
};

class UncombinedIndexBuilder final : public IndexBuilder {
public:
  explicit UncombinedIndexBuilder(const file::FileReader &file_reader);

  BuiltIndices build(const file::PartitionView &partition_view,
                     size_t max_depth,
                     const gpjson::EngineOptions &options) const override;

private:
  const file::FileReader &file_reader_;
};

class CombinedIndexBuilder final : public IndexBuilder {
public:
  explicit CombinedIndexBuilder(const file::FileReader &file_reader);

  BuiltIndices build(const file::PartitionView &partition_view,
                     size_t max_depth,
                     const gpjson::EngineOptions &options) const override;

private:
  const file::FileReader &file_reader_;
};
} // namespace gpjson::index
