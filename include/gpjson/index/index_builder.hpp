#pragma once

#include "gpjson/file/file.hpp"
#include "gpjson/file/file_reader.hpp"
#include "gpjson/index/index.hpp"

#include <cstddef>
#include <iosfwd>
#include <ostream>
#include <type_traits>

namespace gpjson::index {

enum class IndexBuilderType { UNCOMBINED = 0, COMBINED, FUSED };

inline std::ostream &operator<<(std::ostream &os, IndexBuilderType type) {
  switch (type) {
  case IndexBuilderType::UNCOMBINED:
    return os << "UNCOMBINED";
  case IndexBuilderType::COMBINED:
    return os << "COMBINED";
  case IndexBuilderType::FUSED:
    return os << "FUSED";
  default:
    return os << "UNKNOWN("
              << static_cast<std::underlying_type_t<IndexBuilderType>>(type)
              << ")";
  }
}

struct IndexBuilderOptions {
  index::IndexBuilderType index_builder_type{
      index::IndexBuilderType::UNCOMBINED};
  size_t file_partition_size{0};
  int grid_size{0};
  int block_size{0};
  int reduction_grid_size{0};
  int reduction_block_size{0};
};

inline std::ostream &operator<<(std::ostream &os,
                                const IndexBuilderOptions &options) {
  return os << "IndexBuilderOptions{index_builder_type="
            << options.index_builder_type
            << ", file_partition_size=" << options.file_partition_size
            << ", grid_size=" << options.grid_size
            << ", block_size=" << options.block_size
            << ", reduction_grid_size=" << options.reduction_grid_size
            << ", reduction_block_size=" << options.reduction_block_size << "}";
}

class IndexBuilder {
public:
  virtual ~IndexBuilder() = default;

  virtual BuiltIndices build(const file::FilePartition &partition,
                             size_t max_depth,
                             const IndexBuilderOptions &options) const = 0;
};

class UncombinedIndexBuilder final : public IndexBuilder {
public:
  explicit UncombinedIndexBuilder(const file::FileReader &file_reader);

  BuiltIndices build(const file::FilePartition &partition, size_t max_depth,
                     const IndexBuilderOptions &options) const override;

private:
  const file::FileReader &file_reader_;
};

class CombinedIndexBuilder final : public IndexBuilder {
public:
  explicit CombinedIndexBuilder(const file::FileReader &file_reader);

  BuiltIndices build(const file::FilePartition &partition, size_t max_depth,
                     const IndexBuilderOptions &options) const override;

private:
  const file::FileReader &file_reader_;
};

class FusedIndexBuilder final : public IndexBuilder {
public:
  explicit FusedIndexBuilder(const file::FileReader &file_reader);

  BuiltIndices build(const file::FilePartition &partition, size_t max_depth,
                     const IndexBuilderOptions &options) const override;

private:
  const file::FileReader &file_reader_;
};

} // namespace gpjson::index
