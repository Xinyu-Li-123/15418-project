#include "gpjson/index/index_builder.hpp"

namespace gpjson::index {

UncombinedIndexBuilder::UncombinedIndexBuilder(
    const file::FileReader &file_reader)
    : file_reader_(file_reader) {}

BuiltIndices
UncombinedIndexBuilder::build(const file::PartitionView &partition_view,
                              size_t max_depth,
                              const gpjson::EngineOptions &options) const {
  (void)partition_view;
  (void)max_depth;
  (void)options;
  return {};
}

CombinedIndexBuilder::CombinedIndexBuilder(const file::FileReader &file_reader)
    : file_reader_(file_reader) {}

BuiltIndices
CombinedIndexBuilder::build(const file::PartitionView &partition_view,
                            size_t max_depth,
                            const gpjson::EngineOptions &options) const {
  (void)partition_view;
  (void)max_depth;
  (void)options;
  return {};
}

} // namespace gpjson::index
