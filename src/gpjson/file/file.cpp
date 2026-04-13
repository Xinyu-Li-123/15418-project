#include "gpjson/file/file.hpp"

namespace gpjson::file {

size_t PartitionView::partition_id() const { return partition_id_; }

size_t PartitionView::global_start_offset() const {
  return global_start_offset_;
}

size_t PartitionView::global_end_offset() const { return global_end_offset_; }

size_t PartitionView::size_bytes() const {
  return global_end_offset_ - global_start_offset_;
}

const void *PartitionView::bytes() const { return data_; }

} // namespace gpjson::file
