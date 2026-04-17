#include "gpjson/file/file.hpp"

namespace gpjson::file {

PartitionView::PartitionView(size_t partition_id,
                             size_t global_start_offset,
                             size_t global_end_offset,
                             const void *data)
    : partition_id_(partition_id), global_start_offset_(global_start_offset),
      global_end_offset_(global_end_offset), data_(data) {}

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
