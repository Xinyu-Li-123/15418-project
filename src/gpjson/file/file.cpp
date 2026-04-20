#include "gpjson/file/file.hpp"

namespace gpjson::file {

FilePartition::FilePartition(size_t partition_id, size_t global_start_offset,
                             size_t global_end_offset, const void *data)
    : partition_id_(partition_id), global_start_offset_(global_start_offset),
      global_end_offset_(global_end_offset), host_data_(data) {}

size_t FilePartition::partition_id() const { return partition_id_; }

size_t FilePartition::global_start_offset() const {
  return global_start_offset_;
}

size_t FilePartition::global_end_offset() const { return global_end_offset_; }

size_t FilePartition::size_bytes() const {
  return global_end_offset_ - global_start_offset_;
}

const void *FilePartition::host_bytes() const { return host_data_; }

const void *FilePartition::device_bytes() const { return device_data_.data(); }

bool FilePartition::device_loaded() const {
  return size_bytes() == 0 || device_data_.data() != nullptr;
}

void FilePartition::load_to_device() {
  if (device_loaded()) {
    return;
  }
  device_data_ = cuda::DeviceArray(size_bytes());
  device_data_.copy_from_host(host_data_, size_bytes());
}

} // namespace gpjson::file
