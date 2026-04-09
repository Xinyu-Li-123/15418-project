#include "gpjson/file/file_reader.hpp"

#include <utility>

namespace gpjson::file {

FileReader::FileReader(std::string file_path)
    : file_path_(std::move(file_path)) {
  metadata_.file_path = file_path_;
}

void FileReader::create_partitions(size_t partition_size_bytes) {
  (void)partition_size_bytes;
  partitions_.clear();
  metadata_.num_partitions = partitions_.size();
}

const std::vector<PartitionView> &FileReader::get_partition_views() const {
  return partitions_;
}

const FileMetadata &FileReader::metadata() const { return metadata_; }

} // namespace gpjson::file
