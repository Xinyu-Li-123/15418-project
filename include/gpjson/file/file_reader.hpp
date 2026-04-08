#pragma once

#include "gpjson/file/file.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace gpjson::file {

class FileReader {
public:
  explicit FileReader(std::string file_path);

  void create_partitions(size_t partition_size_bytes);
  const std::vector<PartitionView> &get_partition_views() const;
  const FileMetadata &metadata() const;

private:
  std::string file_path_;
  FileMetadata metadata_;
  std::vector<std::byte> mapped_bytes_;
  std::vector<PartitionView> partitions_;
};

} // namespace gpjson::file
