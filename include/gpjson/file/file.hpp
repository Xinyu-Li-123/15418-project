#pragma once

#include <cstddef>
#include <span>
#include <string>

namespace gpjson::file {

class PartitionView {
public:
  size_t partition_id() const;
  size_t global_start_offset() const;
  size_t global_end_offset() const;
  size_t size_bytes() const;
  std::span<const std::byte> bytes() const;

private:
  size_t partition_id_{0};
  size_t global_start_offset_{0};
  size_t global_end_offset_{0};
  const std::byte *data_{nullptr};
};

struct FileMetadata {
  std::string file_path;
  size_t file_size_bytes{0};
  size_t num_partitions{0};
};

} // namespace gpjson::file
