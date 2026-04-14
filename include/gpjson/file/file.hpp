#pragma once

#include <cstddef>
#include <string>

namespace gpjson::file {

class PartitionView {
public:
  PartitionView() = default;
  PartitionView(size_t partition_id,
                size_t global_start_offset,
                size_t global_end_offset,
                const void *data);

  size_t partition_id() const;
  size_t global_start_offset() const;
  size_t global_end_offset() const;
  size_t size_bytes() const;
  const void *bytes() const;

private:
  size_t partition_id_{0};
  size_t global_start_offset_{0};
  size_t global_end_offset_{0};
  const void *data_{nullptr};
};

struct FileMetadata {
  std::string file_path;
  size_t file_size_bytes{0};
  size_t num_partitions{0};
};

} // namespace gpjson::file
