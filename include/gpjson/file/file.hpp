#pragma once

#include "gpjson/cuda/cuda.hpp"

#include <cstddef>
#include <string>

namespace gpjson::file {

/*
 * RAII class that refers to a file partition on host and owns a copy of that
 * file partition on device.
 *
 * The host file partition is owned by FileReader instead of FilePartition. This
 * is because we may read the file in different ways, e.g. mmap'd or by
 * duplicating same small file multiple times on-the-fly. It would be better to
 * let FileReader handle this complexity.
 **/
class FilePartition {
public:
  FilePartition() = default;
  FilePartition(size_t partition_id, size_t global_start_offset,
                size_t global_end_offset, const void *data);

  FilePartition(const FilePartition &) = delete;
  FilePartition &operator=(const FilePartition &) = delete;

  FilePartition(FilePartition &&) noexcept = default;
  FilePartition &operator=(FilePartition &&) noexcept = default;

  size_t partition_id() const;
  size_t global_start_offset() const;
  size_t global_end_offset() const;
  size_t size_bytes() const;
  const void *host_bytes() const;
  const void *device_bytes() const;
  bool device_loaded() const;
  /* Copy partition data from host to device. This method is idempotent: calling
   * it twice won't copy data twice, and will just reuse existing copy. */
  void load_to_device();

private:
  size_t partition_id_{0};
  size_t global_start_offset_{0};
  size_t global_end_offset_{0};
  const void *host_data_{nullptr};
  cuda::DeviceArray device_data_;
};

struct FileMetadata {
  std::string file_path;
  size_t file_size_bytes{0};
  size_t num_partitions{0};
};

} // namespace gpjson::file
