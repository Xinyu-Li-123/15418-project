#pragma once

#include "gpjson/file/file.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace gpjson::file {

enum class FileReaderType { COPIED = 0, MMAP };

struct FileReaderOptions {
  file::FileReaderType file_reader_type{file::FileReaderType::COPIED};
};

class FileReader {
public:
  virtual ~FileReader() = default;

  virtual void create_partitions(size_t partition_size_bytes) = 0;
  virtual std::vector<FilePartition> &get_partitions() = 0;
  virtual const std::vector<FilePartition> &get_partitions() const = 0;
  virtual const FileMetadata &metadata() const = 0;

protected:
  FileReader() = default;
};

class CopiedFileReader final : public FileReader {
public:
  explicit CopiedFileReader(std::string file_path);

  void create_partitions(size_t partition_size_bytes) override;
  std::vector<FilePartition> &get_partitions() override;
  const std::vector<FilePartition> &get_partitions() const override;
  const FileMetadata &metadata() const override;

private:
  void load_file_bytes();
  size_t find_next_partition_start(size_t current_partition_start,
                                   size_t partition_size_bytes) const;

  std::string file_path_;
  FileMetadata metadata_;
  std::vector<std::byte> mapped_bytes_;
  std::vector<FilePartition> partitions_;
};

} // namespace gpjson::file
