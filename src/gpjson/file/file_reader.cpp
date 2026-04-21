#include "gpjson/file/file_reader.hpp"

#include "gpjson/file/error.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>

namespace gpjson::file {
namespace {

size_t find_next_partition_start(const std::string &file_path,
                                 const std::byte *file_bytes,
                                 size_t current_partition_start,
                                 size_t partition_size_bytes) {
  const size_t target_offset = current_partition_start + partition_size_bytes;
  size_t offset = target_offset;
  while (true) {
    if (file_bytes[offset] == static_cast<std::byte>('\n')) {
      return offset + 1;
    }
    if (offset == current_partition_start) {
      break;
    }
    --offset;
  }

  std::ostringstream message;
  message << "Cannot partition file '" << file_path << "' using partition size "
          << partition_size_bytes << ": no newline found before offset "
          << target_offset << ".";
  throw error::file::PartitionError(message.str());
}

std::vector<FilePartition>
create_partitions_for_bytes(const std::string &file_path,
                            const std::byte *base_ptr, size_t file_size_bytes,
                            size_t partition_size_bytes) {
  std::vector<FilePartition> partitions;

  if (partition_size_bytes == 0 || file_size_bytes == 0 ||
      file_size_bytes <= partition_size_bytes) {
    partitions.emplace_back(0, 0, file_size_bytes, base_ptr);
    return partitions;
  }

  size_t current_partition_start = 0;
  size_t partition_id = 0;

  while (file_size_bytes - current_partition_start > partition_size_bytes) {
    const size_t next_partition_start = find_next_partition_start(
        file_path, base_ptr, current_partition_start, partition_size_bytes);
    if (next_partition_start >= file_size_bytes) {
      break;
    }

    const void *partition_ptr =
        base_ptr == nullptr ? nullptr : base_ptr + current_partition_start;
    partitions.emplace_back(partition_id, current_partition_start,
                            next_partition_start - 1, partition_ptr);

    current_partition_start = next_partition_start;
    ++partition_id;
  }

  const void *partition_ptr =
      base_ptr == nullptr ? nullptr : base_ptr + current_partition_start;
  partitions.emplace_back(partition_id, current_partition_start,
                          file_size_bytes, partition_ptr);
  return partitions;
}

} // namespace

CopiedFileReader::CopiedFileReader(std::string file_path)
    : file_path_(std::move(file_path)) {
  metadata_.file_path = file_path_;
}

void CopiedFileReader::create_partitions(size_t partition_size_bytes) {
  load_file_bytes();
  const std::byte *base_ptr =
      mapped_bytes_.empty() ? nullptr : mapped_bytes_.data();
  partitions_ = create_partitions_for_bytes(
      file_path_, base_ptr, metadata_.file_size_bytes, partition_size_bytes);
  metadata_.num_partitions = partitions_.size();
}

std::vector<FilePartition> &CopiedFileReader::get_partitions() {
  return partitions_;
}

const std::vector<FilePartition> &CopiedFileReader::get_partitions() const {
  return partitions_;
}

const FileMetadata &CopiedFileReader::metadata() const { return metadata_; }

void CopiedFileReader::load_file_bytes() {
  std::error_code filesystem_error;
  const std::filesystem::path path(file_path_);
  const auto file_size = std::filesystem::file_size(path, filesystem_error);
  if (filesystem_error) {
    throw error::file::FileOpenError("Failed to get size for file '" +
                                     file_path_ +
                                     "': " + filesystem_error.message());
  }

  metadata_.file_size_bytes = static_cast<size_t>(file_size);
  mapped_bytes_.assign(metadata_.file_size_bytes, std::byte{0});

  std::ifstream input(file_path_, std::ios::binary);
  if (!input) {
    throw error::file::FileOpenError("Failed to open file '" + file_path_ +
                                     "' for reading.");
  }

  if (metadata_.file_size_bytes == 0) {
    return;
  }

  input.read(reinterpret_cast<char *>(mapped_bytes_.data()),
             static_cast<std::streamsize>(metadata_.file_size_bytes));
  if (!input ||
      static_cast<size_t>(input.gcount()) != metadata_.file_size_bytes) {
    throw error::file::FileReadError("Failed to read all bytes from file '" +
                                     file_path_ + "'.");
  }
}

MmapFileReader::MmapFileReader(std::string file_path)
    : file_path_(std::move(file_path)) {
  metadata_.file_path = file_path_;
}

MmapFileReader::~MmapFileReader() { release_mapping(); }

void MmapFileReader::create_partitions(size_t partition_size_bytes) {
  map_file_bytes();
  partitions_ = create_partitions_for_bytes(file_path_, mapped_bytes_,
                                            metadata_.file_size_bytes,
                                            partition_size_bytes);
  metadata_.num_partitions = partitions_.size();
}

std::vector<FilePartition> &MmapFileReader::get_partitions() {
  return partitions_;
}

const std::vector<FilePartition> &MmapFileReader::get_partitions() const {
  return partitions_;
}

const FileMetadata &MmapFileReader::metadata() const { return metadata_; }

void MmapFileReader::map_file_bytes() {
  if (fd_ != -1 || metadata_.file_size_bytes != 0 || mapped_size_bytes_ != 0) {
    return;
  }

  std::error_code filesystem_error;
  const std::filesystem::path path(file_path_);
  const auto file_size = std::filesystem::file_size(path, filesystem_error);
  if (filesystem_error) {
    throw error::file::FileOpenError("Failed to get size for file '" +
                                     file_path_ +
                                     "': " + filesystem_error.message());
  }

  metadata_.file_size_bytes = static_cast<size_t>(file_size);
  fd_ = open(file_path_.c_str(), O_RDONLY);
  if (fd_ == -1) {
    throw error::file::FileOpenError("Failed to open file '" + file_path_ +
                                     "' for reading: " + std::strerror(errno) +
                                     ".");
  }

  if (metadata_.file_size_bytes == 0) {
    return;
  }

  void *mapping =
      mmap(nullptr, metadata_.file_size_bytes, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (mapping == MAP_FAILED) {
    const std::string error_message = std::strerror(errno);
    release_mapping();
    throw error::file::FileReadError("Failed to mmap file '" + file_path_ +
                                     "': " + error_message + ".");
  }

  mapped_bytes_ = static_cast<const std::byte *>(mapping);
  mapped_size_bytes_ = metadata_.file_size_bytes;
}

void MmapFileReader::release_mapping() noexcept {
  if (mapped_bytes_ != nullptr && mapped_size_bytes_ > 0) {
    munmap(const_cast<std::byte *>(mapped_bytes_), mapped_size_bytes_);
  }
  mapped_bytes_ = nullptr;
  mapped_size_bytes_ = 0;

  if (fd_ != -1) {
    close(fd_);
    fd_ = -1;
  }
}

} // namespace gpjson::file
