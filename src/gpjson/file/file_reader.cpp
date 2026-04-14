#include "gpjson/file/file_reader.hpp"

#include "gpjson/file/error.hpp"

#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

namespace gpjson::file {

FileReader::FileReader(std::string file_path)
    : file_path_(std::move(file_path)) {
  metadata_.file_path = file_path_;
}


void FileReader::create_partitions(size_t partition_size_bytes) {

  load_file_bytes();

  partitions_.clear();

  const std::byte *base_ptr =
      mapped_bytes_.empty() ? nullptr : mapped_bytes_.data();


  // if file size is too small, there is only one partition
  if (partition_size_bytes == 0 || metadata_.file_size_bytes == 0 ||
      metadata_.file_size_bytes <= partition_size_bytes) {
    partitions_.push_back(PartitionView{0, 0, metadata_.file_size_bytes, base_ptr});
    metadata_.num_partitions = partitions_.size();
    return;
  }

  size_t current_partition_start = 0;
  size_t partition_id = 0;

  while (metadata_.file_size_bytes - current_partition_start >
         partition_size_bytes) {
    const size_t next_partition_start =
        find_next_partition_start(current_partition_start, partition_size_bytes);
    if (next_partition_start >= metadata_.file_size_bytes) {
      break;
    }

    const void *partition_ptr =
        base_ptr == nullptr ? nullptr : base_ptr + current_partition_start;
    partitions_.push_back(PartitionView{partition_id, current_partition_start,
                             next_partition_start - 1, partition_ptr});

    current_partition_start = next_partition_start;
    ++partition_id;
  }

  const void *partition_ptr =
      base_ptr == nullptr ? nullptr : base_ptr + current_partition_start;
  partitions_.push_back(PartitionView{partition_id, current_partition_start,
                           metadata_.file_size_bytes, partition_ptr});

  metadata_.num_partitions = partitions_.size();
}

const std::vector<PartitionView> &FileReader::get_partition_views() const {
  return partitions_;
}

const FileMetadata &FileReader::metadata() const { return metadata_; }


void FileReader::load_file_bytes() {
  std::error_code filesystem_error;
  const std::filesystem::path path(file_path_);
  const auto file_size = std::filesystem::file_size(path, filesystem_error);
  if (filesystem_error) {
    throw error::file::FileOpenError("Failed to get size for file '" +
                                     file_path_ + "': " +
                                     filesystem_error.message());
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


// Finds the next partition start offset by searching backwards from the target offset for the nearest newline character. 
// If no newline is found before the target offset, the function throw an error 
size_t FileReader::find_next_partition_start(size_t current_partition_start,
                                             size_t partition_size_bytes) const {
  const size_t target_offset = current_partition_start + partition_size_bytes;
  size_t offset = target_offset;
  while (true) {
    if (mapped_bytes_[offset] == static_cast<std::byte>('\n')) {
      return offset + 1;
    }
    if (offset == current_partition_start)
      break;
    offset--;
  }

  std::ostringstream message;
  message << "Cannot partition file '" << file_path_
          << "' using partition size " << partition_size_bytes
          << ": no newline found before offset " << target_offset << ".";
  throw error::file::PartitionError(message.str());
}

} // namespace gpjson::file
