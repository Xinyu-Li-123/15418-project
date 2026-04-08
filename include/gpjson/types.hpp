#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>

namespace gpjson {

// Stores one loaded input file and the memory buffers associated with it.
struct LoadedFile {
  // Source file path on disk.
  std::filesystem::path path{};
  // Total file size in bytes.
  std::uint64_t file_size_bytes = 0;
  // Optional host-side copy or mapping of the input file.
  void *host_data = nullptr;
  // Size in bytes of the host-side input buffer.
  std::uint64_t host_data_bytes = 0;
  // Device buffer containing the full input file.
  void *device_data = nullptr;
  // Size in bytes of the device-side input buffer.
  std::uint64_t device_data_bytes = 0;
};

// Stores the newline index generated for one input file.
struct NewlineIndex {
  // Device or host pointer to the newline index buffer.
  void *data = nullptr;
  // Size in bytes of the newline index buffer.
  std::uint64_t bytes = 0;
  // Number of JSON lines described by this index.
  std::uint64_t num_lines = 0;
};

// Stores the string mask index generated for one input file.
struct StringIndex {
  // Device or host pointer to the string index buffer.
  void *data = nullptr;
  // Size in bytes of the string index buffer.
  std::uint64_t bytes = 0;
};

// Stores the leveled bitmap index generated for one input file.
struct LeveledBitmapIndex {
  // Device or host pointer to the leveled bitmap buffer.
  void *data = nullptr;
  // Size in bytes of the leveled bitmap buffer.
  std::uint64_t bytes = 0;
  // Maximum nesting depth encoded in the bitmap.
  std::uint32_t max_level = 0;
};

// Groups together all indexes generated from one loaded file.
struct BuiltIndex {
  // Source file path used to build this index.
  std::filesystem::path source_path{};
  // Size in bytes of the indexed source file.
  std::uint64_t source_size_bytes = 0;
  // Newline index for locating record boundaries.
  NewlineIndex newline{};
  // String index for identifying quoted regions.
  StringIndex string{};
  // Leveled bitmap index for structural navigation by depth.
  LeveledBitmapIndex leveled_bitmap{};
};

// Shared owner for one loaded file object.
using LoadedFileHandle = std::shared_ptr<LoadedFile>;
// Shared owner for one complete built index object.
using BuiltIndexHandle = std::shared_ptr<BuiltIndex>;

// Writes a newline index to disk.
void save_newline_index(const NewlineIndex &index,
                        const std::filesystem::path &path);

// Writes a string index to disk.
void save_string_index(const StringIndex &index,
                       const std::filesystem::path &path);

// Writes a leveled bitmap index to disk.
void save_leveled_bitmap_index(const LeveledBitmapIndex &index,
                               const std::filesystem::path &path);

// Loads a newline index from disk.
NewlineIndex load_newline_index(const std::filesystem::path &path);

// Loads a string index from disk.
StringIndex load_string_index(const std::filesystem::path &path);

// Loads a leveled bitmap index from disk.
LeveledBitmapIndex load_leveled_bitmap_index(
    const std::filesystem::path &path);

// Writes all index components to a directory on disk.
void save_built_index(const BuiltIndex &index,
                      const std::filesystem::path &directory);

// Loads a complete built index from a directory on disk.
BuiltIndexHandle load_built_index(const std::filesystem::path &directory);

}  // namespace gpjson
