#pragma once

#include "gpjson/types.hpp"

#include <filesystem>

namespace gpjson {

// Loads a full input file into host and/or device memory.
class FileLoader {
 public:
  // Creates a loader that uses the project's fixed loading policy.
  FileLoader() = default;

  // Loads the file and returns the resulting buffers and metadata.
  LoadedFileHandle load(const std::filesystem::path &path);

  // Releases any resources owned by a previously loaded file.
  void release(const LoadedFileHandle &loaded_file) ;
};

}  // namespace gpjson
