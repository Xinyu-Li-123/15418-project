#pragma once

#include "gpjson/error/error.hpp"

namespace gpjson::error::file {

class FileError : public gpjson::error::GpJSONError {
public:
  using GpJSONError::GpJSONError;
};

class FileOpenError : public FileError {
public:
  using FileError::FileError;
};

class FileReadError : public FileError {
public:
  using FileError::FileError;
};

class PartitionError : public FileError {
public:
  using FileError::FileError;
};

} // namespace gpjson::error::file
