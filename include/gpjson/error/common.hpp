#pragma once

#include "gpjson/error/error.hpp"

namespace gpjson::error::common {

class ImplementationError : public gpjson::error::GpJSONError {
public:
  using GpJSONError::GpJSONError;
};

class NotImplementedError : public gpjson::error::GpJSONError {
public:
  using GpJSONError::GpJSONError;
};

} // namespace gpjson::error::common
