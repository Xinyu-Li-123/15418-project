#pragma once

#include "gpjson/error/error.hpp"

namespace gpjson::error::cuda {

class CudaError : public gpjson::error::GpJSONError {
public:
  using GpJSONError::GpJSONError;
};

} // namespace gpjson::error::cuda
