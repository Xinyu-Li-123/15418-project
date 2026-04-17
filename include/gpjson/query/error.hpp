#pragma once

#include "gpjson/error/error.hpp"

namespace gpjson::error::query {

class MalformedQueryError : public gpjson::error::GpJSONError {
public:
  using GpJSONError::GpJSONError;
};

class UnsupportedQueryError : public gpjson::error::GpJSONError {
public:
  using GpJSONError::GpJSONError;
};

class QueryExecutionError : public gpjson::error::GpJSONError {
public:
  using GpJSONError::GpJSONError;
};

} // namespace gpjson::error::query
