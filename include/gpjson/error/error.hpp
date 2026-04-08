#pragma once

#include <exception>
#include <string>
#include <utility>

namespace gpjson::error {

class GpJSONError : public std::exception {
public:
  explicit GpJSONError(std::string message)
      : message_(std::move(message)) {}

  const char *what() const noexcept override { return message_.c_str(); }

private:
  std::string message_;
};

} // namespace gpjson::error
