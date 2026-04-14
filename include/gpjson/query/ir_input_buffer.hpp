#pragma once

#include "gpjson/query/error.hpp"
#include "gpjson/query/query.hpp"

#include <cstddef>
#include <string>

namespace gpjson::query {

class IRByteInputBuffer {
public:
  IRByteInputBuffer(const std::byte *bytes, size_t size);

  bool has_next() const;
  QueryOpcode read_opcode();
  size_t read_varint();
  std::string read_string();

private:
  std::byte read_byte();

  const std::byte *bytes_{nullptr};
  size_t size_{0};
  size_t position_{0};
};

} // namespace gpjson::query
