#pragma once

#include "gpjson/query/error.hpp"
#include "gpjson/query/query.hpp"

#include <cstddef>
#include <span>
#include <string>

namespace gpjson::query {

class IRByteInputBuffer {
public:
  explicit IRByteInputBuffer(std::span<const std::byte> bytes);

  bool has_next() const;
  QueryOpcode read_opcode();
  size_t read_varint();
  std::string read_string();

private:
  std::byte read_byte();

  std::span<const std::byte> bytes_;
  size_t position_{0};
};

} // namespace gpjson::query
