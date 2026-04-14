#include "gpjson/query/ir_input_buffer.hpp"

namespace gpjson::query {

IRByteInputBuffer::IRByteInputBuffer(const std::byte *bytes, size_t size)
    : bytes_(bytes), size_(size) {}

bool IRByteInputBuffer::has_next() const { return position_ < size_; }

QueryOpcode IRByteInputBuffer::read_opcode() {
  return static_cast<QueryOpcode>(read_byte());
}

size_t IRByteInputBuffer::read_varint() {
  size_t value = 0;
  size_t shift = 0;
  while (true) {
    const auto byte = static_cast<unsigned char>(read_byte());
    value |= static_cast<size_t>(byte & 0x7F) << shift;
    if ((byte & 0x80U) == 0) {
      break;
    }
    shift += 7;
  }
  return value;
}

std::string IRByteInputBuffer::read_string() {
  const size_t size = read_varint();
  std::string value;
  value.reserve(size);
  for (size_t index = 0; index < size; ++index) {
    value.push_back(static_cast<char>(read_byte()));
  }
  return value;
}

std::byte IRByteInputBuffer::read_byte() {
  if (position_ >= size_) {
    throw error::query::QueryExecutionError("Unexpected end of query IR");
  }
  return bytes_[position_++];
}

} // namespace gpjson::query
