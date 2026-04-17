#include "gpjson/query/query_compiler.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace gpjson::query {

void IRByteOutputBuffer::write_opcode(QueryOpcode opcode) {
  bytes_.push_back(static_cast<std::byte>(opcode));
}

void IRByteOutputBuffer::write_byte(std::byte value) { bytes_.push_back(value); }

void IRByteOutputBuffer::write_varint(size_t value) {
  while ((value & ~size_t{0x7F}) != 0) {
    write_byte(static_cast<std::byte>((value & 0x7F) | 0x80));
    value >>= 7U;
  }
  write_byte(static_cast<std::byte>(value & 0x7F));
}

void IRByteOutputBuffer::write_string(const std::string &value) {
  write_varint(value.size());
  for (unsigned char byte : value) {
    write_byte(static_cast<std::byte>(byte));
  }
}

const std::vector<std::byte> &IRByteOutputBuffer::bytes() const {
  return bytes_;
}

IRBuilder::IRBuilder(IRByteOutputBuffer &buffer) : buffer_(buffer) {}

void IRBuilder::property(const std::string &name) {
  buffer_.write_opcode(QueryOpcode::MOVE_TO_KEY);
  buffer_.write_string(name);
}

void IRBuilder::index(size_t index) {
  buffer_.write_opcode(QueryOpcode::MOVE_TO_INDEX);
  buffer_.write_varint(index);
}

void IRBuilder::reverse_index(size_t index) {
  buffer_.write_opcode(QueryOpcode::MOVE_TO_INDEX_REVERSE);
  buffer_.write_varint(index);
}

void IRBuilder::down() {
  buffer_.write_opcode(QueryOpcode::MOVE_DOWN);
  ++current_level_;
}

void IRBuilder::up() {
  buffer_.write_opcode(QueryOpcode::MOVE_UP);
  if (current_level_ > 0) {
    --current_level_;
  }
}

void IRBuilder::mark() { buffer_.write_opcode(QueryOpcode::MARK_POSITION); }

void IRBuilder::reset() { buffer_.write_opcode(QueryOpcode::RESET_POSITION); }

void IRBuilder::store_result() {
  buffer_.write_opcode(QueryOpcode::STORE_RESULT);
  ++num_result_stores_;
}

void IRBuilder::end() { buffer_.write_opcode(QueryOpcode::END); }

void IRBuilder::expression_string_equals(const std::string &value) {
  buffer_.write_opcode(QueryOpcode::EXPRESSION_STRING_EQUALS);
  buffer_.write_string("\"" + value + "\"");
}

size_t IRBuilder::current_level() const { return current_level_; }

size_t IRBuilder::num_result_stores() const { return num_result_stores_; }

QueryScanner::QueryScanner(std::string query) : query(std::move(query)) {}

bool QueryScanner::has_next() const { return position + 1 < query.size(); }

char QueryScanner::next() {
  if (!has_next()) {
    throw malformed("Expected character, got EOF");
  }
  ++position;
  return query[position];
}

char QueryScanner::peek() const {
  if (!has_next()) {
    throw malformed("Expected character, got EOF");
  }
  return query[position + 1];
}

void QueryScanner::expect_char(char expected) {
  const char actual = next();
  if (actual != expected) {
    std::ostringstream message;
    message << "Expected character '" << expected << "', got '" << actual
            << "'";
    throw malformed(message.str());
  }
}

bool QueryScanner::skip_if_char(char expected) {
  if (!has_next() || peek() != expected) {
    return false;
  }
  ++position;
  return true;
}

void QueryScanner::test_digit() const {
  const char next_char = peek();
  if (!std::isdigit(static_cast<unsigned char>(next_char))) {
    std::ostringstream message;
    message << "Expected digit, got '" << next_char << "'";
    throw malformed(message.str());
  }
}

void QueryScanner::mark() { marked_positions.push_back(position); }

void QueryScanner::reset() {
  position = marked_positions.back();
  marked_positions.pop_back();
}

size_t QueryScanner::get_position() const { return position; }

std::string QueryScanner::substring(size_t start, size_t end) const {
  if (end < start) {
    return {};
  }
  return query.substr(start + 1, end - start);
}

error::query::MalformedQueryError
QueryScanner::malformed(const std::string &message) const {
  std::ostringstream output;
  output << message << " at " << (position + 1);
  if (position + 1 < query.size()) {
    output << " ('" << query[position + 1] << "')";
  }
  return error::query::MalformedQueryError(output.str());
}

error::query::UnsupportedQueryError
QueryScanner::unsupported(const std::string &message) const {
  std::ostringstream output;
  output << message << " at " << (position + 1);
  if (position + 1 < query.size()) {
    output << " ('" << query[position + 1] << "')";
  }
  return error::query::UnsupportedQueryError(output.str());
}

QueryParser::QueryParser(const std::string &query)
    : scanner(query), ir_buffer(), ir(ir_buffer), query_text(query) {}

CompiledQuery QueryParser::compile() {
  scanner.expect_char('$');
  compile_next_expression();
  ir.store_result();
  ir.end();
  return CompiledQuery(query_text, ir_buffer.bytes(), max_level,
                       ir.num_result_stores());
}

void QueryParser::compile_next_expression() {
  if (!scanner.has_next()) {
    return;
  }

  switch (scanner.peek()) {
  case '.':
    compile_dot_expression();
    break;
  case '[':
    compile_index_expression();
    break;
  case ' ':
    return;
  default:
    throw scanner.unsupported("Unsupported expression type");
  }
}

void QueryParser::compile_dot_expression() {
  scanner.expect_char('.');
  if (scanner.peek() == '.') {
    throw scanner.unsupported("Unsupported recursive descent");
  }

  const std::string property = read_property();
  if (property.empty()) {
    throw scanner.malformed("Unexpected empty property");
  }

  create_property_ir(property);
  if (scanner.has_next()) {
    compile_next_expression();
  }
}

void QueryParser::compile_index_expression() {
  scanner.expect_char('[');
  if (scanner.peek() == '\'' || scanner.peek() == '"') {
    const std::string property = read_quoted_string();
    if (property.empty()) {
      throw scanner.malformed("Unexpected empty property");
    }
    create_property_ir(property);
  } else if (std::isdigit(static_cast<unsigned char>(scanner.peek()))) {
    const size_t index =
        read_integer([](char c) { return c == ']' || c == ':' || c == ','; });
    switch (scanner.peek()) {
    case ':': {
      scanner.expect_char(':');
      const size_t end_index = read_integer([](char c) { return c == ']'; });
      scanner.expect_char(']');
      compile_index_range_expression(index, end_index, false);
      return;
    }
    case ',': {
      std::vector<size_t> indexes{index};
      while (scanner.peek() == ',') {
        scanner.expect_char(',');
        scanner.test_digit();
        indexes.push_back(
            read_integer([](char c) { return c == ',' || c == ']'; }));
      }
      scanner.expect_char(']');
      compile_multiple_index_expression(indexes);
      return;
    }
    case ']':
      create_index_ir(index);
      break;
    default:
      throw scanner.malformed("Unexpected character in index expression");
    }
  } else if (scanner.peek() == ':') {
    scanner.expect_char(':');
    if (!std::isdigit(static_cast<unsigned char>(scanner.peek()))) {
      throw scanner.malformed(
          "Unexpected character in index, expected an integer");
    }
    const size_t end_index = read_integer([](char c) { return c == ']'; });
    scanner.expect_char(']');
    compile_index_range_expression(0, end_index, false);
    return;
  } else if (scanner.peek() == '-') {
    scanner.expect_char('-');
    if (std::isdigit(static_cast<unsigned char>(scanner.peek()))) {
      const size_t index =
          read_integer([](char c) { return c == ']' || c == ':'; });
      if (index == 0) {
        throw scanner.malformed("Invalid reverse index 0");
      }
      if (scanner.peek() == ']') {
        create_reverse_index_ir(index);
      } else {
        scanner.expect_char(':');
        scanner.expect_char(']');
        compile_index_range_expression(0, index, true);
        return;
      }
    } else if (scanner.peek() == '-') {
      throw scanner.unsupported("Unsupported last n elements of the array query");
    } else {
      throw scanner.malformed(
          "Unexpected character in index, expected an integer");
    }
  } else if (scanner.peek() == '*') {
    throw scanner.unsupported("Unsupported wildcard expression");
  } else if (scanner.peek() == '?') {
    compile_filter_expression();
  } else {
    throw scanner.malformed(
        "Unexpected character in index, expected a quoted string or integer");
  }

  scanner.expect_char(']');
  if (scanner.has_next()) {
    compile_next_expression();
  }
}

void QueryParser::compile_multiple_index_expression(
    const std::vector<size_t> &indexes) {
  size_t max_max_level = max_level;
  for (size_t branch_index = 0; branch_index < indexes.size();
       ++branch_index) {
    max_max_level =
        compile_index_aux(indexes[branch_index],
                          branch_index + 1 == indexes.size(), max_max_level,
                          false);
  }
  max_level = max_max_level + 1;
}

void QueryParser::compile_index_range_expression(size_t start_index,
                                                 size_t end_index,
                                                 bool reverse) {
  size_t max_max_level = max_level;
  if (reverse) {
    for (size_t index = end_index; index > start_index; --index) {
      max_max_level = compile_index_aux(index, index == start_index + 1,
                                        max_max_level, true);
    }
  } else {
    for (size_t index = start_index; index < end_index; ++index) {
      max_max_level =
          compile_index_aux(index, index + 1 == end_index, max_max_level,
                            false);
    }
  }
  max_level = max_max_level + 1;
}

size_t QueryParser::compile_index_aux(size_t index,
                                      bool last,
                                      size_t max_max_level,
                                      bool reverse) {
  const size_t start_level = ir.current_level();
  if (reverse) {
    ir.reverse_index(index);
  } else {
    ir.index(index);
  }
  ir.down();

  scanner.mark();
  const size_t current_max_level = max_level;
  if (scanner.has_next()) {
    compile_next_expression();
  }
  max_max_level = std::max(max_max_level, max_level);
  max_level = current_max_level;

  if (!last) {
    ir.store_result();
    scanner.reset();
    const size_t end_level = ir.current_level();
    for (size_t level = start_level; level < end_level; ++level) {
      ir.up();
    }
  }

  return max_max_level;
}

void QueryParser::compile_filter_expression() {
  scanner.expect_char('?');
  scanner.expect_char('(');
  scanner.expect_char('@');
  ir.mark();
  compile_next_expression();

  while (scanner.has_next() && scanner.peek() == ' ') {
    scanner.skip_if_char(' ');
  }

  if (scanner.peek() != '=') {
    throw scanner.unsupported("Unsupported character for expression");
  }

  scanner.expect_char('=');
  scanner.expect_char('=');
  while (scanner.has_next() && scanner.peek() == ' ') {
    scanner.skip_if_char(' ');
  }

  const std::string equal_to = read_quoted_string();
  ir.expression_string_equals(equal_to);
  ir.reset();
  scanner.expect_char(')');
}

void QueryParser::create_property_ir(const std::string &property_name) {
  ir.property(property_name);
  ir.down();
  ++max_level;
}

void QueryParser::create_index_ir(size_t index) {
  ir.index(index);
  ir.down();
  ++max_level;
}

void QueryParser::create_reverse_index_ir(size_t index) {
  ir.reverse_index(index);
  ir.down();
  ++max_level;
}

std::string QueryParser::read_property() {
  const size_t start_position = scanner.get_position();
  while (scanner.has_next()) {
    const char next_char = scanner.peek();
    if (next_char == '.' || next_char == '[' || next_char == ' ') {
      break;
    }
    scanner.next();
  }
  return scanner.substring(start_position, scanner.get_position());
}

std::string QueryParser::read_quoted_string() {
  const char quote_character = scanner.next();
  if (quote_character != '\'' && quote_character != '"') {
    throw scanner.malformed("Invalid quoted string");
  }

  const size_t start_position = scanner.get_position();
  bool escaped = false;
  while (scanner.has_next()) {
    const char next_char = scanner.peek();
    if (escaped) {
      escaped = false;
    } else if (next_char == '\\') {
      escaped = true;
    } else if (next_char == quote_character) {
      break;
    }
    scanner.next();
  }

  const size_t end_position = scanner.get_position();
  scanner.expect_char(quote_character);
  return scanner.substring(start_position, end_position);
}

size_t QueryParser::read_integer(bool (*is_end_character)(char)) {
  const size_t start_position = scanner.get_position();
  while (scanner.has_next()) {
    const char next_char = scanner.peek();
    if (std::isdigit(static_cast<unsigned char>(next_char))) {
      scanner.next();
      continue;
    }
    if (is_end_character(next_char)) {
      break;
    }
    throw scanner.malformed("Invalid integer");
  }

  const std::string integer_text =
      scanner.substring(start_position, scanner.get_position());
  return static_cast<size_t>(std::stoull(integer_text));
}

QueryCompiler::QueryCompiler(const gpjson::EngineOptions &options) {
  (void)options;
}

CompiledQuery
QueryCompiler::compile(const std::string &query_src,
                       const gpjson::EngineOptions &options) const {
  (void)options;
  QueryParser parser(query_src);
  return parser.compile();
}

} // namespace gpjson::query
