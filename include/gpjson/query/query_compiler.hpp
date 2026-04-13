#pragma once

#include "gpjson/query/error.hpp"
#include "gpjson/query/query.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace gpjson {
struct EngineOptions;
}

namespace gpjson::query {

class IRByteOutputBuffer {
public:
  void write_opcode(QueryOpcode opcode);
  void write_byte(std::byte value);
  void write_varint(size_t value);
  void write_string(const std::string &value);
  const std::vector<std::byte> &bytes() const;

private:
  std::vector<std::byte> bytes_;
};

class IRBuilder {
public:
  explicit IRBuilder(IRByteOutputBuffer &buffer);

  void property(const std::string &name);
  void index(size_t index);
  void reverse_index(size_t index);
  void down();
  void up();
  void mark();
  void reset();
  void store_result();
  void end();
  void expression_string_equals(const std::string &value);
  size_t current_level() const;
  size_t num_result_stores() const;

private:
  IRByteOutputBuffer &buffer_;
  size_t current_level_{0};
  size_t num_result_stores_{0};
};

class QueryScanner {
public:
  explicit QueryScanner(std::string query);

  bool has_next() const;
  char next();
  char peek() const;
  void expect_char(char expected);
  bool skip_if_char(char expected);
  void test_digit() const;
  void mark();
  void reset();
  size_t get_position() const;
  std::string substring(size_t start, size_t end) const;
  error::query::MalformedQueryError malformed(const std::string &message) const;
  error::query::UnsupportedQueryError
  unsupported(const std::string &message) const;

private:
  std::string query;
  size_t position{static_cast<size_t>(-1)};
  std::vector<size_t> marked_positions;
};

class QueryParser {
public:
  explicit QueryParser(const std::string &query);
  QueryParser(const QueryParser &) = delete;
  QueryParser &operator=(const QueryParser &) = delete;

  CompiledQuery compile();

private:
  void compile_next_expression();
  void compile_dot_expression();
  void compile_index_expression();
  void compile_multiple_index_expression(const std::vector<size_t> &indexes);
  void compile_index_range_expression(size_t start_index,
                                      size_t end_index,
                                      bool reverse);
  size_t compile_index_aux(size_t index,
                           bool last,
                           size_t max_max_level,
                           bool reverse);
  void compile_filter_expression();
  void create_property_ir(const std::string &property_name);
  void create_index_ir(size_t index);
  void create_reverse_index_ir(size_t index);
  std::string read_property();
  std::string read_quoted_string();
  size_t read_integer(bool (*is_end_character)(char));

  QueryScanner scanner;
  IRByteOutputBuffer ir_buffer;
  IRBuilder ir;
  size_t max_level{0};
  std::string query_text{};
};

class QueryCompiler {
public:
  QueryCompiler(const gpjson::EngineOptions &options);

  CompiledQuery compile(const std::string &query_src,
                        const gpjson::EngineOptions &options) const;
};

} // namespace gpjson::query
