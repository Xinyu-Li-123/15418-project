#pragma once

#include "gpjson/jsonpath_lexer.hpp"
#include "gpjson/query_compiler.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace gpjson {

// Stores one property access step in a parsed JSONPath query.
struct JSONPathPropertyStep {
  // Property name selected by this step.
  std::string name{};
};

// Stores one array index access step in a parsed JSONPath query.
struct JSONPathIndexStep {
  // Zero-based array index selected by this step.
  std::int32_t index = 0;
  // Whether the index is interpreted from the end of the array.
  bool is_reverse = false;
};

// Stores one array slice access step in a parsed JSONPath query.
struct JSONPathSliceStep {
  // Inclusive slice start index.
  std::int32_t start = 0;
  // Exclusive slice end index.
  std::int32_t end = 0;
  // Whether the slice is interpreted from the end of the array.
  bool is_reverse = false;
};

// Stores one equality filter expression in a parsed JSONPath query.
struct JSONPathFilterExpression {
  // Property path referenced by the filter.
  std::vector<std::string> path{};
  // String literal used in the equality comparison.
  std::string equals_value{};
};

// Parses JSONPath queries and produces the compiled query representation.
class JSONPathParser {
 public:
  // Creates a parser that consumes characters from an existing lexer.
  explicit JSONPathParser(JSONPathLexer lexer);

  // Creates a parser for one JSONPath source string.
  explicit JSONPathParser(const std::string &source);

  // Parses and compiles the full JSONPath query.
  JSONPathQuery parse();

 private:
  // Parses the next expression after the current point in the query.
  void parse_next_expression();

  // Parses a property access expression introduced by '.'.
  void parse_dot_expression();

  // Parses an index, slice, property, or filter expression introduced by '['.
  void parse_index_expression();

  // Parses one multi-index expression such as [0,2,4].
  void parse_multiple_index_expression(
      const std::vector<std::int32_t> &indexes);

  // Parses one slice expression such as [0:4] or [-3:].
  void parse_index_range_expression(std::int32_t start,
                                    std::int32_t end,
                                    bool is_reverse);

  // Parses a filter expression of the form [?(@... == "...")].
  void parse_filter_expression();

  // Parses one unquoted property name.
  std::string parse_property_name();

  // Parses one quoted string literal.
  std::string parse_quoted_string();

  // Parses one integer literal.
  std::int32_t parse_integer();

  // Emits IR for one property access step.
  void emit_property_step(const std::string &property_name);

  // Emits IR for one array index access step.
  void emit_index_step(std::int32_t index, bool is_reverse);

  // Emits IR for one array slice access step.
  void emit_slice_step(std::int32_t start,
                       std::int32_t end,
                       bool is_reverse);

  // Emits IR for one equality filter expression.
  void emit_filter_expression(const JSONPathFilterExpression &filter);

  JSONPathLexer lexer_;
  std::string source_{};
  std::vector<std::uint8_t> ir_{};
  std::uint32_t max_depth_ = 0;
  std::uint32_t num_results_ = 0;
};

}  // namespace gpjson
