#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpjson {

// Reports a generic JSONPath parse failure.
class JSONPathError : public std::runtime_error {
 public:
  // Creates an error with a message describing the parse failure.
  explicit JSONPathError(const std::string &message);
};

// Reports a JSONPath feature that this project does not support.
class UnsupportedJSONPathError : public JSONPathError {
 public:
  // Creates an error with a message describing the unsupported feature.
  explicit UnsupportedJSONPathError(const std::string &message);
};

// Scans a JSONPath string one character at a time for the parser.
class JSONPathLexer {
 public:
  // Creates a lexer for one JSONPath source string.
  explicit JSONPathLexer(std::string source);

  // Consumes and returns the next character.
  char next();

  // Returns the next character without consuming it.
  char peek() ;

  // Returns whether another character is available.
  bool has_next() ;

  // Returns the current character position.
  std::size_t position();

  // Returns the substring between two parser positions.
  std::string substring(std::size_t start, std::size_t end);

  // Consumes the next character and checks that it matches the expectation.
  void expect_char(char expected);

  // Checks that the next character is a digit.
  void expect_digit();

  // Consumes the next character only if it matches the expectation.
  bool skip_if_char(char expected);

  // Builds an error referring to the current character position.
  JSONPathError error(const std::string &message) ;

  // Builds an error referring to the next character position.
  JSONPathError error_next(const std::string &message) ;

  // Builds an unsupported-feature error for the current character position.
  UnsupportedJSONPathError unsupported(const std::string &message) ;

  // Builds an unsupported-feature error for the next character position.
  UnsupportedJSONPathError unsupported_next(const std::string &message) ;

  // Saves the current lexer position for later backtracking.
  void mark();

  // Restores the most recently saved lexer position.
  void reset();

 private:
  std::string source_{};
  std::ptrdiff_t position_ = -1;
  std::vector<std::ptrdiff_t> marks_{};
};

}  // namespace gpjson
