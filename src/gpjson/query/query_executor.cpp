#include "gpjson/query/query_executor.hpp"

#include "gpjson/query/error.hpp"
#include "gpjson/query/ir_input_buffer.hpp"

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gpjson::query {

namespace {

enum class JsonNodeType { OBJECT, ARRAY, STRING, NUMBER, BOOLEAN, NULL_VALUE };

struct JsonNode {
  JsonNodeType type{JsonNodeType::NULL_VALUE};
  size_t start_offset{0};
  size_t end_offset{0};
  std::vector<std::pair<std::string, std::unique_ptr<JsonNode>>> object_members;
  std::vector<std::unique_ptr<JsonNode>> array_elements;
};

class JsonLineParser {
public:
  explicit JsonLineParser(std::string_view input) : input_(input) {}

  std::unique_ptr<JsonNode> parse() {
    skip_whitespace();
    auto node = parse_value();
    skip_whitespace();
    if (position_ != input_.size()) {
      throw execution_error("Unexpected trailing characters in JSON line");
    }
    return node;
  }

private:
  std::unique_ptr<JsonNode> parse_value() {
    skip_whitespace();
    if (position_ >= input_.size()) {
      throw execution_error("Unexpected end of JSON input");
    }

    switch (input_[position_]) {
    case '{':
      return parse_object();
    case '[':
      return parse_array();
    case '"':
      return parse_string_node();
    case 't':
      return parse_literal("true", JsonNodeType::BOOLEAN);
    case 'f':
      return parse_literal("false", JsonNodeType::BOOLEAN);
    case 'n':
      return parse_literal("null", JsonNodeType::NULL_VALUE);
    default:
      if (input_[position_] == '-' ||
          std::isdigit(static_cast<unsigned char>(input_[position_]))) {
        return parse_number();
      }
      throw execution_error("Unexpected character in JSON value");
    }
  }

  std::unique_ptr<JsonNode> parse_object() {
    auto node = std::make_unique<JsonNode>();
    node->type = JsonNodeType::OBJECT;
    node->start_offset = position_;

    ++position_;
    skip_whitespace();
    if (position_ < input_.size() && input_[position_] == '}') {
      ++position_;
      node->end_offset = position_;
      return node;
    }

    while (true) {
      skip_whitespace();
      if (position_ >= input_.size() || input_[position_] != '"') {
        throw execution_error("Expected string key in object");
      }
      const std::string key = parse_string_literal();
      skip_whitespace();
      expect_char(':');
      auto value = parse_value();
      node->object_members.emplace_back(key, std::move(value));

      skip_whitespace();
      if (position_ < input_.size() && input_[position_] == ',') {
        ++position_;
        continue;
      }
      break;
    }

    expect_char('}');
    node->end_offset = position_;
    return node;
  }

  std::unique_ptr<JsonNode> parse_array() {
    auto node = std::make_unique<JsonNode>();
    node->type = JsonNodeType::ARRAY;
    node->start_offset = position_;

    ++position_;
    skip_whitespace();
    if (position_ < input_.size() && input_[position_] == ']') {
      ++position_;
      node->end_offset = position_;
      return node;
    }

    while (true) {
      node->array_elements.push_back(parse_value());
      skip_whitespace();
      if (position_ < input_.size() && input_[position_] == ',') {
        ++position_;
        continue;
      }
      break;
    }

    expect_char(']');
    node->end_offset = position_;
    return node;
  }

  std::unique_ptr<JsonNode> parse_string_node() {
    auto node = std::make_unique<JsonNode>();
    node->type = JsonNodeType::STRING;
    node->start_offset = position_;
    parse_string_literal();
    node->end_offset = position_;
    return node;
  }

  std::unique_ptr<JsonNode> parse_number() {
    auto node = std::make_unique<JsonNode>();
    node->type = JsonNodeType::NUMBER;
    node->start_offset = position_;

    if (input_[position_] == '-') {
      ++position_;
    }
    parse_digits();
    if (position_ < input_.size() && input_[position_] == '.') {
      ++position_;
      parse_digits();
    }
    if (position_ < input_.size() &&
        (input_[position_] == 'e' || input_[position_] == 'E')) {
      ++position_;
      if (position_ < input_.size() &&
          (input_[position_] == '+' || input_[position_] == '-')) {
        ++position_;
      }
      parse_digits();
    }

    node->end_offset = position_;
    return node;
  }

  std::unique_ptr<JsonNode> parse_literal(std::string_view literal,
                                          JsonNodeType type) {
    auto node = std::make_unique<JsonNode>();
    node->type = type;
    node->start_offset = position_;
    if (input_.substr(position_, literal.size()) != literal) {
      throw execution_error("Invalid literal in JSON value");
    }
    position_ += literal.size();
    node->end_offset = position_;
    return node;
  }

  std::string parse_string_literal() {
    expect_char('"');
    std::string value;
    while (position_ < input_.size()) {
      const char current = input_[position_++];
      if (current == '"') {
        return value;
      }
      if (current != '\\') {
        value.push_back(current);
        continue;
      }

      if (position_ >= input_.size()) {
        throw execution_error("Invalid escape sequence in string");
      }

      const char escaped = input_[position_++];
      switch (escaped) {
      case '"':
      case '\\':
      case '/':
        value.push_back(escaped);
        break;
      case 'b':
        value.push_back('\b');
        break;
      case 'f':
        value.push_back('\f');
        break;
      case 'n':
        value.push_back('\n');
        break;
      case 'r':
        value.push_back('\r');
        break;
      case 't':
        value.push_back('\t');
        break;
      case 'u': {
        if (position_ + 4 > input_.size()) {
          throw execution_error("Invalid unicode escape in string");
        }
        unsigned int codepoint = 0;
        for (int digit = 0; digit < 4; ++digit) {
          const char hex = input_[position_++];
          codepoint <<= 4U;
          if (hex >= '0' && hex <= '9') {
            codepoint |= static_cast<unsigned int>(hex - '0');
          } else if (hex >= 'a' && hex <= 'f') {
            codepoint |= static_cast<unsigned int>(10 + hex - 'a');
          } else if (hex >= 'A' && hex <= 'F') {
            codepoint |= static_cast<unsigned int>(10 + hex - 'A');
          } else {
            throw execution_error("Invalid unicode escape in string");
          }
        }
        append_utf8(value, codepoint);
        break;
      }
      default:
        throw execution_error("Invalid escape sequence in string");
      }
    }

    throw execution_error("Unterminated string in JSON input");
  }

  void parse_digits() {
    if (position_ >= input_.size() ||
        !std::isdigit(static_cast<unsigned char>(input_[position_]))) {
      throw execution_error("Expected digit in JSON number");
    }
    while (position_ < input_.size() &&
           std::isdigit(static_cast<unsigned char>(input_[position_]))) {
      ++position_;
    }
  }

  void skip_whitespace() {
    while (position_ < input_.size() &&
           std::isspace(static_cast<unsigned char>(input_[position_]))) {
      ++position_;
    }
  }

  void expect_char(char expected) {
    if (position_ >= input_.size() || input_[position_] != expected) {
      throw execution_error("Unexpected JSON character");
    }
    ++position_;
  }

  static void append_utf8(std::string &output, unsigned int codepoint) {
    if (codepoint <= 0x7F) {
      output.push_back(static_cast<char>(codepoint));
      return;
    }
    if (codepoint <= 0x7FF) {
      output.push_back(static_cast<char>(0xC0 | ((codepoint >> 6U) & 0x1F)));
      output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
      return;
    }
    output.push_back(static_cast<char>(0xE0 | ((codepoint >> 12U) & 0x0F)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 6U) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  }

  static error::query::QueryExecutionError execution_error(
      const std::string &message) {
    return error::query::QueryExecutionError(message);
  }

  std::string_view input_;
  size_t position_{0};
};

struct DecodedInstruction {
  QueryOpcode opcode{QueryOpcode::END};
  std::string string_operand;
  size_t integer_operand{0};
};

} // namespace

namespace {

struct DecodedResult {
  std::optional<size_t> start;
  std::optional<size_t> end;
  std::optional<std::string> json_text;
};

const JsonNode *find_object_member(const JsonNode *node, const std::string &key) {
  if (node == nullptr || node->type != JsonNodeType::OBJECT) {
    return nullptr;
  }
  for (const auto &member : node->object_members) {
    if (member.first == key) {
      return member.second.get();
    }
  }
  return nullptr;
}

const JsonNode *find_array_index(const JsonNode *node, size_t index) {
  if (node == nullptr || node->type != JsonNodeType::ARRAY ||
      index >= node->array_elements.size()) {
    return nullptr;
  }
  return node->array_elements[index].get();
}

const JsonNode *find_reverse_array_index(const JsonNode *node, size_t index) {
  if (node == nullptr || node->type != JsonNodeType::ARRAY || index == 0 ||
      index > node->array_elements.size()) {
    return nullptr;
  }
  return node->array_elements[node->array_elements.size() - index].get();
}

std::vector<DecodedResult>
execute_query_ir(const CompiledQuery &compiled_query,
                 const JsonNode &root,
                 std::string_view line,
                 size_t global_line_start) {
  struct MarkState {
    std::vector<const JsonNode *> node_stack;
    std::optional<const JsonNode *> pending_node;
    bool predicate_matched{true};
  };

  std::vector<DecodedResult> results(compiled_query.num_result_slots());
  std::vector<const JsonNode *> node_stack{&root};
  std::optional<const JsonNode *> pending_node;
  std::vector<MarkState> marks;
  size_t result_slot = 0;

  IRByteInputBuffer input(compiled_query.ir_bytes().data(),
                          compiled_query.ir_bytes().size());
  while (input.has_next()) {
    const QueryOpcode opcode = input.read_opcode();
    switch (opcode) {
    case QueryOpcode::END:
      return results;

    case QueryOpcode::STORE_RESULT: {
      if (result_slot >= results.size()) {
        throw error::query::QueryExecutionError(
            "Query IR produced more results than expected");
      }
      const JsonNode *node = node_stack.empty() ? nullptr : node_stack.back();
      if (node != nullptr) {
        results[result_slot].start = global_line_start + node->start_offset;
        results[result_slot].end = global_line_start + node->end_offset;
        results[result_slot].json_text =
            std::string(line.substr(node->start_offset,
                                    node->end_offset - node->start_offset));
      }
      ++result_slot;
      break;
    }

    case QueryOpcode::MOVE_UP:
      if (node_stack.size() > 1) {
        node_stack.pop_back();
      }
      pending_node.reset();
      break;

    case QueryOpcode::MOVE_DOWN:
      node_stack.push_back(pending_node.value_or(nullptr));
      pending_node.reset();
      break;

    case QueryOpcode::MOVE_TO_KEY:
      pending_node = find_object_member(node_stack.back(), input.read_string());
      break;

    case QueryOpcode::MOVE_TO_INDEX:
      pending_node = find_array_index(node_stack.back(), input.read_varint());
      break;

    case QueryOpcode::MOVE_TO_INDEX_REVERSE:
      pending_node =
          find_reverse_array_index(node_stack.back(), input.read_varint());
      break;

    case QueryOpcode::MARK_POSITION:
      marks.push_back(MarkState{node_stack, pending_node, true});
      break;

    case QueryOpcode::RESET_POSITION: {
      if (marks.empty()) {
        throw error::query::QueryExecutionError("Unbalanced RESET_POSITION");
      }
      MarkState mark = std::move(marks.back());
      marks.pop_back();
      node_stack = std::move(mark.node_stack);
      pending_node = std::move(mark.pending_node);
      if (!mark.predicate_matched && !node_stack.empty()) {
        node_stack.back() = nullptr;
      }
      break;
    }

    case QueryOpcode::EXPRESSION_STRING_EQUALS: {
      if (marks.empty()) {
        throw error::query::QueryExecutionError(
            "EXPRESSION_STRING_EQUALS used outside a filter");
      }
      const std::string expected = input.read_string();
      const JsonNode *node = node_stack.empty() ? nullptr : node_stack.back();
      marks.back().predicate_matched =
          node != nullptr &&
          line.substr(node->start_offset, node->end_offset - node->start_offset) ==
              expected;
      break;
    }
    }
  }

  return results;
}

std::string partition_to_string(const file::PartitionView &partition_view) {
  if (partition_view.bytes() == nullptr || partition_view.size_bytes() == 0) {
    return {};
  }
  return {reinterpret_cast<const char *>(partition_view.bytes()),
          partition_view.size_bytes()};
}

LineQueryResult make_line_result(
                        const std::vector<DecodedResult> &decoded_results) {
  LineQueryResult line_result;
  for (const auto &result : decoded_results) {
    line_result.add_offset(
        QueryOffset{result.start, result.end, result.json_text});
  }
  return line_result;
}

} // namespace

QueryExecutor::QueryExecutor(const gpjson::EngineOptions &options) {
  (void)options;
}

BatchQueryResult
QueryExecutor::execute_batch(const BatchCompiledQuery &compiled_queries,
                             const file::PartitionView &partition_view,
                             const index::BuiltIndices &built_indices) const {
  (void)built_indices;

  BatchQueryResult batch_result(compiled_queries.size());
  for (size_t query_index = 0; query_index < compiled_queries.size();
       ++query_index) {
    batch_result.set_query_text(query_index,
                                compiled_queries.queries()[query_index]
                                    .query_text());
  }

  const std::string partition_text = partition_to_string(partition_view);
  size_t line_start = 0;
  while (line_start < partition_text.size()) {
    size_t line_end = partition_text.find('\n', line_start);
    if (line_end == std::string::npos) {
      line_end = partition_text.size();
    }

    const std::string_view line(partition_text.data() + line_start,
                                line_end - line_start);
    if (!line.empty()) {
      JsonLineParser parser(line);
      std::unique_ptr<JsonNode> root = parser.parse();
      for (size_t query_index = 0; query_index < compiled_queries.size();
           ++query_index) {
        batch_result.add_line_result(
            query_index,
            make_line_result(execute_query_ir(compiled_queries.queries()[query_index],
                                              *root, line,
                                              partition_view.global_start_offset() +
                                                  line_start)));
      }
    }

    line_start = line_end + 1;
  }

  return batch_result;
}

} // namespace gpjson::query
