#include "gpjson/query/error.hpp"
#include "gpjson/query/ir_input_buffer.hpp"
#include "gpjson/query/query_compiler.hpp"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

namespace {

using gpjson::query::CompiledQuery;
using gpjson::query::IRByteInputBuffer;
using gpjson::query::QueryCompiler;
using gpjson::query::QueryOpcode;

} // namespace

namespace gpjson {
struct EngineOptions {};
} // namespace gpjson

namespace {

struct DecodedInstruction {
  QueryOpcode opcode{QueryOpcode::END};
  std::string string_operand;
  size_t integer_operand{0};
};

std::vector<DecodedInstruction> decode_ir(const CompiledQuery &query) {
  std::vector<DecodedInstruction> instructions;
  IRByteInputBuffer input(query.ir_bytes().data(), query.ir_bytes().size());
  while (input.has_next()) {
    DecodedInstruction instruction;
    instruction.opcode = input.read_opcode();
    switch (instruction.opcode) {
    case QueryOpcode::MOVE_TO_KEY:
    case QueryOpcode::EXPRESSION_STRING_EQUALS:
      instruction.string_operand = input.read_string();
      break;
    case QueryOpcode::MOVE_TO_INDEX:
    case QueryOpcode::MOVE_TO_INDEX_REVERSE:
      instruction.integer_operand = input.read_varint();
      break;
    default:
      break;
    }
    instructions.push_back(std::move(instruction));
    if (instructions.back().opcode == QueryOpcode::END) {
      break;
    }
  }
  return instructions;
}

TEST(QueryCompilerTest, CompilesSimplePropertyQuery) {
  gpjson::EngineOptions options{};
  QueryCompiler compiler(options);

  const CompiledQuery compiled = compiler.compile("$.name", options);
  const auto instructions = decode_ir(compiled);

  ASSERT_EQ(compiled.query_text(), "$.name");
  ASSERT_EQ(compiled.max_depth(), 1U);
  ASSERT_EQ(compiled.num_result_slots(), 1U);
  ASSERT_EQ(instructions.size(), 4U);

  EXPECT_EQ(instructions[0].opcode, QueryOpcode::MOVE_TO_KEY);
  EXPECT_EQ(instructions[0].string_operand, "name");
  EXPECT_EQ(instructions[1].opcode, QueryOpcode::MOVE_DOWN);
  EXPECT_EQ(instructions[2].opcode, QueryOpcode::STORE_RESULT);
  EXPECT_EQ(instructions[3].opcode, QueryOpcode::END);
}

TEST(QueryCompilerTest, CompilesArrayRangeIntoTwoResultSlots) {
  gpjson::EngineOptions options{};
  QueryCompiler compiler(options);

  const CompiledQuery compiled = compiler.compile("$.arr[0:2]", options);
  const auto instructions = decode_ir(compiled);

  ASSERT_EQ(compiled.max_depth(), 2U);
  ASSERT_EQ(compiled.num_result_slots(), 2U);

  ASSERT_EQ(instructions.size(), 10U);
  EXPECT_EQ(instructions[0].opcode, QueryOpcode::MOVE_TO_KEY);
  EXPECT_EQ(instructions[0].string_operand, "arr");
  EXPECT_EQ(instructions[1].opcode, QueryOpcode::MOVE_DOWN);
  EXPECT_EQ(instructions[2].opcode, QueryOpcode::MOVE_TO_INDEX);
  EXPECT_EQ(instructions[2].integer_operand, 0U);
  EXPECT_EQ(instructions[3].opcode, QueryOpcode::MOVE_DOWN);
  EXPECT_EQ(instructions[4].opcode, QueryOpcode::STORE_RESULT);
  EXPECT_EQ(instructions[5].opcode, QueryOpcode::MOVE_UP);
  EXPECT_EQ(instructions[6].opcode, QueryOpcode::MOVE_TO_INDEX);
  EXPECT_EQ(instructions[6].integer_operand, 1U);
  EXPECT_EQ(instructions[7].opcode, QueryOpcode::MOVE_DOWN);
  EXPECT_EQ(instructions[8].opcode, QueryOpcode::STORE_RESULT);
  EXPECT_EQ(instructions[9].opcode, QueryOpcode::END);
}

TEST(QueryCompilerTest, CompilesFilterExpression) {
  gpjson::EngineOptions options{};
  QueryCompiler compiler(options);

  const CompiledQuery compiled =
      compiler.compile("$.user.lang[?(@ == \"en\")]", options);
  const auto instructions = decode_ir(compiled);

  ASSERT_EQ(compiled.max_depth(), 2U);
  ASSERT_EQ(compiled.num_result_slots(), 1U);

  ASSERT_EQ(instructions.size(), 9U);
  EXPECT_EQ(instructions[0].opcode, QueryOpcode::MOVE_TO_KEY);
  EXPECT_EQ(instructions[0].string_operand, "user");
  EXPECT_EQ(instructions[1].opcode, QueryOpcode::MOVE_DOWN);
  EXPECT_EQ(instructions[2].opcode, QueryOpcode::MOVE_TO_KEY);
  EXPECT_EQ(instructions[2].string_operand, "lang");
  EXPECT_EQ(instructions[3].opcode, QueryOpcode::MOVE_DOWN);
  EXPECT_EQ(instructions[4].opcode, QueryOpcode::MARK_POSITION);
  EXPECT_EQ(instructions[5].opcode, QueryOpcode::EXPRESSION_STRING_EQUALS);
  EXPECT_EQ(instructions[5].string_operand, "\"en\"");
  EXPECT_EQ(instructions[6].opcode, QueryOpcode::RESET_POSITION);
  EXPECT_EQ(instructions[7].opcode, QueryOpcode::STORE_RESULT);
  EXPECT_EQ(instructions[8].opcode, QueryOpcode::END);
}

TEST(QueryCompilerTest, RejectsMalformedQueries) {
  gpjson::EngineOptions options{};
  QueryCompiler compiler(options);

  EXPECT_THROW(compiler.compile("$.", options),
               gpjson::error::query::MalformedQueryError);
  EXPECT_THROW(compiler.compile("$.arr[-0]", options),
               gpjson::error::query::MalformedQueryError);
}

TEST(QueryCompilerTest, RejectsUnsupportedQueries) {
  gpjson::EngineOptions options{};
  QueryCompiler compiler(options);

  EXPECT_THROW(compiler.compile("$..name", options),
               gpjson::error::query::UnsupportedQueryError);
  EXPECT_THROW(compiler.compile("$.arr[*]", options),
               gpjson::error::query::UnsupportedQueryError);
}

} // namespace
