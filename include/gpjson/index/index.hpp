#pragma once

#include "gpjson/cuda/cuda.hpp"

namespace gpjson::index {

class Index {
public:
  virtual ~Index() = default;

protected:
  Index() = default;
};

class NewlineIndex final : public Index {};

class EscapeCarryIndex final : public Index {};

class EscapeIndex final : public Index {};

class QuoteCarryIndex final : public Index {};

class QuoteIndex final : public Index {};

class StringCarryIndex final : public Index {};

class StringIndex final : public Index {};

class LeveledBitmapIndex final : public Index {};

class BuiltIndices {
public:
  const NewlineIndex &get_newline_index() const;
  const StringIndex &get_string_index() const;
  const LeveledBitmapIndex &get_leveled_bitmap_index() const;

private:
  NewlineIndex newline_index_;
  StringIndex string_index_;
  LeveledBitmapIndex leveled_bitmap_index_;
};
} // namespace gpjson::index
