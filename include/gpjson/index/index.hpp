#pragma once

#include "gpjson/cuda/cuda.hpp"
#include <utility>

namespace gpjson::index {

/* RAII class that owns a device array which represents an index stored in GPU
 * memory */
class Index {
public:
  Index() = default;
  virtual ~Index() = default;

  Index(Index &) = delete;
  Index &operator=(Index &) = delete;

  Index(Index &&) noexcept = default;
  Index &operator=(Index &&) noexcept = default;

  void *data() { return this->data_.data(); }
  const void *data() const { return this->data_.data(); }

  size_t size_bytes() const { return this->data_.size_bytes(); }

protected:
  explicit Index(cuda::DeviceArray data) : data_(std::move(data)) {};

  cuda::DeviceArray data_;
};

class NewlineIndex final : public Index {
public:
  NewlineIndex() = default;

  NewlineIndex(cuda::DeviceArray data, int num_lines)
      : Index(std::move(data)), num_lines_(num_lines) {}

  NewlineIndex(NewlineIndex &) = delete;
  NewlineIndex &operator=(NewlineIndex &) = delete;

  NewlineIndex(NewlineIndex &&) noexcept = default;
  NewlineIndex &operator=(NewlineIndex &&) noexcept = default;

  int num_lines() const { return this->num_lines_; }

private:
  int num_lines_{0};
};

class StringIndex final : public Index {
public:
  StringIndex() = default;

  StringIndex(cuda::DeviceArray data) : Index(std::move(data)) {}

  StringIndex(StringIndex &) = delete;
  StringIndex &operator=(StringIndex &) = delete;

  StringIndex(StringIndex &&) noexcept = default;
  StringIndex &operator=(StringIndex &&) noexcept = default;
};

class LeveledBitmapIndex final : public Index {
public:
  LeveledBitmapIndex() = default;

  LeveledBitmapIndex(cuda::DeviceArray data, int num_levels)
      : Index(std::move(data)), num_levels_(num_levels) {}

  LeveledBitmapIndex(LeveledBitmapIndex &) = delete;
  LeveledBitmapIndex &operator=(LeveledBitmapIndex &) = delete;

  LeveledBitmapIndex(LeveledBitmapIndex &&) noexcept = default;
  LeveledBitmapIndex &operator=(LeveledBitmapIndex &&) noexcept = default;

  int num_levels() const { return this->num_levels_; }

private:
  int num_levels_{0};
};

class BuiltIndices {
public:
  BuiltIndices() = default;

  BuiltIndices(NewlineIndex newline_index, StringIndex string_index,
               LeveledBitmapIndex leveled_bitmap_index)
      : newline_index_(std::move(newline_index)),
        string_index_(std::move(string_index)),
        leveled_bitmap_index_(std::move(leveled_bitmap_index)) {}

  BuiltIndices(BuiltIndices &&) noexcept = default;
  BuiltIndices &operator=(BuiltIndices &&) noexcept = default;

  BuiltIndices(const BuiltIndices &) = delete;
  BuiltIndices &operator=(const BuiltIndices &) = delete;

  const NewlineIndex &get_newline_index() const { return this->newline_index_; }
  const StringIndex &get_string_index() const { return this->string_index_; }
  const LeveledBitmapIndex &get_leveled_bitmap_index() const {
    return this->leveled_bitmap_index_;
  }

private:
  NewlineIndex newline_index_;
  StringIndex string_index_;
  LeveledBitmapIndex leveled_bitmap_index_;
};
} // namespace gpjson::index
