#include "gpjson/index/index.hpp"

namespace gpjson::index {

const NewlineIndex &BuiltIndices::get_newline_index() const {
  return newline_index_;
}

const StringIndex &BuiltIndices::get_string_index() const {
  return string_index_;
}

const LeveledBitmapIndex &BuiltIndices::get_leveled_bitmap_index() const {
  return leveled_bitmap_index_;
}

} // namespace gpjson::index
