#pragma once

namespace gpjson::index {
// abstract class
class Index {};

// RAII class, owns a gpjson::cuda::DeviceArray
class NewlineIndex : Index {};

class EscapeCarryIndex : Index {};

class EscapeIndex : Index {};

class QuoteCarryIndex : Index {};

class QuoteIndex : Index {};

class StringCarryIndex : Index {};

class StringIndex : Index {};

class LeveledBitmapIndex : Index {};

// RAII class for a collection of indices we will use for our executor.
// Owns the indices
class BuiltIndices {
public:
  NewlineIndex get_newline_index();
  StringIndex get_string_index();
  LeveledBitmapIndex get_leveled_bitmap_index();
}
} // namespace gpjson::index
