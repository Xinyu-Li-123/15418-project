#pragma once

#include "gpjson/types.hpp"

namespace gpjson {

// Builds newline, string, and leveled bitmap indexes from a loaded file.
class IndexCreator {
 public:
  // Creates an index creator that uses the project's fixed build policy.
  IndexCreator() = default;

  // Builds the complete set of indexes for the loaded file.
  BuiltIndexHandle build(const LoadedFile &loaded_file);

  // Builds only the newline index.
  NewlineIndex create_newline_index(const LoadedFile &loaded_file) ;

  // Builds only the string index.
  StringIndex create_string_index(const LoadedFile &loaded_file) ;

  // Builds only the leveled bitmap index.
  LeveledBitmapIndex create_leveled_bitmap_index(
      const LoadedFile &loaded_file) const;

  // Releases resources owned by one newline index.
  void release(NewlineIndex &index);

  // Releases resources owned by one string index.
  void release(StringIndex &index) ;

  // Releases resources owned by one leveled bitmap index.
  void release(LeveledBitmapIndex &index) ;

  // Releases resources owned by a complete built index.
  void release(const BuiltIndexHandle &index);
};

}  // namespace gpjson
