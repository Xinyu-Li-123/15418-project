#pragma once

#include "gpjson/cuda/cuda.hpp"

#define private public
#include "gpjson/file/file.hpp"
#undef private

#include "gpjson/index/index.hpp"
#include "gpjson/index/index_builder.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace gpjson::test::index {

inline constexpr int kMaxDepth = 8;

inline long word_from_bits(uint64_t bits) {
  long word = 0;
  static_assert(sizeof(word) == sizeof(bits));
  std::memcpy(&word, &bits, sizeof(word));
  return word;
}

inline uint64_t bit_for(size_t offset) { return uint64_t{1} << (offset % 64); }

inline size_t level_size_for(size_t file_size) {
  return (file_size + 64 - 1) / 64;
}

/* Create a PartitionView from a cpp string */
class StaticPartition {
public:
  explicit StaticPartition(std::string data) : data_(std::move(data)) {
    view_.partition_id_ = 0;
    view_.global_start_offset_ = 0;
    view_.global_end_offset_ = data_.size();
    view_.data_ = reinterpret_cast<const std::byte *>(data_.data());
  }

  const file::PartitionView &view() const { return view_; }

  const std::string &data() const { return data_; }

private:
  std::string data_;
  file::PartitionView view_;
};

inline std::string ld_json_fixture() {
  return std::string{
      "{\"id\":1,\"name\":\"Ada\",\"tags\":[\"gpu\",\"json\"],\"active\":true}"
      "\n"
      "{\"id\":2,\"name\":\"Bob \\\"The Builder\\\"\","
      "\"note\":\"literal { braces } and [brackets] stay strings\"}"
      "\n"
      "{\"id\":3,\"nested\":{\"arr\":[1,{\"x\":\"y,z\"}],\"empty\":{}},"
      "\"path\":\"C:\\\\tmp\\\\file\"}"};
}

inline std::vector<long> copy_index_words(const gpjson::index::Index &index) {
  std::vector<long> words(index.size_bytes() / sizeof(long), 0);
  if (index.size_bytes() > 0) {
    cuda::check(cudaMemcpy(words.data(), index.data(), index.size_bytes(),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy index to host");
  }
  return words;
}

inline gpjson::index::IndexBuilderOptions small_builder_options() {
  return gpjson::index::IndexBuilderOptions{
      gpjson::index::IndexBuilderType::UNCOMBINED, 0, 1, 64, 1, 64,
  };
}

} // namespace gpjson::test::index
