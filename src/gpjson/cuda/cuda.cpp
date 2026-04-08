#include "gpjson/cuda/cuda.hpp"

#include <utility>

namespace gpjson::cuda {

DeviceArray::DeviceArray(size_t bytes) : size_bytes_(bytes) {}

DeviceArray::~DeviceArray() = default;

DeviceArray::DeviceArray(DeviceArray &&other) noexcept
    : ptr_(other.ptr_), size_bytes_(other.size_bytes_) {
  other.ptr_ = nullptr;
  other.size_bytes_ = 0;
}

DeviceArray &DeviceArray::operator=(DeviceArray &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  ptr_ = other.ptr_;
  size_bytes_ = other.size_bytes_;
  other.ptr_ = nullptr;
  other.size_bytes_ = 0;
  return *this;
}

void *DeviceArray::data() { return ptr_; }

const void *DeviceArray::data() const { return ptr_; }

size_t DeviceArray::size_bytes() const { return size_bytes_; }

} // namespace gpjson::cuda
