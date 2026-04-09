#pragma once

#include <cstddef>

namespace gpjson::cuda {

/* RAII handler of an array on GPU global memory */
class DeviceArray {
public:
  DeviceArray() = default;
  explicit DeviceArray(size_t bytes);
  ~DeviceArray();

  DeviceArray(DeviceArray &&other) noexcept;
  DeviceArray &operator=(DeviceArray &&other) noexcept;

  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;

  void *data();
  const void *data() const;
  size_t size_bytes() const;

private:
  void *ptr_{nullptr};
  size_t size_bytes_{0};
};

} // namespace gpjson::cuda
