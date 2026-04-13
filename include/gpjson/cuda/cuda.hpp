#pragma once

#include <cuda_runtime_api.h>

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
  bool empty() const;

  void copy_from_host(const void *src, size_t bytes);
  void copy_to_host(void *dst, size_t bytes) const;

  void memset(int value);

private:
  void *ptr_{nullptr};
  size_t size_bytes_{0};
};

void check(cudaError_t status, const char *context);
bool device_available();
void synchronize();

} // namespace gpjson::cuda
