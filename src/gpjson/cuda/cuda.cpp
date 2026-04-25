#include "gpjson/cuda/cuda.hpp"
#include "gpjson/cuda/error.hpp"

#include <cuda_runtime_api.h>

#include <string>

namespace gpjson::cuda {

namespace {

std::string cuda_error_message(const char *context, cudaError_t status) {
  std::string message = context;
  message += ": ";
  message += cudaGetErrorString(status);
  return message;
}

void validate_copy_args(const void *host_ptr, size_t bytes,
                        size_t allocation_bytes, const char *context) {
  if (bytes > allocation_bytes) {
    throw error::cuda::CudaError(std::string(context) +
                                 ": requested byte count exceeds allocation");
  }
  if (bytes > 0 && host_ptr == nullptr) {
    throw error::cuda::CudaError(std::string(context) +
                                 ": host pointer is null");
  }
}

} // namespace

DeviceArray::DeviceArray(size_t bytes) : size_bytes_(bytes) {
  if (bytes == 0) {
    return;
  }

  check(cudaMalloc(&ptr_, bytes), "cudaMalloc");
}

DeviceArray::~DeviceArray() {
  if (ptr_ != nullptr) {
    cudaFree(ptr_);
  }
}

DeviceArray::DeviceArray(DeviceArray &&other) noexcept
    : ptr_(other.ptr_), size_bytes_(other.size_bytes_) {
  other.ptr_ = nullptr;
  other.size_bytes_ = 0;
}

DeviceArray &DeviceArray::operator=(DeviceArray &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  // If I currently points to some device array, free my array before taking
  // other's array
  if (ptr_ != nullptr) {
    cudaFree(ptr_);
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

bool DeviceArray::empty() const { return size_bytes_ == 0; }

void DeviceArray::copy_from_host(const void *src, size_t bytes) {
  validate_copy_args(src, bytes, size_bytes_, "copy_from_host");
  if (bytes == 0) {
    return;
  }

  check(cudaMemcpy(ptr_, src, bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy host to device");
}

void DeviceArray::copy_to_host(void *dst, size_t bytes) const {
  validate_copy_args(dst, bytes, size_bytes_, "copy_to_host");
  if (bytes == 0) {
    return;
  }

  check(cudaMemcpy(dst, ptr_, bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy device to host");
}

void DeviceArray::memset(int value) {
  if (size_bytes_ == 0) {
    return;
  }

  check(cudaMemset(ptr_, value, size_bytes_), "cudaMemset");
}

void check(cudaError_t status, const char *context) {
  if (status != cudaSuccess) {
    throw error::cuda::CudaError(cuda_error_message(context, status));
  }
}

bool device_available() {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaSuccess) {
    return count > 0;
  }
  return false;
}

void synchronize_and_check() {
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  check(cudaGetLastError(), "cudaGetLastError");
}

} // namespace gpjson::cuda
