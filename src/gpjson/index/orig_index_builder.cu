/*
 * Index builders from the original java codebase
 */

#include "gpjson/cuda/cuda.hpp"
#include "gpjson/error/common.hpp"
#include "gpjson/file/file.hpp"
#include "gpjson/index/index.hpp"
#include "gpjson/index/index_builder.hpp"
#include "gpjson/index/kernels/orig.cuh"
#include "gpjson/log/log.hpp"
namespace gpjson::index::kernels::orig {
__global__ void newline_count_index(const char *file, int fileSize,
                                    int *newlineCountIndex);

__global__ void newline_index(const char *file, int fileSize,
                              int *newlineCountIndex, long *newlineIndex);
} // namespace gpjson::index::kernels::orig

namespace gpjson::index {
namespace {

class OrigIndexBuilderContext {
public:
  OrigIndexBuilderContext(int grid_size, int block_size,
                          int reduction_grid_size, int reduction_block_size,
                          int max_depth,
                          const file::PartitionView &partition_view)
      : grid_size(grid_size), block_size(block_size),
        reduction_grid_size(reduction_grid_size),
        reduction_block_size(reduction_block_size), max_depth(max_depth),
        file_size(partition_view.size_bytes()),
        device_file_(static_cast<const char *>(partition_view.device_bytes())) {
    if (partition_view.size_bytes() > 0 && device_file_ == nullptr) {
      throw error::common::ImplementationError(
          "Index building requires a GPU-resident partition buffer");
    }
  }

  OrigIndexBuilderContext(const IndexBuilderOptions &options, int max_depth,
                          const file::PartitionView &partition_view)
      : OrigIndexBuilderContext(
            options.grid_size, options.block_size, options.reduction_grid_size,
            options.reduction_block_size, max_depth, partition_view) {}

  int num_cuda_threads() const { return grid_size * block_size; }
  int level_size() const {
    // TODO: Why? This is copied from java codebase, but 64 seems random
    return (this->file_size + 64 - 1) / 64;
  }

  const char *device_file() const { return device_file_; }

  const int grid_size;
  const int block_size;
  const int reduction_grid_size;
  const int reduction_block_size;
  const int max_depth;
  const int file_size;

private:
  const char *device_file_{nullptr};
};

template <typename T>
void copy_scalar_to_device(cuda::DeviceArray &array, size_t index,
                           const T &value) {
  cuda::check(cudaMemcpy(array.as<T>() + index, &value, sizeof(T),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy scalar host to device");
}

template <typename T>
T copy_scalar_from_device(const cuda::DeviceArray &array, size_t index) {
  T value{};
  cuda::check(cudaMemcpy(&value, array.as<const T>() + index, sizeof(T),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy scalar device to host");
  return value;
}

struct NewlineStringIndices {
  NewlineIndex newline_index;
  StringIndex string_index;

  NewlineStringIndices() = delete;

  NewlineStringIndices(NewlineIndex newline_index, StringIndex string_index)
      : newline_index(std::move(newline_index)),
        string_index(std::move(string_index)) {};
};

NewlineStringIndices
create_newline_and_string_index(bool combined,
                                const OrigIndexBuilderContext &ctx) {
  LogInfo("Create newline and string index, combined=%d", combined);
  cuda::DeviceArray string_index_mem(ctx.level_size() * sizeof(long));
  cuda::DeviceArray string_carry_index_mem(ctx.num_cuda_threads() *
                                           sizeof(char));
  cuda::DeviceArray newline_count_index_mem(ctx.num_cuda_threads() *
                                            sizeof(int));
  cuda::DeviceArray newline_index_offset_mem((ctx.num_cuda_threads() + 1) *
                                             sizeof(int));
  if (combined) {
    kernels::orig::combined_escape_carry_newline_count_index<<<
        ctx.grid_size, ctx.block_size>>>(ctx.device_file(), ctx.file_size,
                                         string_carry_index_mem.as<char>(),
                                         newline_count_index_mem.as<int>());
  } else {
    kernels::orig::newline_count_index<<<ctx.grid_size, ctx.block_size>>>(
        ctx.device_file(), ctx.file_size, newline_count_index_mem.as<int>());
    kernels::orig::escape_carry_index<<<ctx.grid_size, ctx.block_size>>>(
        ctx.device_file(), ctx.file_size, string_carry_index_mem.as<char>());
  }

  cuda::DeviceArray int_sum_base_mem(ctx.reduction_grid_size *
                                     ctx.reduction_block_size * sizeof(int));
  const int num_reduction_cuda_threads =
      ctx.reduction_grid_size * ctx.reduction_block_size;
  kernels::orig::
      int_sum_pre_scan<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          newline_count_index_mem.as<int>(), ctx.num_cuda_threads());
  kernels::orig::int_sum_post_scan<<<1, 1>>>(
      newline_count_index_mem.as<int>(), ctx.num_cuda_threads(),
      num_reduction_cuda_threads, 1, int_sum_base_mem.as<int>());
  kernels::orig::
      int_sum_rebase<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          newline_count_index_mem.as<int>(), ctx.num_cuda_threads(),
          int_sum_base_mem.as<int>(), 1, newline_index_offset_mem.as<int>());

  copy_scalar_to_device<int>(newline_index_offset_mem, 0, 1);
  cuda::synchronize();
  const int num_lines = copy_scalar_from_device<int>(newline_index_offset_mem,
                                                     ctx.num_cuda_threads());
  cuda::DeviceArray newline_index_mem(num_lines * sizeof(long));
  cuda::DeviceArray escape_index_mem(ctx.level_size() * sizeof(long));

  if (combined) {
    escape_index_mem.memset(0);
    kernels::orig::
        combined_escape_newline_index<<<ctx.grid_size, ctx.block_size>>>(
            ctx.device_file(), ctx.file_size, string_carry_index_mem.as<char>(),
            newline_index_offset_mem.as<int>(), escape_index_mem.as<long>(),
            newline_index_mem.as<long>());
  } else {
    kernels::orig::escape_index<<<ctx.grid_size, ctx.block_size>>>(
        ctx.device_file(), ctx.file_size, string_carry_index_mem.as<char>(),
        escape_index_mem.as<long>());
    kernels::orig::newline_index<<<ctx.grid_size, ctx.block_size>>>(
        ctx.device_file(), ctx.file_size, newline_index_offset_mem.as<int>(),
        newline_index_mem.as<long>());
  }

  kernels::orig::quote_index<<<ctx.grid_size, ctx.block_size>>>(
      ctx.device_file(), ctx.file_size, escape_index_mem.as<long>(),
      string_index_mem.as<long>(), string_carry_index_mem.as<char>());

  cuda::DeviceArray xor_base_mem(ctx.reduction_grid_size *
                                 ctx.reduction_block_size * sizeof(char));
  kernels::orig::
      xor_pre_scan<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          string_carry_index_mem.as<char>(), ctx.num_cuda_threads());
  kernels::orig::xor_post_scan<<<1, 1>>>(
      string_carry_index_mem.as<char>(), ctx.num_cuda_threads(),
      num_reduction_cuda_threads, xor_base_mem.as<char>());
  kernels::orig::
      xor_rebase<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          string_carry_index_mem.as<char>(), ctx.num_cuda_threads(),
          xor_base_mem.as<char>());

  kernels::orig::string_index<<<ctx.grid_size, ctx.block_size>>>(
      string_index_mem.as<long>(), ctx.level_size(),
      string_carry_index_mem.as<char>());

  NewlineIndex newline_index(std::move(newline_index_mem), num_lines);
  StringIndex string_index(std::move(string_index_mem));

  return {std::move(newline_index), std::move(string_index)};
}

LeveledBitmapIndex
create_leveled_bitmap_index(const OrigIndexBuilderContext &ctx,
                            const StringIndex &string_index) {
  LogInfo("Create leveled bitmap index");
  cuda::DeviceArray carry_index_mem(ctx.num_cuda_threads() * sizeof(char));
  kernels::orig::leveled_bitmaps_carry_index<<<ctx.grid_size, ctx.block_size>>>(
      ctx.device_file(), ctx.file_size,
      static_cast<const long *>(string_index.data()),
      carry_index_mem.as<char>());

  cuda::DeviceArray carry_index_with_offset_mem((ctx.num_cuda_threads() + 1) *
                                                sizeof(char));
  copy_scalar_to_device<char>(carry_index_with_offset_mem, 0, -1);

  cuda::DeviceArray char_sum_base_mem(ctx.reduction_grid_size *
                                      ctx.reduction_block_size * sizeof(char));
  const int num_reduction_cuda_threads =
      ctx.reduction_grid_size * ctx.reduction_block_size;
  kernels::orig::
      char_sum_pre_scan<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          carry_index_mem.as<char>(), ctx.num_cuda_threads());
  kernels::orig::char_sum_post_scan<<<1, 1>>>(
      carry_index_mem.as<char>(), ctx.num_cuda_threads(),
      num_reduction_cuda_threads, static_cast<char>(-1),
      char_sum_base_mem.as<char>());
  kernels::orig::
      char_sum_rebase<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          carry_index_mem.as<char>(), ctx.num_cuda_threads(),
          char_sum_base_mem.as<char>(), 1,
          carry_index_with_offset_mem.as<char>());

  cuda::DeviceArray leveled_bitmap_index_mem(ctx.level_size() * ctx.max_depth *
                                             sizeof(long));
  leveled_bitmap_index_mem.memset(0);
  kernels::orig::leveled_bitmaps_index<<<ctx.grid_size, ctx.block_size>>>(
      ctx.device_file(), ctx.file_size,
      static_cast<const long *>(string_index.data()),
      carry_index_with_offset_mem.as<char>(),
      leveled_bitmap_index_mem.as<long>(), ctx.level_size(), ctx.max_depth);

  return LeveledBitmapIndex(std::move(leveled_bitmap_index_mem), ctx.max_depth);
}
} // namespace
} // namespace gpjson::index

namespace gpjson::index {

UncombinedIndexBuilder::UncombinedIndexBuilder(
    const file::FileReader &file_reader)
    : file_reader_(file_reader) {
  LogInfo("Initialize uncombined index builder.");
}

BuiltIndices
UncombinedIndexBuilder::build(const file::PartitionView &partition_view,
                              size_t max_depth,
                              const IndexBuilderOptions &options) const {
  LogInfo("Build uncombined index builder.");
  const OrigIndexBuilderContext ctx(options, max_depth, partition_view);

  auto [newline_index, string_index] =
      create_newline_and_string_index(false, ctx);
  auto leveled_bitmap_index = create_leveled_bitmap_index(ctx, string_index);
  return {std::move(newline_index), std::move(string_index),
          std::move(leveled_bitmap_index)};
}

CombinedIndexBuilder::CombinedIndexBuilder(const file::FileReader &file_reader)
    : file_reader_(file_reader) {
  LogInfo("Initialize uncombined index builder.");
}

BuiltIndices
CombinedIndexBuilder::build(const file::PartitionView &partition_view,
                            size_t max_depth,
                            const IndexBuilderOptions &options) const {
  LogInfo("Build uncombined index builder.");
  const OrigIndexBuilderContext ctx(options, max_depth, partition_view);

  auto [newline_index, string_index] =
      create_newline_and_string_index(true, ctx);
  auto leveled_bitmap_index = create_leveled_bitmap_index(ctx, string_index);
  return {std::move(newline_index), std::move(string_index),
          std::move(leveled_bitmap_index)};
}

} // namespace gpjson::index
