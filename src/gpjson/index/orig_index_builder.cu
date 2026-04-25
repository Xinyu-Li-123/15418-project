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
#include "gpjson/profiler/profiler.hpp"

#include <algorithm>

namespace gpjson::index {
namespace {

class OrigIndexBuilderContext {
public:
  OrigIndexBuilderContext(int grid_size, int block_size,
                          int reduction_grid_size, int reduction_block_size,
                          int max_depth, const file::FilePartition &partition)
      : grid_size(grid_size), block_size(block_size),
        reduction_grid_size(reduction_grid_size),
        reduction_block_size(reduction_block_size), max_depth(max_depth),
        file_size(partition.size_bytes()),
        device_file_(static_cast<const char *>(partition.device_bytes())) {
    if (partition.size_bytes() > 0 && device_file_ == nullptr) {
      throw error::common::ImplementationError(
          "Index building requires a GPU-resident partition buffer");
    }
  }

  OrigIndexBuilderContext(const IndexBuilderOptions &options, int max_depth,
                          const file::FilePartition &partition)
      : OrigIndexBuilderContext(
            options.grid_size, options.block_size, options.reduction_grid_size,
            options.reduction_block_size, max_depth, partition) {}

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

int reduction_scan_stride(const OrigIndexBuilderContext &ctx) {
  return std::min(ctx.num_cuda_threads(),
                  ctx.reduction_grid_size * ctx.reduction_block_size);
}

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

inline void run_int_sum_scan(const OrigIndexBuilderContext &ctx,
                             cuda::DeviceArray &newline_count_index_mem,
                             cuda::DeviceArray &newline_index_offset_mem,
                             profiler::Profiler &profiler) {
  const profiler::Profiler::SegmentId int_sum_scan_timer =
      profiler.begin_nested("int_sum_scan (orig)");
  cuda::DeviceArray int_sum_base_mem(ctx.reduction_grid_size *
                                     ctx.reduction_block_size * sizeof(int));
  const int scan_stride = reduction_scan_stride(ctx);
  const profiler::Profiler::SegmentId int_sum_pre_scan_timer =
      profiler.begin("int_sum_pre_scan");
  kernels::orig::
      int_sum_pre_scan<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          newline_count_index_mem.as<int>(), ctx.num_cuda_threads());
  cuda::synchronize_and_check();
  profiler.end(int_sum_pre_scan_timer);

  const profiler::Profiler::SegmentId int_sum_post_scan_timer =
      profiler.begin("int_sum_post_scan");
  kernels::orig::int_sum_post_scan<<<1, 1>>>(
      newline_count_index_mem.as<int>(), ctx.num_cuda_threads(), scan_stride, 1,
      int_sum_base_mem.as<int>());
  cuda::synchronize_and_check();
  profiler.end(int_sum_post_scan_timer);

  const profiler::Profiler::SegmentId int_sum_rebase_timer =
      profiler.begin("int_sum_rebase");
  kernels::orig::
      int_sum_rebase<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          newline_count_index_mem.as<int>(), ctx.num_cuda_threads(),
          int_sum_base_mem.as<int>(), 1, newline_index_offset_mem.as<int>());
  cuda::synchronize_and_check();
  profiler.end(int_sum_rebase_timer);
  profiler.end(int_sum_scan_timer);
}

inline void run_xor_scan(const OrigIndexBuilderContext &ctx,
                         cuda::DeviceArray &string_carry_index_mem,
                         profiler::Profiler &profiler) {
  const profiler::Profiler::SegmentId xor_scan_timer =
      profiler.begin_nested("xor_scan (orig)");
  cuda::DeviceArray xor_base_mem(ctx.reduction_grid_size *
                                 ctx.reduction_block_size * sizeof(char));
  const int scan_stride = reduction_scan_stride(ctx);
  const profiler::Profiler::SegmentId xor_pre_scan_timer =
      profiler.begin("xor_pre_scan");
  kernels::orig::
      xor_pre_scan<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          string_carry_index_mem.as<char>(), ctx.num_cuda_threads());
  cuda::synchronize_and_check();
  profiler.end(xor_pre_scan_timer);

  const profiler::Profiler::SegmentId xor_post_scan_timer =
      profiler.begin("xor_post_scan");
  kernels::orig::xor_post_scan<<<1, 1>>>(string_carry_index_mem.as<char>(),
                                         ctx.num_cuda_threads(), scan_stride,
                                         xor_base_mem.as<char>());
  cuda::synchronize_and_check();
  profiler.end(xor_post_scan_timer);

  const profiler::Profiler::SegmentId xor_rebase_timer =
      profiler.begin("xor_rebase");
  kernels::orig::
      xor_rebase<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          string_carry_index_mem.as<char>(), ctx.num_cuda_threads(),
          xor_base_mem.as<char>());
  cuda::synchronize_and_check();
  profiler.end(xor_rebase_timer);
  profiler.end(xor_scan_timer);
}

inline void create_string_index_from_escape_index(
    const OrigIndexBuilderContext &ctx, cuda::DeviceArray &escape_index_mem,
    cuda::DeviceArray &string_index_mem,
    cuda::DeviceArray &string_carry_index_mem, profiler::Profiler &profiler) {
  const profiler::Profiler::SegmentId quote_index_timer =
      profiler.begin("quote_index");
  kernels::orig::quote_index<<<ctx.grid_size, ctx.block_size>>>(
      ctx.device_file(), ctx.file_size, escape_index_mem.as<long>(),
      string_index_mem.as<long>(), string_carry_index_mem.as<char>());
  cuda::synchronize_and_check();
  profiler.end(quote_index_timer);

  run_xor_scan(ctx, string_carry_index_mem, profiler);

  const profiler::Profiler::SegmentId string_index_timer =
      profiler.begin("string_index");
  kernels::orig::string_index<<<ctx.grid_size, ctx.block_size>>>(
      string_index_mem.as<long>(), ctx.level_size(),
      string_carry_index_mem.as<char>());
  cuda::synchronize_and_check();
  profiler.end(string_index_timer);
}

inline void run_char_sum_scan(const OrigIndexBuilderContext &ctx,
                              cuda::DeviceArray &carry_index_mem,
                              cuda::DeviceArray &carry_index_with_offset_mem,
                              profiler::Profiler &profiler) {
  cuda::DeviceArray char_sum_base_mem(ctx.reduction_grid_size *
                                      ctx.reduction_block_size * sizeof(char));
  const int scan_stride = reduction_scan_stride(ctx);
  const profiler::Profiler::SegmentId char_sum_pre_scan_timer =
      profiler.begin("char_sum_pre_scan");
  kernels::orig::
      char_sum_pre_scan<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          carry_index_mem.as<char>(), ctx.num_cuda_threads());
  cuda::synchronize_and_check();
  profiler.end(char_sum_pre_scan_timer);

  const profiler::Profiler::SegmentId char_sum_post_scan_timer =
      profiler.begin("char_sum_post_scan");
  kernels::orig::char_sum_post_scan<<<1, 1>>>(
      carry_index_mem.as<char>(), ctx.num_cuda_threads(), scan_stride,
      static_cast<char>(-1), char_sum_base_mem.as<char>());
  cuda::synchronize_and_check();
  profiler.end(char_sum_post_scan_timer);

  const profiler::Profiler::SegmentId char_sum_rebase_timer =
      profiler.begin("char_sum_rebase");
  kernels::orig::
      char_sum_rebase<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          carry_index_mem.as<char>(), ctx.num_cuda_threads(),
          char_sum_base_mem.as<char>(), 1,
          carry_index_with_offset_mem.as<char>());
  cuda::synchronize_and_check();
  profiler.end(char_sum_rebase_timer);
}

NewlineStringIndices
create_uncombined_newline_and_string_index(const OrigIndexBuilderContext &ctx,
                                           profiler::Profiler &profiler) {
  LogInfo("Create newline and string index, combined=0");
  const profiler::Profiler::SegmentId total_timer =
      profiler.begin_nested("create_newline_and_string_index");

  cuda::DeviceArray string_index_mem(ctx.level_size() * sizeof(long));
  cuda::DeviceArray string_carry_index_mem(ctx.num_cuda_threads() *
                                           sizeof(char));
  cuda::DeviceArray newline_count_index_mem(ctx.num_cuda_threads() *
                                            sizeof(int));
  cuda::DeviceArray newline_index_offset_mem((ctx.num_cuda_threads() + 1) *
                                             sizeof(int));

  const profiler::Profiler::SegmentId newline_related_timer =
      profiler.begin_nested("newline_index related kernels");

  const profiler::Profiler::SegmentId newline_count_timer =
      profiler.begin("newline_count_index");
  kernels::orig::newline_count_index<<<ctx.grid_size, ctx.block_size>>>(
      ctx.device_file(), ctx.file_size, newline_count_index_mem.as<int>());
  cuda::synchronize_and_check();
  profiler.end(newline_count_timer);

  run_int_sum_scan(ctx, newline_count_index_mem, newline_index_offset_mem,
                   profiler);

  copy_scalar_to_device<int>(newline_index_offset_mem, 0, 1);
  const int num_lines = copy_scalar_from_device<int>(newline_index_offset_mem,
                                                     ctx.num_cuda_threads());
  LogInfo("num_lines: %d", num_lines);
  cuda::DeviceArray newline_index_mem(num_lines * sizeof(long));
  // Slot 0 is the synthetic start offset for the first line; kernels append
  // discovered newline offsets starting at slot 1.
  copy_scalar_to_device<long>(newline_index_mem, 0, 0L);

  const profiler::Profiler::SegmentId newline_index_timer =
      profiler.begin("newline_index");
  kernels::orig::newline_index<<<ctx.grid_size, ctx.block_size>>>(
      ctx.device_file(), ctx.file_size, newline_index_offset_mem.as<int>(),
      newline_index_mem.as<long>());
  cuda::synchronize_and_check();
  profiler.end(newline_index_timer);
  profiler.end(newline_related_timer);

  cuda::DeviceArray escape_index_mem(ctx.level_size() * sizeof(long));
  const profiler::Profiler::SegmentId string_related_timer =
      profiler.begin_nested("string_index related kernels");

  const profiler::Profiler::SegmentId escape_carry_timer =
      profiler.begin("escape_carry_index");
  kernels::orig::escape_carry_index<<<ctx.grid_size, ctx.block_size>>>(
      ctx.device_file(), ctx.file_size, string_carry_index_mem.as<char>());
  cuda::synchronize_and_check();
  profiler.end(escape_carry_timer);

  const profiler::Profiler::SegmentId escape_index_timer =
      profiler.begin("escape_index");
  kernels::orig::escape_index<<<ctx.grid_size, ctx.block_size>>>(
      ctx.device_file(), ctx.file_size, string_carry_index_mem.as<char>(),
      escape_index_mem.as<long>());
  cuda::synchronize_and_check();
  profiler.end(escape_index_timer);

  create_string_index_from_escape_index(ctx, escape_index_mem, string_index_mem,
                                        string_carry_index_mem, profiler);
  profiler.end(string_related_timer);

  NewlineIndex newline_index(std::move(newline_index_mem), num_lines);
  StringIndex string_index(std::move(string_index_mem));
  profiler.end(total_timer);
  return {std::move(newline_index), std::move(string_index)};
}

NewlineStringIndices
create_combined_newline_and_string_index(const OrigIndexBuilderContext &ctx,
                                         profiler::Profiler &profiler) {
  LogInfo("Create newline and string index, combined=1");
  const profiler::Profiler::SegmentId total_timer =
      profiler.begin_nested("create_newline_and_string_index");

  cuda::DeviceArray string_index_mem(ctx.level_size() * sizeof(long));
  cuda::DeviceArray string_carry_index_mem(ctx.num_cuda_threads() *
                                           sizeof(char));
  cuda::DeviceArray newline_count_index_mem(ctx.num_cuda_threads() *
                                            sizeof(int));
  cuda::DeviceArray newline_index_offset_mem((ctx.num_cuda_threads() + 1) *
                                             sizeof(int));
  cuda::DeviceArray escape_index_mem(ctx.level_size() * sizeof(long));

  const profiler::Profiler::SegmentId newline_related_timer =
      profiler.begin_nested("newline_index related kernels");

  const profiler::Profiler::SegmentId combined_count_timer =
      profiler.begin("combined_escape_carry_newline_count_index");
  kernels::orig::combined_escape_carry_newline_count_index<<<ctx.grid_size,
                                                             ctx.block_size>>>(
      ctx.device_file(), ctx.file_size, string_carry_index_mem.as<char>(),
      newline_count_index_mem.as<int>());
  cuda::synchronize_and_check();
  profiler.end(combined_count_timer);

  run_int_sum_scan(ctx, newline_count_index_mem, newline_index_offset_mem,
                   profiler);

  copy_scalar_to_device<int>(newline_index_offset_mem, 0, 1);
  const int num_lines = copy_scalar_from_device<int>(newline_index_offset_mem,
                                                     ctx.num_cuda_threads());
  LogInfo("num_lines: %d", num_lines);
  cuda::DeviceArray newline_index_mem(num_lines * sizeof(long));
  copy_scalar_to_device<long>(newline_index_mem, 0, 0L);

  escape_index_mem.memset(0);
  const profiler::Profiler::SegmentId combined_index_timer =
      profiler.begin("combined_escape_newline_index");
  kernels::orig::
      combined_escape_newline_index<<<ctx.grid_size, ctx.block_size>>>(
          ctx.device_file(), ctx.file_size, string_carry_index_mem.as<char>(),
          newline_index_offset_mem.as<int>(), escape_index_mem.as<long>(),
          newline_index_mem.as<long>());
  cuda::synchronize_and_check();
  profiler.end(combined_index_timer);
  profiler.end(newline_related_timer);

  const profiler::Profiler::SegmentId string_related_timer =
      profiler.begin_nested("string_index related kernels");

  create_string_index_from_escape_index(ctx, escape_index_mem, string_index_mem,
                                        string_carry_index_mem, profiler);
  profiler.end(string_related_timer);

  NewlineIndex newline_index(std::move(newline_index_mem), num_lines);
  StringIndex string_index(std::move(string_index_mem));
  profiler.end(total_timer);
  return {std::move(newline_index), std::move(string_index)};
}

LeveledBitmapIndex
create_leveled_bitmap_index(const OrigIndexBuilderContext &ctx,
                            const StringIndex &string_index,
                            profiler::Profiler &profiler) {
  LogInfo("Create leveled bitmap index");
  const profiler::Profiler::SegmentId total_timer =
      profiler.begin_nested("create_leveled_bitmap_index");

  const profiler::Profiler::SegmentId leveled_bitmap_related_timer =
      profiler.begin_nested("leveled_bitmap related kernels");

  cuda::DeviceArray carry_index_mem(ctx.num_cuda_threads() * sizeof(char));
  const profiler::Profiler::SegmentId carry_index_timer =
      profiler.begin("leveled_bitmaps_carry_index");
  kernels::orig::leveled_bitmaps_carry_index<<<ctx.grid_size, ctx.block_size>>>(
      ctx.device_file(), ctx.file_size,
      static_cast<const long *>(string_index.data()),
      carry_index_mem.as<char>());
  cuda::synchronize_and_check();
  profiler.end(carry_index_timer);

  cuda::DeviceArray carry_index_with_offset_mem((ctx.num_cuda_threads() + 1) *
                                                sizeof(char));
  copy_scalar_to_device<char>(carry_index_with_offset_mem, 0, -1);

  run_char_sum_scan(ctx, carry_index_mem, carry_index_with_offset_mem,
                    profiler);

  cuda::DeviceArray leveled_bitmap_index_mem(ctx.level_size() * ctx.max_depth *
                                             sizeof(long));
  leveled_bitmap_index_mem.memset(0);
  const profiler::Profiler::SegmentId leveled_bitmaps_index_timer =
      profiler.begin("leveled_bitmaps_index");
  kernels::orig::leveled_bitmaps_index<<<ctx.grid_size, ctx.block_size>>>(
      ctx.device_file(), ctx.file_size,
      static_cast<const long *>(string_index.data()),
      carry_index_with_offset_mem.as<char>(),
      leveled_bitmap_index_mem.as<long>(), ctx.level_size(), ctx.max_depth);
  cuda::synchronize_and_check();
  profiler.end(leveled_bitmaps_index_timer);
  profiler.end(leveled_bitmap_related_timer);
  profiler.end(total_timer);
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
UncombinedIndexBuilder::build(const file::FilePartition &partition,
                              size_t max_depth,
                              const IndexBuilderOptions &options) const {
  LogInfo("Build uncombined index builder.");
  profiler::Profiler profiler("UncombinedIndexBuilder profiler");
  const profiler::Profiler::SegmentId build_timer =
      profiler.begin_nested("build");
  const OrigIndexBuilderContext ctx(options, max_depth, partition);

  auto [newline_index, string_index] =
      create_uncombined_newline_and_string_index(ctx, profiler);
  auto leveled_bitmap_index =
      create_leveled_bitmap_index(ctx, string_index, profiler);
  profiler.end(build_timer);
  return {std::move(newline_index), std::move(string_index),
          std::move(leveled_bitmap_index)};
}

CombinedIndexBuilder::CombinedIndexBuilder(const file::FileReader &file_reader)
    : file_reader_(file_reader) {
  LogInfo("Initialize uncombined index builder.");
}

BuiltIndices
CombinedIndexBuilder::build(const file::FilePartition &partition,
                            size_t max_depth,
                            const IndexBuilderOptions &options) const {
  LogInfo("Build uncombined index builder.");
  profiler::Profiler profiler("CombinedIndexBuilder profiler");
  const profiler::Profiler::SegmentId build_timer =
      profiler.begin_nested("build");
  const OrigIndexBuilderContext ctx(options, max_depth, partition);

  auto [newline_index, string_index] =
      create_combined_newline_and_string_index(ctx, profiler);
  auto leveled_bitmap_index =
      create_leveled_bitmap_index(ctx, string_index, profiler);
  profiler.end(build_timer);
  return {std::move(newline_index), std::move(string_index),
          std::move(leveled_bitmap_index)};
}

} // namespace gpjson::index
