/*
 * Index builders from the original java codebase
 */

#include "gpjson/cuda/cuda.hpp"
#include "gpjson/error/common.hpp"
#include "gpjson/file/file.hpp"
#include "gpjson/index/index.hpp"
#include "gpjson/index/index_builder.hpp"
#include "gpjson/index/kernels/orig.cuh"
#include "gpjson/index/kernels/sharemem.cuh"
#include "gpjson/log/log.hpp"
#include "gpjson/profiler/profiler.hpp"

#include <algorithm>
#include <cuda_runtime_api.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace gpjson::index {
namespace {

class SharememIndexBuilderContext {
public:
  SharememIndexBuilderContext(int grid_size, int block_size,
                              int reduction_grid_size, int reduction_block_size,
                              int max_depth,
                              const file::FilePartition &partition)
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

  SharememIndexBuilderContext(const IndexBuilderOptions &options, int max_depth,
                              const file::FilePartition &partition)
      : SharememIndexBuilderContext(
            options.grid_size, options.block_size, options.reduction_grid_size,
            options.reduction_block_size, max_depth, partition) {}

  int num_cuda_threads() const { return grid_size * block_size; }
  int level_size() const {
    // TODO: Why? This is copied from java codebase, but 64 seems random
    return static_cast<int>((this->file_size + 64 - 1) / 64);
  }

  const char *device_file() const { return device_file_; }

  const int grid_size;
  const int block_size;
  const int reduction_grid_size;
  const int reduction_block_size;
  const int max_depth;
  const size_t file_size;

private:
  const char *device_file_{nullptr};
};

int reduction_scan_stride(const SharememIndexBuilderContext &ctx) {
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

inline void run_int_sum_scan_orig(const SharememIndexBuilderContext &ctx,
                                  cuda::DeviceArray &newline_count_index_mem,
                                  profiler::Profiler &profiler) {
  const profiler::Profiler::SegmentId int_sum_scan_timer =
      profiler.begin_nested("int_sum_scan");
  cuda::DeviceArray int_sum_base_mem(ctx.reduction_grid_size *
                                     ctx.reduction_block_size * sizeof(int));
  cuda::DeviceArray newline_index_offset_mem((ctx.grid_size + 1) * sizeof(int));
  copy_scalar_to_device<int>(newline_index_offset_mem, 0, 0);

  const int scan_stride = std::min(ctx.grid_size, ctx.reduction_grid_size *
                                                      ctx.reduction_block_size);
  const profiler::Profiler::SegmentId int_sum_pre_scan_timer =
      profiler.begin("int_sum_pre_scan");
  kernels::sharemem::
      int_sum_pre_scan<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          newline_count_index_mem.as<int>(), ctx.grid_size);
  cuda::synchronize_and_check();
  profiler.end(int_sum_pre_scan_timer);

  const profiler::Profiler::SegmentId int_sum_post_scan_timer =
      profiler.begin("int_sum_post_scan");
  kernels::sharemem::int_sum_post_scan<<<1, 1>>>(
      newline_count_index_mem.as<int>(), ctx.grid_size, scan_stride, 0,
      int_sum_base_mem.as<int>());
  cuda::synchronize_and_check();
  profiler.end(int_sum_post_scan_timer);

  const profiler::Profiler::SegmentId int_sum_rebase_timer =
      profiler.begin("int_sum_rebase");
  kernels::sharemem::
      int_sum_rebase<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          newline_count_index_mem.as<int>(), ctx.grid_size,
          int_sum_base_mem.as<int>(), 1, newline_index_offset_mem.as<int>());
  cuda::synchronize_and_check();
  profiler.end(int_sum_rebase_timer);

  cuda::check(cudaMemcpy(newline_count_index_mem.as<int>(),
                         newline_index_offset_mem.as<const int>(),
                         (ctx.grid_size + 1) * sizeof(int),
                         cudaMemcpyDeviceToDevice),
              "cudaMemcpy newline index offsets device to device");
  profiler.end(int_sum_scan_timer);
}

inline void run_int_sum_scan_thrust(const SharememIndexBuilderContext &ctx,
                                    cuda::DeviceArray &newline_count_index_mem,
                                    profiler::Profiler &profiler) {
  const profiler::Profiler::SegmentId newline_index_offset_timer =
      profiler.begin("int_sum_scan");
  thrust::exclusive_scan(thrust::device, newline_count_index_mem.as<int>(),
                         newline_count_index_mem.as<int>() + ctx.grid_size + 1,
                         newline_count_index_mem.as<int>());
  cuda::synchronize_and_check();
  profiler.end(newline_index_offset_timer);
}

inline void run_xor_scan_orig(const SharememIndexBuilderContext &ctx,
                              cuda::DeviceArray &string_carry_index_mem,
                              profiler::Profiler &profiler) {
  const profiler::Profiler::SegmentId xor_scan_timer =
      profiler.begin_nested("xor_scan");
  cuda::DeviceArray xor_base_mem(ctx.reduction_grid_size *
                                 ctx.reduction_block_size * sizeof(char));
  const int scan_stride = reduction_scan_stride(ctx);
  const profiler::Profiler::SegmentId xor_pre_scan_timer =
      profiler.begin("xor_pre_scan");
  kernels::sharemem::
      xor_pre_scan<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          string_carry_index_mem.as<char>(), ctx.num_cuda_threads());
  cuda::synchronize_and_check();
  profiler.end(xor_pre_scan_timer);

  const profiler::Profiler::SegmentId xor_post_scan_timer =
      profiler.begin("xor_post_scan");
  kernels::sharemem::xor_post_scan<<<1, 1>>>(
      string_carry_index_mem.as<char>(), ctx.num_cuda_threads(), scan_stride,
      xor_base_mem.as<char>());
  cuda::synchronize_and_check();
  profiler.end(xor_post_scan_timer);

  const profiler::Profiler::SegmentId xor_rebase_timer =
      profiler.begin("xor_rebase");
  kernels::sharemem::
      xor_rebase<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          string_carry_index_mem.as<char>(), ctx.num_cuda_threads(),
          xor_base_mem.as<char>());
  cuda::synchronize_and_check();
  profiler.end(xor_rebase_timer);
  profiler.end(xor_scan_timer);
}

inline void run_xor_scan_thrust(const SharememIndexBuilderContext &ctx,
                                cuda::DeviceArray &string_carry_index_mem,
                                profiler::Profiler &profiler) {
  const profiler::Profiler::SegmentId xor_scan_timer =
      profiler.begin("xor_scan");
  thrust::inclusive_scan(
      thrust::device, string_carry_index_mem.as<char>(),
      string_carry_index_mem.as<char>() + ctx.num_cuda_threads(),
      string_carry_index_mem.as<char>(), thrust::bit_xor<char>());
  cuda::synchronize_and_check();
  profiler.end(xor_scan_timer);
}

inline void create_string_index_from_escape_index(
    const SharememIndexBuilderContext &ctx, cuda::DeviceArray &escape_index_mem,
    cuda::DeviceArray &string_index_mem,
    cuda::DeviceArray &string_carry_index_mem, profiler::Profiler &profiler) {
  const profiler::Profiler::SegmentId quote_index_timer =
      profiler.begin("quote_index");
  kernels::sharemem::quote_index<<<ctx.grid_size, ctx.block_size>>>(
      ctx.device_file(), ctx.file_size, escape_index_mem.as<long>(),
      string_index_mem.as<long>(), string_carry_index_mem.as<char>());
  cuda::synchronize_and_check();
  profiler.end(quote_index_timer);

  // run_xor_scan_orig(ctx, string_carry_index_mem, profiler);
  run_xor_scan_thrust(ctx, string_carry_index_mem, profiler);

  const profiler::Profiler::SegmentId string_index_timer =
      profiler.begin("string_index");
  kernels::sharemem::string_index<<<ctx.grid_size, ctx.block_size>>>(
      string_index_mem.as<long>(), ctx.level_size(),
      string_carry_index_mem.as<char>());
  cuda::synchronize_and_check();
  profiler.end(string_index_timer);
}

inline void
run_char_sum_scan_orig(const SharememIndexBuilderContext &ctx,
                       cuda::DeviceArray &carry_index_mem,
                       cuda::DeviceArray &carry_index_with_offset_mem,
                       profiler::Profiler &profiler) {
  const profiler::Profiler::SegmentId char_sum_scan_timer =
      profiler.begin_nested("char_sum_scan");
  cuda::DeviceArray char_sum_base_mem(ctx.reduction_grid_size *
                                      ctx.reduction_block_size * sizeof(char));
  const int scan_stride = reduction_scan_stride(ctx);
  const profiler::Profiler::SegmentId char_sum_pre_scan_timer =
      profiler.begin("char_sum_pre_scan");
  kernels::sharemem::
      char_sum_pre_scan<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          carry_index_mem.as<char>(), ctx.num_cuda_threads());
  cuda::synchronize_and_check();
  profiler.end(char_sum_pre_scan_timer);

  const profiler::Profiler::SegmentId char_sum_post_scan_timer =
      profiler.begin("char_sum_post_scan");
  kernels::sharemem::char_sum_post_scan<<<1, 1>>>(
      carry_index_mem.as<char>(), ctx.num_cuda_threads(), scan_stride,
      static_cast<char>(-1), char_sum_base_mem.as<char>());
  cuda::synchronize_and_check();
  profiler.end(char_sum_post_scan_timer);

  const profiler::Profiler::SegmentId char_sum_rebase_timer =
      profiler.begin("char_sum_rebase");
  kernels::sharemem::
      char_sum_rebase<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          carry_index_mem.as<char>(), ctx.num_cuda_threads(),
          char_sum_base_mem.as<char>(), 1,
          carry_index_with_offset_mem.as<char>());
  cuda::synchronize_and_check();
  profiler.end(char_sum_rebase_timer);
  profiler.end(char_sum_scan_timer);
}

inline void
run_char_sum_scan_thrust(const SharememIndexBuilderContext &ctx,
                         cuda::DeviceArray &carry_index_mem,
                         cuda::DeviceArray &carry_index_with_offset_mem,
                         profiler::Profiler &profiler) {
  const profiler::Profiler::SegmentId char_sum_scan_timer =
      profiler.begin("char_sum_scan");
  thrust::inclusive_scan(thrust::device, carry_index_mem.as<char>(),
                         carry_index_mem.as<char>() + ctx.num_cuda_threads(),
                         carry_index_with_offset_mem.as<char>() + 1,
                         thrust::plus<char>());
  // This minus 1 from all entries of the array
  thrust::transform(
      thrust::device, carry_index_with_offset_mem.as<char>() + 1,
      carry_index_with_offset_mem.as<char>() + ctx.num_cuda_threads() + 1,
      thrust::make_constant_iterator(static_cast<char>(1)),
      carry_index_with_offset_mem.as<char>() + 1, thrust::minus<char>());

  cuda::synchronize_and_check();
  profiler.end(char_sum_scan_timer);
}

NewlineStringIndices
create_newline_and_string_index(const SharememIndexBuilderContext &ctx,
                                profiler::Profiler &profiler) {
  LogInfo("Create newline and string index");

  cuda::DeviceArray string_index_mem(ctx.level_size() * sizeof(long));
  cuda::DeviceArray string_carry_index_mem(ctx.num_cuda_threads() *
                                           sizeof(char));
  // 1 extra slot in the end to store number of newlines w/ exclusive scan
  cuda::DeviceArray per_tile_newline_count_index_mem((ctx.grid_size + 1) *
                                                     sizeof(int));
  // int *per_tile_newline_count_index_mem =
  //     per_tile_newline_offset_index_mem.as<int>() + 1;

  const profiler::Profiler::SegmentId total_newline_index_timer =
      profiler.begin_nested("newline_index related kernels");

  // NOTE: We compute per-tile newline count instead of per-thread newline count
  // mainly because this will save us 256x less global write. Per-thread newline
  // count can be computed on-the-fly.
  const profiler::Profiler::SegmentId newline_count_timer =
      profiler.begin("newline_count_index");
  kernels::sharemem::newline_count_index<<<ctx.grid_size, ctx.block_size>>>(
      ctx.device_file(), ctx.file_size,
      per_tile_newline_count_index_mem.as<int>());
  cuda::synchronize_and_check();
  profiler.end(newline_count_timer);

  // run_int_sum_scan_orig(ctx, per_tile_newline_count_index_mem, profiler);
  run_int_sum_scan_thrust(ctx, per_tile_newline_count_index_mem, profiler);

  // exclusive prefix sum stores the number of physical newline characters.
  const int num_newline_chars = copy_scalar_from_device<int>(
      per_tile_newline_count_index_mem, ctx.grid_size);
  const int num_lines = num_newline_chars + 1;
  LogInfo("num_lines: %d", num_lines);
  // Assert(num_lines == 310978,
  //        "Number of newline is incorrect. Expect 310978, got %d", num_lines);
  cuda::DeviceArray newline_index_mem(num_lines * sizeof(long));
  copy_scalar_to_device<long>(newline_index_mem, 0, 0L);

  const profiler::Profiler::SegmentId newline_index_timer =
      profiler.begin("newline_index");
  kernels::sharemem::newline_index<<<ctx.grid_size, ctx.block_size>>>(
      ctx.device_file(), ctx.file_size,
      per_tile_newline_count_index_mem.as<int>(), newline_index_mem.as<long>());
  cuda::synchronize_and_check();
  profiler.end(newline_index_timer);
  profiler.end(total_newline_index_timer);

  cuda::DeviceArray escape_index_mem(ctx.level_size() * sizeof(long));

  const profiler::Profiler::SegmentId total_string_index_related =
      profiler.begin_nested("string_index related kernels");

  const profiler::Profiler::SegmentId escape_carry_timer =
      profiler.begin("escape_carry_index");
  kernels::sharemem::escape_carry_index<<<ctx.grid_size, ctx.block_size>>>(
      ctx.device_file(), ctx.file_size, string_carry_index_mem.as<char>());
  cuda::synchronize_and_check();
  profiler.end(escape_carry_timer);

  const profiler::Profiler::SegmentId escape_index_timer =
      profiler.begin("escape_index");
  kernels::sharemem::escape_index<<<ctx.grid_size, ctx.block_size>>>(
      ctx.device_file(), ctx.file_size, string_carry_index_mem.as<char>(),
      escape_index_mem.as<long>());
  cuda::synchronize_and_check();
  profiler.end(escape_index_timer);

  create_string_index_from_escape_index(ctx, escape_index_mem, string_index_mem,
                                        string_carry_index_mem, profiler);
  profiler.end(total_string_index_related);

  NewlineIndex newline_index(std::move(newline_index_mem), num_lines);
  StringIndex string_index(std::move(string_index_mem));
  return {std::move(newline_index), std::move(string_index)};
}

LeveledBitmapIndex
create_leveled_bitmap_index(const SharememIndexBuilderContext &ctx,
                            const StringIndex &string_index,
                            profiler::Profiler &profiler) {
  LogInfo("Create leveled bitmap index");

  const profiler::Profiler::SegmentId leveled_bitmap_related_timer =
      profiler.begin_nested("leveled_bitmap related kernels");

  cuda::DeviceArray carry_index_mem(ctx.num_cuda_threads() * sizeof(char));
  const profiler::Profiler::SegmentId carry_index_timer =
      profiler.begin("leveled_bitmaps_carry_index");
  kernels::sharemem::
      leveled_bitmaps_carry_index<<<ctx.grid_size, ctx.block_size>>>(
          ctx.device_file(), ctx.file_size,
          static_cast<const long *>(string_index.data()),
          carry_index_mem.as<char>());
  cuda::synchronize_and_check();
  profiler.end(carry_index_timer);

  cuda::DeviceArray carry_index_with_offset_mem((ctx.num_cuda_threads() + 1) *
                                                sizeof(char));
  copy_scalar_to_device<char>(carry_index_with_offset_mem, 0, -1);

  // run_char_sum_scan_orig(ctx, carry_index_mem, carry_index_with_offset_mem,
  //                        profiler);
  run_char_sum_scan_thrust(ctx, carry_index_mem, carry_index_with_offset_mem,
                           profiler);

  cuda::DeviceArray leveled_bitmap_index_mem(ctx.level_size() * ctx.max_depth *
                                             sizeof(long));
  leveled_bitmap_index_mem.memset(0);
  const profiler::Profiler::SegmentId leveled_bitmaps_index_timer =
      profiler.begin("leveled_bitmaps_index");
  kernels::sharemem::leveled_bitmaps_index<<<ctx.grid_size, ctx.block_size>>>(
      ctx.device_file(), ctx.file_size,
      static_cast<const long *>(string_index.data()),
      carry_index_with_offset_mem.as<char>(),
      leveled_bitmap_index_mem.as<long>(), ctx.level_size(), ctx.max_depth);
  cuda::synchronize_and_check();
  profiler.end(leveled_bitmaps_index_timer);
  profiler.end(leveled_bitmap_related_timer);
  return LeveledBitmapIndex(std::move(leveled_bitmap_index_mem), ctx.max_depth);
}
} // namespace
} // namespace gpjson::index

namespace gpjson::index {

SharememIndexBuilder::SharememIndexBuilder(const file::FileReader &file_reader)
    : file_reader_(file_reader) {
  LogInfo("Initialize sharemem index builder.");
}

BuiltIndices
SharememIndexBuilder::build(const file::FilePartition &partition,
                            size_t max_depth,
                            const IndexBuilderOptions &options,
                            const gpjson::RunContext &run_context) const {
  LogInfo("Build sharemem index builder.");
  profiler::Profiler &profiler = run_context.profiler;
  const SharememIndexBuilderContext ctx(options, max_depth, partition);

  auto [newline_index, string_index] =
      create_newline_and_string_index(ctx, profiler);
  auto leveled_bitmap_index =
      create_leveled_bitmap_index(ctx, string_index, profiler);
  return {std::move(newline_index), std::move(string_index),
          std::move(leveled_bitmap_index)};
}

} // namespace gpjson::index
