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

namespace gpjson::index {
namespace {

constexpr int kTargetChunkBytes = 64 * 1024;

class SharememIndexBuilderContext {
public:
  SharememIndexBuilderContext(int block_size, int reduction_grid_size,
                              int reduction_block_size, int max_depth,
                              const file::FilePartition &partition)
      : block_size(sanitize_block_size(block_size)),
        reduction_grid_size(sanitize_positive_dim(reduction_grid_size, 32)),
        reduction_block_size(sanitize_positive_dim(reduction_block_size, 32)),
        max_depth(max_depth),
        file_size(static_cast<int>(partition.size_bytes())),
        device_file_(static_cast<const char *>(partition.device_bytes())),
        num_chunks(
            compute_num_chunks(static_cast<int>(partition.size_bytes()))),
        warp_grid_size(compute_warp_grid_size(block_size, num_chunks)),
        linear_grid_size(compute_linear_grid_size(block_size, num_chunks)) {
    if (partition.size_bytes() > 0 && device_file_ == nullptr) {
      throw error::common::ImplementationError(
          "Index building requires a GPU-resident partition buffer");
    }
  }

  SharememIndexBuilderContext(const IndexBuilderOptions &options, int max_depth,
                              const file::FilePartition &partition)
      : SharememIndexBuilderContext(
            options.block_size, options.reduction_grid_size,
            options.reduction_block_size, max_depth, partition) {}

  int level_size() const { return (file_size + 64 - 1) / 64; }

  int reduction_thread_count() const {
    return reduction_grid_size * reduction_block_size;
  }

  int reduction_scan_stride() const {
    return std::min(num_chunks, reduction_thread_count());
  }

  size_t warp_shared_bytes() const { return 0; }

  const char *device_file() const { return device_file_; }

  const int block_size;
  const int reduction_grid_size;
  const int reduction_block_size;
  const int max_depth;
  const int file_size;
  const int num_chunks;
  const int warp_grid_size;
  const int linear_grid_size;

private:
  static int sanitize_positive_dim(int value, int fallback) {
    return value > 0 ? value : fallback;
  }

  static int sanitize_block_size(int value) {
    int block = value > 0 ? value : 256;
    block = std::max(32, block);
    block = (block / 32) * 32;
    block = std::min(block, 1024);
    return block;
  }

  static int compute_num_chunks(int file_size) {
    if (file_size <= 0) {
      return 1;
    }
    return std::max(1, (file_size + kTargetChunkBytes - 1) / kTargetChunkBytes);
  }

  static int compute_warp_grid_size(int block_size, int num_chunks) {
    const int warps_per_block = std::max(1, block_size / 32);
    return std::max(1, (num_chunks + warps_per_block - 1) / warps_per_block);
  }

  static int compute_linear_grid_size(int block_size, int num_chunks) {
    return std::max(1, (num_chunks + block_size - 1) / block_size);
  }

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
        string_index(std::move(string_index)) {}
};

NewlineStringIndices
create_newline_and_string_index(const SharememIndexBuilderContext &ctx,
                                profiler::Profiler &profiler) {
  LogInfo("Create newline and string index (sharemem)");
  const profiler::Profiler::SegmentId total_timer =
      profiler.begin("  create_newline_and_string_index");

  cuda::DeviceArray string_index_mem(ctx.level_size() * sizeof(long));
  cuda::DeviceArray escape_carry_index_mem(ctx.num_chunks * sizeof(char));
  cuda::DeviceArray quote_carry_index_mem(ctx.num_chunks * sizeof(char));
  cuda::DeviceArray newline_count_index_mem(ctx.num_chunks * sizeof(int));
  cuda::DeviceArray newline_index_offset_mem((ctx.num_chunks + 1) *
                                             sizeof(int));

  quote_carry_index_mem.memset(0);
  newline_count_index_mem.memset(0);
  newline_index_offset_mem.memset(0);

  const profiler::Profiler::SegmentId
      combined_escape_carry_newline_count_timer =
          profiler.begin("    combined_escape_carry_newline_count_index");
  kernels::sharemem::combined_escape_carry_newline_count_index<<<
      ctx.warp_grid_size, ctx.block_size, ctx.warp_shared_bytes()>>>(
      ctx.device_file(), ctx.file_size, ctx.num_chunks,
      escape_carry_index_mem.as<char>(), newline_count_index_mem.as<int>());
  cuda::check(cudaGetLastError(),
              "sharemem combined_escape_carry_newline_count_index launch");
  cuda::synchronize();
  profiler.end(combined_escape_carry_newline_count_timer);

  cuda::DeviceArray int_sum_base_mem(ctx.reduction_scan_stride() * sizeof(int));
  const profiler::Profiler::SegmentId int_sum_pre_scan_timer =
      profiler.begin("    int_sum_pre_scan");
  kernels::orig::
      int_sum_pre_scan<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          newline_count_index_mem.as<int>(), ctx.num_chunks);
  cuda::synchronize();
  profiler.end(int_sum_pre_scan_timer);

  const profiler::Profiler::SegmentId int_sum_post_scan_timer =
      profiler.begin("    int_sum_post_scan");
  kernels::orig::int_sum_post_scan<<<1, 1>>>(
      newline_count_index_mem.as<int>(), ctx.num_chunks,
      ctx.reduction_scan_stride(), 1, int_sum_base_mem.as<int>());
  cuda::synchronize();
  profiler.end(int_sum_post_scan_timer);

  const profiler::Profiler::SegmentId int_sum_rebase_timer =
      profiler.begin("    int_sum_rebase");
  kernels::orig::
      int_sum_rebase<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          newline_count_index_mem.as<int>(), ctx.num_chunks,
          int_sum_base_mem.as<int>(), 1, newline_index_offset_mem.as<int>());
  cuda::synchronize();
  profiler.end(int_sum_rebase_timer);

  copy_scalar_to_device<int>(newline_index_offset_mem, 0, 1);
  cuda::synchronize();

  const int num_lines =
      copy_scalar_from_device<int>(newline_index_offset_mem, ctx.num_chunks);

  cuda::DeviceArray newline_index_mem(num_lines * sizeof(long));
  copy_scalar_to_device<long>(newline_index_mem, 0, 0L);

  cuda::DeviceArray escape_index_mem(ctx.level_size() * sizeof(long));
  escape_index_mem.memset(0);
  string_index_mem.memset(0);

  const profiler::Profiler::SegmentId combined_escape_newline_index_timer =
      profiler.begin("    combined_escape_newline_index");
  kernels::sharemem::combined_escape_newline_index<<<
      ctx.warp_grid_size, ctx.block_size, ctx.warp_shared_bytes()>>>(
      ctx.device_file(), ctx.file_size, escape_carry_index_mem.as<char>(),
      newline_index_offset_mem.as<int>(), escape_index_mem.as<long>(),
      newline_index_mem.as<long>(), ctx.num_chunks);
  cuda::check(cudaGetLastError(),
              "sharemem combined_escape_newline_index launch");
  cuda::synchronize();
  profiler.end(combined_escape_newline_index_timer);

  const profiler::Profiler::SegmentId quote_index_timer =
      profiler.begin("    quote_index");
  kernels::sharemem::quote_index<<<ctx.warp_grid_size, ctx.block_size,
                                   ctx.warp_shared_bytes()>>>(
      ctx.device_file(), ctx.file_size, escape_index_mem.as<long>(),
      string_index_mem.as<long>(), quote_carry_index_mem.as<char>(),
      ctx.num_chunks);
  cuda::check(cudaGetLastError(), "sharemem quote_index launch");
  cuda::synchronize();
  profiler.end(quote_index_timer);

  cuda::DeviceArray xor_base_mem(ctx.reduction_scan_stride() * sizeof(char));
  const profiler::Profiler::SegmentId xor_pre_scan_timer =
      profiler.begin("    xor_pre_scan");
  kernels::orig::
      xor_pre_scan<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          quote_carry_index_mem.as<char>(), ctx.num_chunks);
  cuda::synchronize();
  profiler.end(xor_pre_scan_timer);

  const profiler::Profiler::SegmentId xor_post_scan_timer =
      profiler.begin("    xor_post_scan");
  kernels::orig::xor_post_scan<<<1, 1>>>(
      quote_carry_index_mem.as<char>(), ctx.num_chunks,
      ctx.reduction_scan_stride(), xor_base_mem.as<char>());
  cuda::synchronize();
  profiler.end(xor_post_scan_timer);

  const profiler::Profiler::SegmentId xor_rebase_timer =
      profiler.begin("    xor_rebase");
  kernels::orig::
      xor_rebase<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          quote_carry_index_mem.as<char>(), ctx.num_chunks,
          xor_base_mem.as<char>());
  cuda::synchronize();
  profiler.end(xor_rebase_timer);

  const profiler::Profiler::SegmentId string_index_timer =
      profiler.begin("    string_index");
  kernels::sharemem::string_index<<<ctx.linear_grid_size, ctx.block_size>>>(
      string_index_mem.as<long>(), ctx.level_size(),
      quote_carry_index_mem.as<char>(), ctx.file_size, ctx.num_chunks);
  cuda::check(cudaGetLastError(), "sharemem string_index launch");
  cuda::synchronize();
  profiler.end(string_index_timer);

  NewlineIndex newline_index(std::move(newline_index_mem), num_lines);
  StringIndex string_index(std::move(string_index_mem));
  profiler.end(total_timer);
  return {std::move(newline_index), std::move(string_index)};
}

LeveledBitmapIndex
create_leveled_bitmap_index(const SharememIndexBuilderContext &ctx,
                            const StringIndex &string_index,
                            profiler::Profiler &profiler) {
  LogInfo("Create leveled bitmap index (sharemem)");
  const profiler::Profiler::SegmentId total_timer =
      profiler.begin("  create_leveled_bitmap_index");

  cuda::DeviceArray carry_index_mem(ctx.num_chunks * sizeof(char));
  carry_index_mem.memset(0);

  const profiler::Profiler::SegmentId carry_index_timer =
      profiler.begin("    leveled_bitmaps_carry_index");
  kernels::sharemem::leveled_bitmaps_carry_index<<<
      ctx.warp_grid_size, ctx.block_size, ctx.warp_shared_bytes()>>>(
      ctx.device_file(), ctx.file_size,
      static_cast<const long *>(string_index.data()),
      carry_index_mem.as<char>(), ctx.num_chunks);
  cuda::check(cudaGetLastError(),
              "sharemem leveled_bitmaps_carry_index launch");
  cuda::synchronize();
  profiler.end(carry_index_timer);

  cuda::DeviceArray carry_index_with_offset_mem((ctx.num_chunks + 1) *
                                                sizeof(char));
  carry_index_with_offset_mem.memset(0);
  copy_scalar_to_device<char>(carry_index_with_offset_mem, 0, -1);

  cuda::DeviceArray char_sum_base_mem(ctx.reduction_scan_stride() *
                                      sizeof(char));
  const profiler::Profiler::SegmentId char_sum_pre_scan_timer =
      profiler.begin("    char_sum_pre_scan");
  kernels::orig::
      char_sum_pre_scan<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          carry_index_mem.as<char>(), ctx.num_chunks);
  cuda::synchronize();
  profiler.end(char_sum_pre_scan_timer);

  const profiler::Profiler::SegmentId char_sum_post_scan_timer =
      profiler.begin("    char_sum_post_scan");
  kernels::orig::char_sum_post_scan<<<1, 1>>>(
      carry_index_mem.as<char>(), ctx.num_chunks, ctx.reduction_scan_stride(),
      static_cast<char>(-1), char_sum_base_mem.as<char>());
  cuda::synchronize();
  profiler.end(char_sum_post_scan_timer);

  const profiler::Profiler::SegmentId char_sum_rebase_timer =
      profiler.begin("    char_sum_rebase");
  kernels::orig::
      char_sum_rebase<<<ctx.reduction_grid_size, ctx.reduction_block_size>>>(
          carry_index_mem.as<char>(), ctx.num_chunks,
          char_sum_base_mem.as<char>(), 1,
          carry_index_with_offset_mem.as<char>());
  cuda::synchronize();
  profiler.end(char_sum_rebase_timer);

  cuda::DeviceArray leveled_bitmap_index_mem(ctx.level_size() * ctx.max_depth *
                                             sizeof(long));
  leveled_bitmap_index_mem.memset(0);

  const profiler::Profiler::SegmentId leveled_bitmaps_index_timer =
      profiler.begin("    leveled_bitmaps_index");
  kernels::sharemem::leveled_bitmaps_index<<<ctx.warp_grid_size, ctx.block_size,
                                             ctx.warp_shared_bytes()>>>(
      ctx.device_file(), ctx.file_size,
      static_cast<const long *>(string_index.data()),
      carry_index_with_offset_mem.as<char>(),
      leveled_bitmap_index_mem.as<long>(), ctx.level_size(), ctx.max_depth,
      ctx.num_chunks);
  cuda::check(cudaGetLastError(), "sharemem leveled_bitmaps_index launch");
  cuda::synchronize();
  profiler.end(leveled_bitmaps_index_timer);

  profiler.end(total_timer);
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
                            const IndexBuilderOptions &options) const {
  LogInfo("Build sharemem index builder.");
  profiler::Profiler profiler("SharememIndexBuilder profiler");
  const profiler::Profiler::SegmentId build_timer = profiler.begin("build");
  const SharememIndexBuilderContext ctx(options, static_cast<int>(max_depth),
                                        partition);

  auto [newline_index, string_index] =
      create_newline_and_string_index(ctx, profiler);
  auto leveled_bitmap_index =
      create_leveled_bitmap_index(ctx, string_index, profiler);
  profiler.end(build_timer);

  return {std::move(newline_index), std::move(string_index),
          std::move(leveled_bitmap_index)};
}

} // namespace gpjson::index
