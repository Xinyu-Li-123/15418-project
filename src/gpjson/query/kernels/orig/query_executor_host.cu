#include "gpjson/query/kernels/orig.hpp"

#include "gpjson/cuda/cuda.hpp"
#include "gpjson/log/log.hpp"
#include "gpjson/query/error.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace gpjson::query::kernels::orig {
__global__ void query(const char *file, int file_size,
                      const long *newline_index, int newline_index_size,
                      const long *string_index,
                      const long *leveled_bitmaps_index, int level_size,
                      int num_results, int *result);

namespace {

error::query::QueryExecutionError execution_error(const std::string &message) {
  return error::query::QueryExecutionError(message);
}

int narrow_size_to_int(size_t value, const char *context) {
  if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw execution_error(std::string(context) + " exceeds CUDA int range");
  }
  return static_cast<int>(value);
}

int sanitize_block_size(int requested_block_size) {
  if (requested_block_size <= 0) {
    return 256;
  }
  return requested_block_size;
}

int compute_grid_size(int num_lines, int block_size, int requested_grid_size) {
  if (requested_grid_size > 0) {
    return requested_grid_size;
  }
  if (num_lines <= 0) {
    return 1;
  }
  return std::max(1, (num_lines + block_size - 1) / block_size);
}

int compute_level_size(size_t partition_size) {
  return narrow_size_to_int((partition_size + 64U - 1U) / 64U, "level size");
}

class QueryExecutionContext {
public:
  QueryExecutionContext(const file::FilePartition &partition,
                        const QueryExecutorOptions &options, int num_lines)
      : block_size(sanitize_block_size(options.block_size)),
        grid_size(compute_grid_size(num_lines, block_size, options.grid_size)),
        file_size(narrow_size_to_int(partition.size_bytes(), "partition size")),
        device_partition_(static_cast<const char *>(partition.device_bytes())) {
    if (partition.size_bytes() > 0 && device_partition_ == nullptr) {
      throw execution_error(
          "Query execution requires a GPU-resident partition buffer");
    }
  }

  const char *device_partition() const { return device_partition_; }

  const int block_size;
  const int grid_size;
  const int file_size;

private:
  const char *device_partition_{nullptr};
};

void append_query_line_results(BatchQueryResult &batch_result,
                               size_t query_index,
                               const file::FilePartition &partition,
                               int num_lines, int num_results,
                               const std::vector<int> &host_result_buffer) {
  const char *partition_bytes =
      reinterpret_cast<const char *>(partition.host_bytes());
  const size_t partition_size = partition.size_bytes();
  const size_t global_partition_offset = partition.global_start_offset();

  for (int line_index = 0; line_index < num_lines; ++line_index) {
    LineQueryResult line_result;
    for (int result_index = 0; result_index < num_results; ++result_index) {
      const size_t buffer_index = static_cast<size_t>(line_index) *
                                      static_cast<size_t>(num_results) * 2U +
                                  static_cast<size_t>(result_index) * 2U;
      const int local_start = host_result_buffer[buffer_index];
      const int local_end = host_result_buffer[buffer_index + 1U];

      if (local_start < 0 || local_end < 0) {
        line_result.add_offset(QueryOffset{});
        continue;
      }

      const size_t start_offset = static_cast<size_t>(local_start);
      const size_t end_offset = static_cast<size_t>(local_end);
      if (start_offset > end_offset || end_offset > partition_size) {
        throw execution_error("Query kernel produced an invalid result range");
      }

      QueryOffset offset;
      offset.start = global_partition_offset + start_offset;
      offset.end = global_partition_offset + end_offset;
      offset.json_text = std::string(partition_bytes + start_offset,
                                     end_offset - start_offset);
      line_result.add_offset(std::move(offset));
    }
    batch_result.add_line_result(query_index, std::move(line_result));
  }
}

void validate_query_inputs(const CompiledQuery &compiled_query,
                           const index::BuiltIndices &built_indices) {
  if (compiled_query.max_depth() >
      static_cast<size_t>(
          built_indices.get_leveled_bitmap_index().num_levels())) {
    throw execution_error(
        "Query requires more bitmap levels than the index provides");
  }
  if (compiled_query.max_depth() >= static_cast<size_t>(kMaxQueryLevels)) {
    throw execution_error("Query exceeds the maximum CUDA executor depth");
  }
}

} // namespace

BatchQueryResult execute_batch(const BatchCompiledQuery &compiled_queries,
                               const file::FilePartition &partition,
                               const index::BuiltIndices &built_indices,
                               const QueryExecutorOptions &options) {
  BatchQueryResult batch_result(compiled_queries.size());
  for (size_t query_index = 0; query_index < compiled_queries.size();
       ++query_index) {
    batch_result.set_query_text(
        query_index, compiled_queries.queries()[query_index].query_text());
  }

  if (compiled_queries.size() == 0 || partition.size_bytes() == 0 ||
      partition.host_bytes() == nullptr) {
    return batch_result;
  }

  if (partition.device_bytes() == nullptr) {
    throw execution_error(
        "Query execution requires a GPU-resident partition buffer");
  }

  if (!cuda::device_available()) {
    throw execution_error("CUDA device unavailable for query execution");
  }

  const int num_lines = built_indices.get_newline_index().num_lines();
  if (num_lines <= 0) {
    return batch_result;
  }

  if (built_indices.get_newline_index().data() == nullptr ||
      built_indices.get_string_index().data() == nullptr ||
      built_indices.get_leveled_bitmap_index().data() == nullptr) {
    throw execution_error("Query execution requires GPU-resident newline, "
                          "string, and bitmap indices");
  }

  const int level_size = compute_level_size(partition.size_bytes());
  const QueryExecutionContext ctx(partition, options, num_lines);

  for (size_t query_index = 0; query_index < compiled_queries.size();
       ++query_index) {
    const CompiledQuery &compiled_query =
        compiled_queries.queries()[query_index];
    validate_query_inputs(compiled_query, built_indices);

    const int num_results =
        narrow_size_to_int(compiled_query.num_result_slots(), "result slots");
    const size_t query_size = compiled_query.ir_bytes().size();
    if (query_size > kMaxQueryIrBytes) {
      throw execution_error("Compiled query IR exceeds constant memory buffer");
    }

    const size_t host_result_count =
        static_cast<size_t>(num_lines) * static_cast<size_t>(num_results) * 2U;

    copy_query_ir_to_constant_memory(compiled_query.ir_bytes().data(),
                                     query_size);

    cuda::DeviceArray device_result(host_result_count * sizeof(int));

    LogInfo(
        "Launch query kernel: grid=%d block=%d lines=%d results=%d bytes=%d",
        ctx.grid_size, ctx.block_size, num_lines, num_results, ctx.file_size);
    query<<<ctx.grid_size, ctx.block_size>>>(
        ctx.device_partition(), ctx.file_size,
        static_cast<const long *>(built_indices.get_newline_index().data()),
        num_lines,
        static_cast<const long *>(built_indices.get_string_index().data()),
        static_cast<const long *>(
            built_indices.get_leveled_bitmap_index().data()),
        level_size, num_results, device_result.as<int>());
    cuda::check(cudaGetLastError(), "execute_query_kernel launch");
    cuda::synchronize();

    std::vector<int> host_result_buffer(host_result_count, -1);
    device_result.copy_to_host(host_result_buffer.data(),
                               host_result_buffer.size() * sizeof(int));
    append_query_line_results(batch_result, query_index, partition, num_lines,
                              num_results, host_result_buffer);
  }

  return batch_result;
}

} // namespace gpjson::query::kernels::orig
