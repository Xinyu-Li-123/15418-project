#include <assert.h>

namespace gpjson::query::kernels::orig {
namespace {

#define MAX_NUM_LEVELS 16

#define OPCODE_END 0x00
#define OPCODE_STORE_RESULT 0x01
#define OPCODE_MOVE_UP 0x02
#define OPCODE_MOVE_DOWN 0x03
#define OPCODE_MOVE_TO_KEY 0x04
#define OPCODE_MOVE_TO_INDEX 0x05
#define OPCODE_MOVE_TO_INDEX_REVERSE 0x06
#define OPCODE_MARK_POSITION 0x07
#define OPCODE_RESET_POSITION 0x08
#define OPCODE_EXPRESSION_STRING_EQUALS 0x09

__device__ int find_next_structural_char(const long *ext_index, int level_end,
                                         int line_index, int current_level,
                                         int level_size) {
  long index = ext_index[level_size * current_level + line_index / 64];
  while (index == 0 && line_index < level_end) {
    line_index += 64 - (line_index % 64);
    index = ext_index[level_size * current_level + line_index / 64];
  }

  bool is_structural = (index & (1L << (line_index % 64))) != 0;
  while (!is_structural && line_index < level_end) {
    ++line_index;
    index = ext_index[level_size * current_level + line_index / 64];
    is_structural = (index & (1L << (line_index % 64))) != 0;
  }
  return line_index;
}

__device__ int find_previous_structural_char(const long *ext_index,
                                             int level_start, int line_index,
                                             int current_level,
                                             int level_size) {
  long index = ext_index[level_size * current_level + line_index / 64];
  while (index == 0 && line_index > level_start) {
    line_index -= 64 - (line_index % 64);
    index = ext_index[level_size * current_level + line_index / 64];
  }

  bool is_structural = (index & (1L << (line_index % 64))) != 0;
  while (!is_structural && line_index > level_start) {
    --line_index;
    index = ext_index[level_size * current_level + line_index / 64];
    is_structural = (index & (1L << (line_index % 64))) != 0;
  }
  return line_index;
}

__device__ int read_varint(const unsigned char *query, int *query_pos) {
  int value = 0;
  int shift = 0;
  int byte = 0;
  while (((byte = query[(*query_pos)++]) & 0x80) != 0) {
    value |= (byte & 0x7F) << shift;
    shift += 7;
    assert(shift <= 35);
  }
  return value | (byte << shift);
}

} // namespace

__global__ void query(const char *file, int file_size,
                      const long *newline_index, int newline_index_size,
                      const long *string_index,
                      const long *leveled_bitmaps_index, int level_size,
                      const unsigned char *query, int num_results,
                      int *result) {
  const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  const int lines_per_thread = (newline_index_size + stride - 1) / stride;
  const int start = thread_index * lines_per_thread;
  const int end = start + lines_per_thread;

  const int init_start = start * 2 * num_results;
  const int init_end = end * 2 * num_results;
  for (int i = init_start;
       i < init_end && i < num_results * 2 * newline_index_size; ++i) {
    result[i] = -1;
  }

  for (int file_index = start;
       file_index < end && file_index < newline_index_size; ++file_index) {
    int line_start = static_cast<int>(newline_index[file_index]);
    int line_end = (file_index + 1 < newline_index_size)
                       ? static_cast<int>(newline_index[file_index + 1])
                       : file_size - 1;

    if (line_start < 0 || line_start >= file_size || line_end < 0) {
      continue;
    }
    if (line_end >= file_size) {
      line_end = file_size - 1;
    }

    while (line_end > line_start && file[line_end] != '}') {
      --line_end;
    }

    while (line_start < line_end && file[line_start] != '{') {
      ++line_start;
    }

    if (line_start == line_end) {
      continue;
    }

    int line_index = line_start;
    assert(file[line_start] == '{');
    assert(file[line_end] == '}');

    int current_level = 0;
    int query_pos = 0;
    int level_end[MAX_NUM_LEVELS];
    level_end[0] = line_end;
    for (int i = 1; i < MAX_NUM_LEVELS; ++i) {
      level_end[i] = -1;
    }

    int marked_pos[MAX_NUM_LEVELS];
    int marked_pos_level[MAX_NUM_LEVELS];
    int marked_pos_index = -1;

    int num_results_index = 0;
    const unsigned char *key = nullptr;
    int key_len = 0;
    int target_index = 0;
    int curr_index[MAX_NUM_LEVELS];
    for (int i = 0; i < MAX_NUM_LEVELS; ++i) {
      curr_index[i] = -1;
    }

    while (true) {
      const unsigned char current_opcode = query[query_pos++];
      switch (current_opcode) {
      case OPCODE_END:
        goto next_line;

      case OPCODE_STORE_RESULT: {
        assert(num_results_index < num_results);

        int end_str = level_end[current_level];
        while (file[line_index] == ' ' &&
               line_index < level_end[current_level]) {
          ++line_index;
        }
        while (end_str > line_index && file[end_str - 1] == ' ') {
          --end_str;
        }

        const int result_index =
            file_index * 2 * num_results + num_results_index * 2;
        result[result_index] = line_index;
        result[result_index + 1] = end_str;
        assert(result[result_index] <= result[result_index + 1]);
        ++num_results_index;
        break;
      }

      case OPCODE_MOVE_UP:
        line_index = level_end[current_level];
        level_end[current_level] = -1;
        --current_level;
        break;

      case OPCODE_MOVE_DOWN:
        ++current_level;
        if (level_end[current_level] == -1) {
          for (int end_candidate = line_index + 1;
               end_candidate <= level_end[current_level - 1]; ++end_candidate) {
            const long index =
                leveled_bitmaps_index[level_size * (current_level - 1) +
                                      end_candidate / 64];
            if (index == 0) {
              end_candidate += 64 - (end_candidate % 64) - 1;
              continue;
            }
            const bool is_structural =
                (index & (1L << (end_candidate % 64))) != 0;
            if (is_structural) {
              level_end[current_level] = end_candidate;
              break;
            }
          }
          assert(level_end[current_level] != -1);
          while (file[line_index] == ' ') {
            ++line_index;
          }
        }
        break;

      case OPCODE_MOVE_TO_KEY:
        key_len = read_varint(query, &query_pos);
        key = query + query_pos;
        query_pos += key_len;

        if (file[line_index] == '{') {
        search_key:
          ++line_index;
          line_index = find_next_structural_char(
              leveled_bitmaps_index, level_end[current_level], line_index,
              current_level, level_size);
          assert(file[line_index] == ':' || file[line_index] == '}');
          if (file[line_index] == ':') {
            int string_end = -1;
            for (int end_candidate = line_index - 1; end_candidate > line_start;
                 --end_candidate) {
              if ((string_index[end_candidate / 64] &
                   (1L << (end_candidate % 64))) != 0) {
                string_end = end_candidate;
                break;
              }
            }

            const int string_start = string_end - key_len;
            if (string_start < line_start || file[string_start] != '"') {
              ++line_index;
              line_index = find_next_structural_char(
                  leveled_bitmaps_index, level_end[current_level], line_index,
                  current_level, level_size);
              assert(file[line_index] == ',' || file[line_index] == '}');
              if (file[line_index] == '}') {
                goto next_line;
              }
              goto search_key;
            }

            for (int i = 0; i < key_len; ++i) {
              if (key[i] !=
                  static_cast<unsigned char>(file[string_start + i + 1])) {
                ++line_index;
                line_index = find_next_structural_char(
                    leveled_bitmaps_index, level_end[current_level], line_index,
                    current_level, level_size);
                assert(file[line_index] == ',' || file[line_index] == '}');
                if (file[line_index] == '}') {
                  goto next_line;
                }
                goto search_key;
              }
            }
          }
          ++line_index;
        } else {
          goto next_line;
        }
        break;

      case OPCODE_MOVE_TO_INDEX:
        target_index = read_varint(query, &query_pos);

        if (file[line_index] == '[' || file[line_index] == ',' ||
            file[line_index] == ']') {
          if (file[line_index] == '[') {
            curr_index[current_level] = 0;
          } else {
            ++curr_index[current_level];
          }

        search_index:
          if (curr_index[current_level] < target_index) {
            ++line_index;
            line_index = find_next_structural_char(
                leveled_bitmaps_index, level_end[current_level], line_index,
                current_level, level_size);
            assert(file[line_index] == ',' || file[line_index] == ']');
            if (file[line_index] == ',') {
              ++curr_index[current_level];
              goto search_index;
            }
            goto next_line;
          }

          if (curr_index[current_level] > target_index) {
            --line_index;
            line_index = find_previous_structural_char(
                leveled_bitmaps_index, 0, line_index, current_level,
                level_size);
            assert(file[line_index] == ',' || file[line_index] == '[');
            --curr_index[current_level];
            goto search_index;
          }

          if (file[line_index] == ']') {
            goto next_line;
          }
          ++line_index;
        } else {
          goto next_line;
        }
        break;

      case OPCODE_MOVE_TO_INDEX_REVERSE:
        target_index = read_varint(query, &query_pos);

        line_index = level_end[current_level] - 1;
        while (file[line_index] == ' ') {
          --line_index;
        }

        if (file[line_index] == ']' || file[line_index] == ',' ||
            file[line_index] == '[') {
          if (file[line_index] == ']') {
            curr_index[current_level] = 0;
          } else {
            ++curr_index[current_level];
          }

        search_index_reverse:
          if (curr_index[current_level] < target_index) {
            --line_index;
            line_index = find_previous_structural_char(
                leveled_bitmaps_index, 0, line_index, current_level,
                level_size);
            assert(file[line_index] == ',' || file[line_index] == '[');
            if (file[line_index] == ',' ||
                curr_index[current_level] + 1 == target_index) {
              ++curr_index[current_level];
              goto search_index_reverse;
            }
            goto next_line;
          }

          if (curr_index[current_level] > target_index) {
            ++line_index;
            line_index = find_next_structural_char(
                leveled_bitmaps_index, level_end[current_level], line_index,
                current_level, level_size);
            assert(file[line_index] == ',' || file[line_index] == ']');
            --curr_index[current_level];
            goto search_index_reverse;
          }

          ++line_index;
        } else {
          goto next_line;
        }
        break;

      case OPCODE_MARK_POSITION:
        assert(marked_pos_index + 1 < MAX_NUM_LEVELS);
        ++marked_pos_index;
        marked_pos[marked_pos_index] = line_index;
        marked_pos_level[marked_pos_index] = current_level;
        break;

      case OPCODE_RESET_POSITION:
        assert(marked_pos_index >= 0);
        line_index = marked_pos[marked_pos_index];
        current_level = marked_pos_level[marked_pos_index];
        --marked_pos_index;
        for (int i = current_level + 1; i < MAX_NUM_LEVELS; ++i) {
          if (level_end[i] == -1) {
            break;
          }
          level_end[i] = -1;
        }
        break;

      case OPCODE_EXPRESSION_STRING_EQUALS:
        key_len = read_varint(query, &query_pos);
        key = query + query_pos;
        query_pos += key_len;

        while (file[line_index] == ' ' &&
               line_index < level_end[current_level]) {
          ++line_index;
        }

        {
          int string_end = level_end[current_level] - 1;
          while (line_index < string_end && file[string_end] == ' ') {
            --string_end;
          }
          assert(file[line_index] == '"' && file[string_end] == '"');

          const int string_length = string_end - line_index + 1;
          if (string_length != key_len) {
            goto next_line;
          }

          for (int i = 0; i < key_len; ++i) {
            if (key[i] != static_cast<unsigned char>(file[line_index + i])) {
              goto next_line;
            }
          }
        }
        break;

      default:
        assert(false);
        break;
      }
    }

  next_line:;
  }
}

} // namespace gpjson::query::kernels::orig
