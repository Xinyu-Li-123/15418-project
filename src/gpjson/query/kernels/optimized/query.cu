#include "gpjson/query/kernels/optimized.hpp"

#include <assert.h>
#include <cstdint>

namespace gpjson::query::kernels::optimized {

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

__device__ __forceinline__ uint64_t mask_through_bit(const int bit_offset) {
  return bit_offset == 63 ? ~uint64_t{0}
                          : ((uint64_t{1} << (bit_offset + 1)) - 1);
}

__device__ __forceinline__ int find_previous_bitmap_bit(const long *bitmap,
                                                        int min_index,
                                                        int line_index) {
  if (line_index < min_index) {
    return -1;
  }

  int word_index = line_index / 64;
  int word_base = word_index * 64;
  uint64_t word = static_cast<uint64_t>(bitmap[word_index]);
  word &= mask_through_bit(line_index - word_base);

  while (word_base + 63 >= min_index) {
    uint64_t candidate = word;
    const int first_bit = min_index - word_base;
    if (first_bit > 0) {
      candidate &= ~mask_through_bit(first_bit - 1);
    }


    // __clzll returns the number of zeros before the first set bit, so 63 - __clzll returns the index of the last set bit.
    if (candidate != 0) {
      return word_base + 63 -
             __clzll(static_cast<unsigned long long>(candidate));
    }

    if (word_index == 0) {
      break;
    }
    --word_index;
    word_base -= 64;
    if (word_base + 63 < min_index) {
      break;
    }
    word = static_cast<uint64_t>(bitmap[word_index]);
  }

  return -1;
}

__device__ __forceinline__ int
find_next_structural_char(const long *ext_index, int level_end, int line_index,
                          int current_level, int level_size) {
  if (line_index >= level_end) {
    return level_end;
  }

  const long *level_index = ext_index + level_size * current_level;
  int word_index = line_index / 64;
  int word_base = word_index * 64;
  uint64_t word = static_cast<uint64_t>(level_index[word_index]);
  word &= ~uint64_t{0} << (line_index - word_base);

  while (word_base <= level_end) {
    uint64_t candidate = word;
    const int last_bit = level_end - word_base;
    if (last_bit < 63) {
      //keep only bits through last_bit(this level)
      candidate &= mask_through_bit(last_bit);
    }

    // __ffsll returns the index of the first set bit.
    if (candidate != 0) {
      return word_base +
             __ffsll(static_cast<long long>(candidate)) - 1;
    }

    ++word_index;
    word_base += 64;
    if (word_base > level_end) {
      break;
    }
    word = static_cast<uint64_t>(level_index[word_index]);
  }

  return level_end;
}

__device__ __forceinline__ int find_previous_structural_char(
    const long *ext_index, int level_start, int line_index, int current_level,
    int level_size) {
  if (line_index <= level_start) {
    return level_start;
  }

  const long *level_index = ext_index + level_size * current_level;
  const int structural_index =
      find_previous_bitmap_bit(level_index, level_start, line_index);
  return structural_index == -1 ? level_start : structural_index;
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

  for (int file_index = thread_index; file_index < newline_index_size;
       file_index += stride) {
    const int result_base = file_index * 2 * num_results;
    for (int i = 0; i < 2 * num_results; ++i) {
      result[result_base + i] = -1;
    }

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
    int key_pos = 0;
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
          level_end[current_level] = find_next_structural_char(
              leveled_bitmaps_index, level_end[current_level - 1],
              line_index + 1, current_level - 1, level_size);
          assert(level_end[current_level] != -1);
          while (file[line_index] == ' ') {
            ++line_index;
          }
        }
        break;

      case OPCODE_MOVE_TO_KEY:
        key_len = read_varint(query, &query_pos);
        key_pos = query_pos;
        query_pos += key_len;

        if (file[line_index] == '{') {
        search_key:
          ++line_index;
          line_index = find_next_structural_char(
              leveled_bitmaps_index, level_end[current_level], line_index,
              current_level, level_size);
          assert(file[line_index] == ':' || file[line_index] == '}');
          if (file[line_index] == ':') {
            const int string_end = find_previous_bitmap_bit(
                string_index, line_start + 1, line_index - 1);

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
              if (query[key_pos + i] !=
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
        key_pos = query_pos;
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
            if (query[key_pos + i] !=
                static_cast<unsigned char>(file[line_index + i])) {
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

} // namespace gpjson::query::kernels::optimized
