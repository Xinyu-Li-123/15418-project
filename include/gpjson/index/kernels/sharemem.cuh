#include <cstddef>

namespace gpjson::index::kernels::sharemem {

__global__ void newline_count_index(const char *file, size_t fileSize,
                                    int *perTileNewlineCountIndex);

__global__ void combined_escape_carry_newline_count_index(const char *file,
                                                          size_t fileSize,
                                                          char *escapeCarry,
                                                          int *newlineCount);

__global__ void int_sum_pre_scan(int *intArr, int n);

__global__ void int_sum_post_scan(int *intArr, int n, int stride,
                                  int startingValue, int *base);

__global__ void int_sum_rebase(int *intArr, int n, int *base, int offset,
                               int *intNewArr);

__global__ void newline_index(const char *file, size_t fileSize,
                              int *perTileNewlineCountIndex,
                              long *newlineIndex);

__global__ void combined_escape_newline_index(const char *file, size_t fileSize,
                                              char *escapeCarryIndex,
                                              int *newlineCountIndex,
                                              long *escapeIndex,
                                              long *newlineIndex);

__global__ void escape_carry_index(const char *file, size_t fileSize,
                                   char *escapeCarryIndex);

__global__ void escape_index(const char *file, size_t fileSize,
                             char *escapeCarryIndex, long *escapeIndex);

__global__ void quote_index(const char *file, size_t fileSize, long *escapeIndex,
                            long *quoteIndex, char *quoteCarryIndex);

__global__ void xor_pre_scan(char *charArr, int n);

__global__ void xor_post_scan(char *charArr, int n, int stride, char *base);

__global__ void xor_rebase(char *charArr, int n, char *base);

__global__ void string_index(long *stringIndex, int stringIndexSize,
                             char *quoteCarryIndex);

__global__ void leveled_bitmaps_carry_index(const char *file, size_t fileSize,
                                            const long *stringIndex,
                                            char *leveledBitmapsAuxIndex);

__global__ void char_sum_pre_scan(char *charArr, int n);

__global__ void char_sum_post_scan(char *charArr, int n, int stride,
                                   char startingValue, char *base);

__global__ void char_sum_rebase(char *charArr, int n, char *base, int offset,
                                char *charNewArr);

__global__ void leveled_bitmaps_index(const char *file, size_t fileSize,
                                      const long *stringIndex,
                                      char *leveledBitmapsAuxIndex,
                                      long *leveledBitmapsIndex, int levelSize,
                                      int numLevels);
} // namespace gpjson::index::kernels::sharemem
