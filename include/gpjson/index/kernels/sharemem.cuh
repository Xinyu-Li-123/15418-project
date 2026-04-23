namespace gpjson::index::kernels::sharemem {

__global__ void newline_count_index(const char *file, int fileSize,
                                    int *newlineCountIndex);

__global__ void combined_escape_carry_newline_count_index(const char *file,
                                                          int file_size,
                                                          int num_chunks,
                                                          char *escape_carry,
                                                          int *newline_count);

__global__ void combined_escape_newline_index(const char *file, int file_size,
                                              const char *escape_carry_index,
                                              const int *newline_count_index,
                                              long *escape_index,
                                              long *newline_index,
                                              int num_chunks);

__global__ void int_sum_pre_scan(int *intArr, int n);

__global__ void int_sum_post_scan(int *intArr, int n, int stride,
                                  int startingValue, int *base);

__global__ void int_sum_rebase(int *intArr, int n, int *base, int offset,
                               int *intNewArr);

__global__ void newline_index(const char *file, int fileSize,
                              int *newlineCountIndex, long *newlineIndex);

__global__ void escape_carry_index(const char *file, int fileSize,
                                   char *escapeCarryIndex);

__global__ void escape_index(const char *file, long fileSize,
                             char *escapeCarryIndex, long *escapeIndex);

__global__ void quote_index(const char *file, int file_size,
                            const long *escape_index, long *quote_index,
                            char *quote_carry_index, int num_chunks);

__global__ void xor_pre_scan(char *charArr, int n);

__global__ void xor_post_scan(char *charArr, int n, int stride, char *base);

__global__ void xor_rebase(char *charArr, int n, char *base);

__global__ void string_index(long *string_index, int string_index_size,
                             const char *quote_carry_index, int file_size,
                             int num_chunks);

__global__ void leveled_bitmaps_carry_index(const char *file, int file_size,
                                            const long *string_index,
                                            char *leveled_bitmaps_aux_index,
                                            int num_chunks);

__global__ void leveled_bitmaps_index(const char *file, int file_size,
                                      const long *string_index,
                                      const char *leveled_bitmaps_aux_index,
                                      long *leveled_bitmaps_index,
                                      int level_size, int num_levels,
                                      int num_chunks);

__global__ void char_sum_pre_scan(char *charArr, int n);

__global__ void char_sum_post_scan(char *charArr, int n, int stride,
                                   char startingValue, char *base);

__global__ void char_sum_rebase(char *charArr, int n, char *base, int offset,
                                char *charNewArr);
} // namespace gpjson::index::kernels::sharemem
