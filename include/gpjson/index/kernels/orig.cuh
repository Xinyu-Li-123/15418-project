namespace gpjson::index::kernels::orig {
__global__ void newline_count_index(const char *file, int fileSize,
                                    int *newlineCountIndex);

__global__ void int_sum_pre_scan(int *intArr, int n);

__global__ void int_sum_post_scan(int *intArr, int n, int stride,
                                  int startingValue, int *base);

__global__ void int_sum_rebase(int *intArr, int n, int *base, int offset,
                               int *intNewArr);

__global__ void newline_index(const char *file, int fileSize,
                              int *newlineCountIndex, long *newlineIndex);
} // namespace gpjson::index::kernels::orig
