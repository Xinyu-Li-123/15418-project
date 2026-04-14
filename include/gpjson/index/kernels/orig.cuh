namespace gpjson::index::kernels::orig {
__global__ void newline_count_index(const char *file, int fileSize,
                                    int *newlineCountIndex);

__global__ void newline_index(const char *file, int fileSize,
                              int *newlineCountIndex, long *newlineIndex);
} // namespace gpjson::index::kernels::orig
